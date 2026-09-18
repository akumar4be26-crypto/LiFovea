"""Configuration objects for the AVR-2.5D mapping stack.

Everything that the pipeline can be tuned with lives here as a frozen-ish
dataclass so that a run is fully described by a single ``PipelineConfig``
instance (and therefore reproducible and serialisable to JSON).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Sequence, Tuple

# --------------------------------------------------------------------------
# Semantic taxonomy
# --------------------------------------------------------------------------
# The problem statement asks for three analyses: terrain (drivable vs not),
# static obstacles and dynamic objects.  That maps onto four point classes.

CLASS_DRIVABLE = 0        # smooth, load-bearing ground the vehicle can drive on
CLASS_ROUGH_TERRAIN = 1   # curb, pothole lip, verge, gravel, steep bank
CLASS_STATIC_OBSTACLE = 2 # wall, building, pole, tree trunk, barrier
CLASS_DYNAMIC_OBJECT = 3  # pedestrian, cyclist, moving/parked vehicle

NUM_CLASSES = 4

CLASS_NAMES: Tuple[str, ...] = (
    "drivable",
    "rough_terrain",
    "static_obstacle",
    "dynamic_object",
)

# RGB used consistently by the CLI exporter, the web dashboard and the docs.
CLASS_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (46, 160, 132),    # teal      - drivable
    (214, 158, 46),    # amber     - rough terrain
    (120, 132, 168),   # slate     - static obstacle
    (219, 84, 97),     # red       - dynamic object
)


# --------------------------------------------------------------------------
# Sensor
# --------------------------------------------------------------------------
@dataclass
class SensorConfig:
    """A spinning multi-beam LiDAR, parameterised like a Velodyne HDL-64E."""

    n_beams: int = 64
    fov_up_deg: float = 2.0
    fov_down_deg: float = -24.9
    azimuth_steps: int = 1024          # columns per full revolution
    max_range: float = 100.0
    min_range: float = 1.2
    #: height of the sensor above the vehicle's contact plane
    mount_height: float = 1.73
    #: range noise model  sigma(d) = range_sigma0 + range_sigma_rel * d
    range_sigma0: float = 0.015
    range_sigma_rel: float = 0.0012
    #: probability that a return is lost entirely (dark / specular surfaces)
    dropout: float = 0.015
    rpm: float = 600.0                 # 10 Hz

    @property
    def frame_period(self) -> float:
        return 60.0 / self.rpm


# --------------------------------------------------------------------------
# Variable resolution grid
# --------------------------------------------------------------------------
@dataclass
class LodConfig:
    """Level-of-detail schedule for the foveated 2.5D grid.

    Level ``L`` has cell size ``base_cell * 2**L``.  Powers of two are not a
    stylistic choice: they are what makes the multi-resolution grid a strict
    quadtree, so every coarse cell is *exactly* four finer cells and no cell
    boundary at any level can ever cut across a boundary at another level.
    That is the property that removes alignment error by construction.

    ``ring_radii[L]`` is the minimum distance-from-sensor at which a cell is
    allowed to be represented at level ``L``.  The test is applied to the
    *nearest corner of the cell*, not to individual points, so every point
    inside a given cell always resolves to the same level.
    """

    base_cell: float = 0.05                                   # 5 cm
    n_levels: int = 5                                         # 5,10,20,40,80 cm
    ring_radii: Tuple[float, ...] = (0.0, 10.0, 20.0, 40.0, 70.0)
    max_range: float = 100.0

    def cell_size(self, level: int) -> float:
        return self.base_cell * (2 ** level)

    @property
    def cell_sizes(self) -> Tuple[float, ...]:
        return tuple(self.cell_size(l) for l in range(self.n_levels))

    def validate(self) -> None:
        if len(self.ring_radii) != self.n_levels:
            raise ValueError(
                f"ring_radii has {len(self.ring_radii)} entries but n_levels={self.n_levels}"
            )
        if self.ring_radii[0] != 0.0:
            raise ValueError("ring_radii[0] must be 0.0 (finest level is the fallback)")
        if any(b < a for a, b in zip(self.ring_radii, self.ring_radii[1:])):
            raise ValueError("ring_radii must be non-decreasing")
        if self.base_cell <= 0:
            raise ValueError("base_cell must be positive")


# --------------------------------------------------------------------------
# Elevation fusion
# --------------------------------------------------------------------------
@dataclass
class FusionConfig:
    """Range-dependent Bayesian elevation filter parameters."""

    #: elevation measurement noise  sigma_z(d)^2 = (z_sigma0 + z_sigma_rel*d)^2
    z_sigma0: float = 0.02
    z_sigma_rel: float = 0.004
    #: extra variance charged to a cell because the cell is coarse: a 80 cm
    #: cell genuinely contains more real height variation than a 5 cm one.
    quantisation_gain: float = 0.35
    #: process noise added per frame so the map can follow a changing world
    process_var: float = 0.0009
    #: variance floor, keeps the filter responsive
    min_var: float = 1e-4
    #: cells not observed for this many frames are dropped from the map
    stale_frames: int = 25


# --------------------------------------------------------------------------
# Traversability / vehicle model
# --------------------------------------------------------------------------
@dataclass
class VehicleConfig:
    max_slope_deg: float = 22.0
    max_step_height: float = 0.10       # tallest step the platform can climb
    max_roughness: float = 0.06         # std of elevation inside a cell
    clearance_height: float = 2.4       # anything higher than this is overhead
    body_radius: float = 0.9            # for obstacle inflation
    #: dynamic objects get an extra safety halo
    dynamic_inflation: float = 1.6
    #: tile size of the navigation layer used for reachability and planning
    nav_cell: float = 0.8


# --------------------------------------------------------------------------
# Model / inference
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    backend: str = "auto"               # auto | torch | numpy
    architecture: str = "sparsevoxel"   # pointnet2lite | sparsevoxel | geometric
    checkpoint: str = "checkpoints/avr25d.pt"
    device: str = "mps"                 # cpu | mps | cuda
    #: points are chunked before being pushed through the network
    chunk_size: int = 32768
    #: voxel size used by the sparse-voxel backbone
    voxel_size: float = 0.20
    #: neighbourhood radius for the shared hand-crafted feature stage
    feature_radius: float = 0.55
    #: near zone retained as full 3D input for deep-learning inference
    foveal_radius: float = 15.0
    #: transition zone ends here; its lightweight inference pass uses a stride
    transition_radius: float = 30.0
    #: powerful transition pass: retain every point for deep-learning inference
    transition_stride: int = 1
    #: maximum-density far-zone inference for strongest obstacle recall
    far_stride: int = 1


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    sensor: SensorConfig = field(default_factory=SensorConfig)
    lod: LodConfig = field(default_factory=LodConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 7

    def validate(self) -> "PipelineConfig":
        self.lod.validate()
        if self.lod.max_range > self.sensor.max_range:
            raise ValueError("grid max_range exceeds sensor max_range")
        return self

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict) -> "PipelineConfig":
        return cls(
            sensor=SensorConfig(**d.get("sensor", {})),
            lod=LodConfig(**{**d.get("lod", {}),
                             "ring_radii": tuple(d.get("lod", {}).get("ring_radii",
                                                LodConfig().ring_radii))}),
            fusion=FusionConfig(**d.get("fusion", {})),
            vehicle=VehicleConfig(**d.get("vehicle", {})),
            model=ModelConfig(**d.get("model", {})),
            seed=d.get("seed", 7),
        ).validate()


def uniform_cell_count(radius: float, cell: float) -> int:
    """Cells a *uniform* grid of resolution ``cell`` needs to cover a square
    of half-width ``radius`` - the baseline every memory claim is measured
    against."""
    side = int(round(2.0 * radius / cell))
    return side * side
