# AVR-2.5D — Adaptive Variable-Resolution 2.5D Lidar Mapping

A perception stack that turns raw spinning-lidar point clouds into a **foveated
2.5D map**: 5 cm cells inside the braking distance, 80 cm cells at the horizon,
with semantics, elevation uncertainty and traversability on every cell.

Built for the DRDO problem statement *"Adaptive Variable Resolution 2.5D Lidar
Mapping for Dynamic Environment Perception"* (Smart Vehicles).

```
scan ─▶ features ─▶ semantic segmentation ─▶ variable-resolution projection
                                                     │
                            Bayesian elevation fusion ┼─▶ 2.5D map ─▶ traversability ─▶ A*
                                                     │
                                          navigation layer (0.8 m tiles)
```

Measured end to end on a held-out 20-frame drive, on **two CPU cores, no GPU**:

| | |
|---|---|
| point accuracy / mIoU | **95.4 %** / **0.827** |
| map size | **9.9 MB** vs 960 MB for a uniform 5 cm grid — **97× smaller** |
| elevation error, 0–10 m | **10 mm** (P95 absolute) |
| elevation error, 70–100 m | 147 mm (P95 absolute) |
| latency | 230 ms/frame (4.4 fps) with the network, 176 ms (5.7 fps) without |
| structural audit | 0 alignment error, 0 cross-level overlaps, 0 ring violations |

---

## Quick start

```bash
pip install -r requirements.txt          # numpy is the only hard requirement
python -m avr25d.cli demo                # run the pipeline, print per-frame stats
python -m avr25d.cli serve --open        # interactive dashboard on :8080
python -m avr25d.cli bench --frames 20 --report BENCHMARK.md
python -m unittest discover -s tests     # 44 tests, ~12 s
```

No dataset download and no annotation: the built-in simulator ray-casts a
64-beam lidar through a procedural urban scene and hands back exact per-point
ground truth, which is what makes every number above a measurement rather than
an estimate.

To train the network (≈100 s on CPU):

```bash
python -m avr25d.cli train --epochs 14   # writes checkpoints/avr25d.pt
```

### Real-world fine-tuning

The trainer also accepts labeled real-world scans exported as `.npz` files.
Each archive must contain `xyz` with shape `(N, 3)` and integer `label` with
shape `(N,)`, using the four project classes `0..3`; `intensity` is optional.
Pass a file or directory of scans for mixed simulator + real training, and a
separate held-out directory for validation:

```bash
python -m avr25d.cli train --real-data data/real/train \
  --real-val-data data/real/val --epochs 14
```

The simulator remains in the training mix by default, preserving controlled
coverage for rare cases while real scans improve sensor and scene realism.

Without a checkpoint the stack falls back to a deterministic geometric
segmenter and keeps running — 91.7 % accuracy instead of 95.4 %.

---

## The problem, and why the obvious answers fail

A 3D point cloud at 10 Hz is ~60 k points per revolution and millions per
second. Two standard responses both lose:

* **Keep the full 3D cloud.** A dense 5 cm voxel volume over a 100 m radius and
  20 m of height is 800 MB of occupancy *bits* before you store anything useful
  in a cell. It does not fit in a vehicle's cache hierarchy, let alone its
  latency budget.
* **Flatten to a 2D occupancy grid.** Now a 16 cm curb and a 5 m wall are the
  same symbol, a pothole is invisible, and a gantry 4.6 m overhead blocks the
  road it spans.

The third answer is to spend resolution where it buys safety. A vehicle at
15 m/s needs centimetres under its wheels and metres at the horizon — the same
trade a retina makes. That is this project.

---

## The variable-resolution grid

### Powers of two, or nothing

Level `L` has cell size `base_cell · 2^L` — 5, 10, 20, 40, 80 cm. This is not
cosmetic. It makes the map a strict **quadtree**: a level-`L` cell is *exactly*
four level-`(L−1)` cells, so every boundary at a coarse level is also a boundary
at every finer one. Two properties follow for free, and the problem statement
asks for both:

1. **No alignment error.** There is no resampling step between resolutions, so
   there is nowhere for a half-cell offset to appear.
2. **No data loss at a ring boundary.**

### Level is a property of the cell, not the point

The naive scheme — *assign each point a resolution from its own range* — breaks
exactly at the ring boundaries. Two points 1 cm apart, straddling `r = 10 m`,
land in different grids and the cell that contains both is split between two
representations.

So the test is applied to the **nearest corner of the candidate cell**:

```
d_min(c) = distance from the sensor to the closest corner of cell c
leaf(x) = the coarsest level L such that d_min(cell_L(x)) ≥ R_L,
          descending from L_max, falling back to level 0
```

`d_min` is a property of the cell, so every point inside a cell resolves to the
same level, and a cell that straddles a ring boundary **refines** to the finer
level rather than being cut in half. Conservative in the safe direction, and
exact.

`tests/test_grid.py::test_level_is_a_property_of_the_cell_not_the_point`
verifies this by jittering 12 000 samples anywhere inside their own leaf — half
of them deliberately placed on the ring boundaries — and asserting the level
never changes.

### The map moves with the vehicle

The grid is anchored to the **world**, not the sensor, so elevation estimates
persist as the vehicle drives. But a cell's correct level changes as the range
to it changes, so every frame `rebalance()` migrates cells:

* **Coarsening** (cell falls behind): four children fuse by inverse-variance
  weighting — information-preserving, `1/σ² = Σ 1/σᵢ²`.
* **Refinement** (cell comes into the near field): the parent seeds all four
  children and each child's variance is inflated by the parent's own vertical
  spread, because a parent genuinely does not know how its children differ.

`test_migration_conserves_the_surface` asserts the mapped elevation does not
move when cells migrate.

### What a cell holds

48 bytes: elevation, variance, ground z-extent, intensity, observation count,
four-class evidence, overhead clearance, last-seen frame.

### Elevation fusion

A scalar Kalman filter per cell, with a range-dependent measurement model:

```
σ_z(d)  = σ₀ + σ_rel · d                    (sensor noise grows with range)
R       = σ_z² + γ · var_in_cell            (a coarse cell really does contain
                                             more height variation)
R      /= n_ground                          (averaging helps)
K       = P⁻ / (P⁻ + R)
z⁺      = z⁻ + K (z_meas − z⁻)
P⁺      = (1 − K) P⁻ ,  P⁻ = P + q
```

The `γ · var_in_cell` term is what stops a 80 cm cell from claiming millimetre
precision for ground it has averaged over.

---

## Perception

### Feature stage (shared by both backends)

13 per-point features from a voxel hash — no neighbour search, all `bincount`
and `reduceat`, ~16 ms for 59 k points: height above the column minimum, column
span, local density, the three eigenvalue shape descriptors (linearity,
planarity, scattering) from per-voxel covariance, surface normal verticality,
roughness, range and intensity.

### `pointnet2lite` (default)

PointNet++ in the shape that matters: shared point-wise MLPs, symmetric
max-pooling inside local neighbourhoods, two scales (0.6 m and 2.4 m), a global
descriptor, and feature propagation back to every point. Neighbourhoods come
from a voxel hash rather than farthest-point sampling + ball query — the same
grouping semantics, deterministic, O(N), and no custom CUDA kernel to ship.
168 k parameters.

### `sparsevoxel`

The sparse-convolution alternative: pillar encoder → scatter into a BEV feature
map → dilated 2D CNN → gather back onto points. This is the backbone that sees
*context* — a flat patch surrounded by road is road, the same patch surrounded
by wall is a pavement.

### `geometric` (the fallback)

Pure NumPy, no checkpoint, fully deterministic: lower-envelope ground surface,
in-cell relief and slope for the terrain split, BEV connected components with
shape priors for the dynamic/static split. It exists so the pipeline degrades
instead of failing when PyTorch or a GPU is missing, and it is a real
classifier: 91.7 % end to end.

### Training

Supervision is free because the simulator knows the answer. Six procedural
worlds for training, two unseen worlds for validation, ~100 s on CPU.

---

## From map to decision

### Terrain analysis is two different questions

*Is this patch flat?* is local, and the point network answers it. *Is this
patch drivable?* is not: a sidewalk is perfectly flat, and it is not drivable
because a 16 cm curb stands between it and the vehicle. That is a
**connectivity** question about the map, and only the map can answer it.

So reachability runs on a **navigation layer**: the variable-resolution map
aggregated into uniform 0.8 m tiles carrying representative elevation, internal
relief (which is what a curb looks like once you stop resolving it), and whether
anything solid intrudes below the vehicle's roof. A few thousand nodes instead
of a hundred thousand, and — crucially — tiles large enough to be *connected*.
At 5 cm a single revolution leaves the ground as disconnected islands and any
reachability test returns "nowhere".

Three details earn their place:

* **Links probe across the radial sampling gap.** A spinning lidar samples
  densely in azimuth and sparsely in range, so strict 4-adjacency fails at
  30 m. Tiles link to the first observed tile in each direction.
* **Connectivity is only asserted across short links.** A curb crossed
  obliquely over 3 m of unobserved ground looks like a 2° ramp. Links longer
  than two tiles inform slope but never grant reachability.
* **Past the horizon, "not reached" means "not known".** Outside ~45 m the
  sampling no longer supports a connectivity verdict, so the map declines to
  give one instead of guessing. The grid may only ever *revoke* a point's
  drivability, never grant it — demotion is the safe direction, and it is the
  only direction the pipeline allows (`test_grid_refinement_only_revokes_drivability`).

### Height decides, not footprint

A gantry at 4.6 m and a wall paint the same footprint in plan view. Here an
obstacle blocks a tile only if part of it hangs below the vehicle's clearance
*above that tile's own ground*. Everything else goes into the overhead-clearance
layer, and the road underneath stays drivable. This is the single clearest thing
2.5D buys over 2D.

### Planning

A* over the navigation tiles: arc length scaled by tile roughness plus an
explicit climb penalty, refusing any edge above the platform's step height. The
planner is the honest test of the map — if the traversability layer is wrong,
the route drives over a curb. `python -m avr25d.cli plan --ahead 40`.

---

## Results

20 frames, seed 7, `torch:pointnet2lite`, 2 vCPU. Reproduce with
`python -m avr25d.cli bench --frames 20 --report BENCHMARK.md`.

### Semantic accuracy

| class | IoU | recall | precision |
|---|---:|---:|---:|
| drivable | 0.905 | 0.909 | 0.996 |
| rough terrain | 0.885 | 0.992 | 0.891 |
| static obstacle | 0.970 | 0.992 | 0.977 |
| dynamic object | 0.548 | 0.582 | 0.904 |
| **overall** | **mIoU 0.827** | **acc 0.954** | |

Network-only output is 96.4 % / 0.840 mIoU; the map's demote-only refinement
trades 1 point of raw accuracy for the guarantee that nothing is called drivable
that the vehicle cannot actually reach.

### By range

| band | points | accuracy | mIoU |
|---|---:|---:|---:|
| 0–10 m | 822 872 | 0.953 | 0.840 |
| 10–20 m | 186 864 | 0.973 | 0.700 |
| 20–40 m | 38 342 | 0.884 | 0.709 |
| 40–70 m | 9 379 | 0.908 | 0.796 |
| 70–100 m | 3 270 | 0.926 | 0.695 |

Accuracy holds at range because obstacles stay separable; what degrades is the
terrain split, which needs resolution the far field no longer has — which is the
design working as intended, not in spite of itself.

### Elevation error against analytic ground truth

| band | median cell | MAE | P95 abs |
|---|---:|---:|---:|
| 0–10 m | 5 cm | 4.3 mm | 10 mm |
| 10–20 m | 10 cm | 5.6 mm | 22 mm |
| 20–40 m | 20 cm | 9.3 mm | 38 mm |
| 40–70 m | 40 cm | 42 mm | 77 mm |
| 70–100 m | 80 cm | 54 mm | 147 mm |

This is the whole trade in one table: millimetres where the wheels are about to
be, decimetres at the horizon, for a map two orders of magnitude smaller.

### Memory

| representation | size | ratio |
|---|---:|---:|
| **adaptive 2.5D map (165 k cells)** | **9.9 MB** | **1×** |
| uniform 5 cm 2.5D grid | 960 MB | 97× |
| dense 5 cm × 20 m voxel occupancy (bits only) | 800 MB | 81× |
| uniform 80 cm grid (and no curbs, no potholes) | 3.8 MB | 0.4× |

The last row is the honest comparison: a coarse uniform grid *is* smaller. It
also cannot see a curb. The point of the adaptive grid is that it costs about
2.5× a useless map and 1 % of a usable uniform one.

### Latency

| stage | mean ms | p95 ms |
|---|---:|---:|
| features | 16.1 | 17.5 |
| segmentation | 118.4 | 130.1 |
| projection + fusion | 56.5 | 66.4 |
| traversability | 38.8 | 48.8 |
| **total** | **229.8** | **259.1** |

4.4 fps on two CPU cores with no GPU and no batching. Segmentation is half the
budget and is the only stage that is GPU-bound; the map itself runs at ~9 fps on
this hardware, and the geometric backend gives 5.7 fps end to end today.

### Structural integrity

| invariant | measured | required |
|---|---:|---:|
| points resolving to exactly one leaf | 100 % | 100 % |
| leaf level matches lookup | 100 % | 100 % |
| cells stored at two levels | 0 | 0 |
| cells inside the wrong ring | 0 | 0 |
| projection alignment error | 0 m | 0 m |

---

## Dashboard

`python -m avr25d.cli serve --open` runs a survey-chart style dashboard that
**re-derives the grid in the browser** from the classified point cloud — so the
resolution-schedule sliders rebuild the map live rather than paging through
pre-rendered images. Layers: elevation (hypsometric), semantics, traversability,
cell size, in-cell relief. Click the plot to plan a route. The same page is what
gets published as a shareable artifact.

---

## Repository layout

```
avr25d/
  config.py              every tunable, as one serialisable dataclass tree
  pipeline.py            scan → semantics → projection → fusion → traversability
  cli.py                 demo / bench / train / export / serve / plan
  sim/
    world.py             procedural urban scene: curbs, potholes, gantries, traffic
    lidar.py             vectorised ray-cast 64-beam simulator with ground truth
  models/
    features.py          13 shared per-point features, O(N), NumPy only
    geometric.py         deterministic fallback segmenter
    nets.py              PointNet2Lite and SparseVoxelNet
    train.py             training on simulator-generated supervision
    infer.py             backend selection, foveated inference, timing
  grid/
    vrgrid.py            the quadtree map: projection, Kalman fusion, rebalancing
    navigation.py        uniform navigation tiles, links, reachability
    traversability.py    slope / step / relief / clearance / terrain verdict
  planner/astar.py       2.5D A* over the navigation layer
  metrics/evaluate.py    accuracy, elevation error, latency, integrity audit
  server/app.py          stdlib HTTP server for the dashboard
web/dashboard.html       the dashboard (template + in-browser grid engine)
tools/                   demo payload export, artifact build
tests/                   44 tests
docs/BENCHMARK.md        generated measurement report
```

## Configuration

Everything is one `PipelineConfig`, serialisable to JSON:

```python
from avr25d.config import PipelineConfig
from avr25d.pipeline import Pipeline

cfg = PipelineConfig()
cfg.lod.base_cell = 0.05
cfg.lod.ring_radii = (0.0, 10.0, 20.0, 40.0, 70.0)   # metres, per level
cfg.vehicle.max_step_height = 0.10                    # curb the platform can climb
cfg.vehicle.clearance_height = 2.4                    # what counts as overhead
pipe = Pipeline(cfg)
```

Or from the command line: `--base-cell 0.10 --rings 0,8,16,32,64`.

---

## Requirements coverage

| asked for | where |
|---|---|
| deep model segmenting terrain / static / dynamic | `models/nets.py`, `models/train.py` — PointNet++-style and sparse-voxel, 0.827 mIoU |
| variable-resolution grid engine, 5 cm near → coarse far | `grid/vrgrid.py` — 5 cm inside 10 m, 80 cm past 70 m, 100 m radius |
| no alignment errors or data loss in the 3D→2.5D projection | cell-level LOD decision + power-of-two quadtree; audited in `metrics/evaluate.py` and `tests/test_grid.py` |
| real-time visualisation, colour-coded | `web/dashboard.html`, live re-gridding in the browser |
| demonstrated memory reduction | 97× vs uniform 5 cm, measured per frame and shown live |
| low latency + accuracy across distances | latency and per-band accuracy tables above |
| curbs, potholes, overhanging obstacles | relief layer, pothole-aware terrain split, overhead-clearance layer |

## Limitations, honestly

* **Pedestrian recall is 0.58.** Legs occupy the same 20 cm of height as the
  road they stand on; without temporal association the network cannot always
  separate them. Instance tracking across frames is the fix and is not here.
* **Ego pose is taken as known.** The simulator supplies it. On a real vehicle
  this is odometry or a scan-matcher, and the map's persistence is only as good
  as that pose.
* **Single sensor, no calibration story.** Multi-lidar extrinsics and
  motion-distortion (deskew) compensation are not implemented.
* **Foveated inference is currently a no-op.** The knob exists
  (`ModelConfig.far_stride`), but 78 % of returns from this sensor geometry fall
  inside 25 m, so subsampling the far field saves nothing. It would pay for a
  longer-range sensor.
* **The simulator is a simulator.** It models range noise, dropout, occlusion
  and beam geometry; it does not model rain, dust, retro-reflectors or
  multi-path. Numbers here are not KITTI numbers, and the loaders for real data
  are not written.

## License

MIT.
