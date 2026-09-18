"""Traversability, planning and the end-to-end pipeline."""

import unittest

import numpy as np

from avr25d.config import (CLASS_DRIVABLE, CLASS_ROUGH_TERRAIN, PipelineConfig,
                           uniform_cell_count)
from avr25d.grid import navigation, traversability as tv
from avr25d.grid.vrgrid import VariableResolutionGrid
from avr25d.metrics.evaluate import evaluate, integrity_audit
from avr25d.pipeline import Pipeline
from avr25d.planner import astar
from avr25d.sim.lidar import LidarSimulator
from avr25d.sim.world import World


def build_stack(frames=4, seed=7):
    cfg = PipelineConfig()
    world = World.generate(seed=seed)
    sim = LidarSimulator(world, cfg.sensor, seed=seed)
    pipe = Pipeline(cfg)
    res = None
    for f in sim.sequence(frames):
        res = pipe.process(f)
    return cfg, world, pipe, res


class TestTraversability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.world, cls.pipe, cls.res = build_stack(frames=3)

    def test_vehicle_stands_on_reachable_ground(self):
        trav = self.res.trav
        sx, sy = self.res.sensor_xy
        d = np.hypot(trav.cells.cx - sx, trav.cells.cy - sy)
        near = d < 8.0
        self.assertGreater(float(trav.reachable[near].mean()), 0.2)

    def test_the_curb_stops_the_flood(self):
        """Sidewalk is flat, and must still come back non-drivable."""
        trav = self.res.trav
        sx, sy = self.res.sensor_xy
        cx, cy = trav.cells.cx, trav.cells.cy
        u = np.abs(cy - self.world.road_center(cx.astype(np.float64)))
        d = np.hypot(cx - sx, cy - sy)
        on_road = (u < 3.4) & (d < 35) & trav.known
        on_walk = (u > 4.6) & (u < 6.0) & (d < 35) & trav.known
        if on_walk.sum() < 50 or on_road.sum() < 50:
            self.skipTest("not enough cells on both sides of the curb")
        self.assertGreater(float(trav.reachable[on_road].mean()), 0.45)
        self.assertLess(float(trav.reachable[on_walk].mean()), 0.25)

    def test_scores_are_bounded(self):
        s = self.res.trav.score
        self.assertTrue(np.all((s >= 0.0) & (s <= 1.0)))

    def test_navigation_layer_covers_the_map(self):
        nav = self.res.trav.nav
        self.assertGreater(len(nav), 200)
        self.assertTrue(np.all(nav.parent >= 0),
                        "a map cell has no navigation tile")

    def test_overhead_structure_does_not_block_the_road(self):
        """Height decides, not footprint.

        A gantry at 4.6 m and a wall paint the same footprint in plan view.
        Where the ground under overhead structure is actually observed -
        most of it is in the structure's own occlusion shadow - the tile
        must stay drivable, while anything reaching into the vehicle's
        envelope must block.
        """
        nav = self.res.trav.nav
        over = nav.overhead_only & ~nav.blocked & nav.has_ground
        self.assertGreater(int(nav.overhead_only.sum()), 20,
                           "no overhanging structure in view")
        if int(over.sum()) >= 2:
            self.assertGreater(float(nav.passable[over].mean()), 0.5)
        self.assertGreater(int(nav.blocked.sum()), 0)
        self.assertEqual(float(nav.passable[nav.blocked].mean()), 0.0)


class TestPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.world, cls.pipe, cls.res = build_stack(frames=4)

    def test_plans_forward_along_the_road(self):
        nav = self.res.trav.nav
        sx, sy = self.res.sensor_xy
        path = astar.plan(nav, (sx, sy), (sx + 30.0, sy), self.cfg.vehicle)
        self.assertTrue(path.found)
        self.assertGreater(path.length(), 20.0)
        self.assertLess(path.length(), 90.0)

    def test_never_climbs_more_than_the_platform_can(self):
        nav = self.res.trav.nav
        sx, sy = self.res.sensor_xy
        path = astar.plan(nav, (sx, sy), (sx + 30.0, sy), self.cfg.vehicle)
        if not path.found:
            self.skipTest("no path")
        steps = np.abs(np.diff(path.z))
        self.assertLessEqual(float(steps.max()),
                             self.cfg.vehicle.max_step_height + 1e-6)

    def test_unreachable_goal_is_reported_not_faked(self):
        nav = self.res.trav.nav
        sx, sy = self.res.sensor_xy
        path = astar.plan(nav, (sx, sy), (sx + 4000.0, sy + 4000.0),
                          self.cfg.vehicle)
        # the goal snaps to the nearest reachable tile, so the honest failure
        # mode is a short path, never a fabricated one across unmapped ground
        if path.found:
            self.assertLess(path.length(), 400.0)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.world, cls.pipe, cls.res = build_stack(frames=4)

    def test_labels_cover_every_point(self):
        self.assertEqual(self.res.labels.shape[0], self.res.n_points)
        self.assertTrue(np.all((self.res.labels >= 0) & (self.res.labels < 4)))

    def test_grid_refinement_only_revokes_drivability(self):
        raw, ref = self.res.raw_labels, self.res.labels
        changed = raw != ref
        self.assertTrue(np.all(raw[changed] == CLASS_DRIVABLE))
        self.assertTrue(np.all(ref[changed] == CLASS_ROUGH_TERRAIN))

    def test_memory_beats_a_uniform_grid_by_orders_of_magnitude(self):
        m = self.pipe.memory_report()
        self.assertGreater(m["reduction_vs_uniform_fine"], 20.0)
        self.assertEqual(
            int(uniform_cell_count(self.cfg.lod.max_range, self.cfg.lod.base_cell)),
            int(round(2 * self.cfg.lod.max_range / self.cfg.lod.base_cell)) ** 2)

    def test_timing_is_reported_per_stage(self):
        for k in ("features_ms", "inference_ms", "projection_ms",
                  "traversability_ms", "total_ms"):
            self.assertIn(k, self.res.timing_ms)
        parts = sum(self.res.timing_ms[k] for k in
                    ("features_ms", "inference_ms", "projection_ms",
                     "traversability_ms"))
        self.assertLessEqual(parts, self.res.timing_ms["total_ms"] * 1.15 + 1.0)

    def test_integrity_audit_is_clean(self):
        a = integrity_audit(self.pipe.grid, self.res.world_xyz[:, 0],
                            self.res.world_xyz[:, 1], self.res.sensor_xy)
        self.assertEqual(a["cross_level_overlaps"], 0.0)
        self.assertEqual(a["ring_violations"], 0.0)
        self.assertAlmostEqual(a["points_mapped"], 1.0, places=3)
        self.assertAlmostEqual(a["leaf_level_consistency"], 1.0, places=3)

    def test_accuracy_is_reported_and_reasonable(self):
        cfg = PipelineConfig()
        world = World.generate(seed=21)
        sim = LidarSimulator(world, cfg.sensor, seed=21)
        rep = evaluate(Pipeline(cfg), sim, world, n_frames=4, warmup=1)
        d = rep.to_dict()
        self.assertGreater(d["semantic"]["accuracy"], 0.70)
        self.assertGreater(d["latency"]["fps_mean"], 0.5)
        self.assertTrue(d["elevation"], "no elevation statistics produced")
        near = d["elevation"][0]
        far = d["elevation"][-1]
        self.assertLess(near["rmse_m"], far["rmse_m"])

    def test_configuration_round_trips(self):
        cfg = PipelineConfig()
        again = PipelineConfig.from_dict(cfg.to_dict())
        self.assertEqual(again.lod.ring_radii, cfg.lod.ring_radii)
        self.assertEqual(again.vehicle.max_step_height, cfg.vehicle.max_step_height)


class TestConfigValidation(unittest.TestCase):
    def test_rejects_a_bad_ring_schedule(self):
        cfg = PipelineConfig()
        cfg.lod.ring_radii = (0.0, 30.0, 10.0, 40.0, 70.0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_rejects_a_nonzero_first_ring(self):
        cfg = PipelineConfig()
        cfg.lod.ring_radii = (5.0, 10.0, 20.0, 40.0, 70.0)
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
