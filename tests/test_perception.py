"""Simulator, features and the segmentation backends."""

import unittest
import tempfile

import numpy as np

from avr25d.config import (CLASS_DRIVABLE, CLASS_DYNAMIC_OBJECT,
                           CLASS_ROUGH_TERRAIN, CLASS_STATIC_OBSTACLE,
                           NUM_CLASSES, SensorConfig)
from avr25d.models.features import NUM_FEATURES, compute_features
from avr25d.models.data import load_npz_dataset
from avr25d.models.geometric import GeometricSegmenter, estimate_ground
from avr25d.models.infer import Segmenter, torch_available
from avr25d.sim.lidar import LidarSimulator
from avr25d.sim.world import World


class TestWorld(unittest.TestCase):
    def setUp(self):
        self.w = World.generate(seed=7)

    def test_reproducible(self):
        a = World.generate(seed=7)
        b = World.generate(seed=7)
        x = np.linspace(0, 100, 50)
        y = np.zeros(50)
        np.testing.assert_allclose(a.ground_height(x, y), b.ground_height(x, y))

    def test_curb_is_a_step(self):
        """The sidewalk sits a curb height above the road, sharply."""
        x = np.full(300, 40.0)
        yc = float(self.w.road_center(np.array([40.0]))[0])
        u = np.linspace(0.0, 7.0, 300)
        z = self.w.ground_height(x, yc + u)
        road = z[u < 3.5].mean()
        walk = z[(u > 4.4) & (u < 5.8)].mean()
        self.assertGreater(walk - road, 0.10)
        # and the transition happens over a fraction of a metre
        jump = np.max(np.abs(np.diff(z[(u > 3.8) & (u < 4.6)])))
        self.assertGreater(jump, 0.01)

    def test_terrain_labels_split_at_the_curb(self):
        x = np.full(200, 25.0)
        yc = float(self.w.road_center(np.array([25.0]))[0])
        u = np.linspace(0.0, 7.0, 200)
        lab = self.w.terrain_label(x, yc + u)
        self.assertTrue(np.all(lab[u < 3.5] == CLASS_DRIVABLE))
        self.assertTrue(np.all(lab[u > 4.5] == CLASS_ROUGH_TERRAIN))

    def test_dynamic_objects_move(self):
        a = self.w.objects_at(0.0)
        b = self.w.objects_at(2.0)
        moved = sum(1 for p, q in zip(a, b) if not np.allclose(p.params, q.params))
        self.assertGreater(moved, 0)


class TestRealDataAdapter(unittest.TestCase):
    def test_loads_labeled_npz_scan(self):
        xyz = np.array([[1.0, 0.0, 0.2], [2.0, 0.1, 0.3]], dtype=np.float32)
        labels = np.array([0, 2], dtype=np.int64)
        with tempfile.TemporaryDirectory() as root:
            path = root + "/scan.npz"
            np.savez(path, xyz=xyz, label=labels)
            dataset = load_npz_dataset(path)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["feats"].shape[0], 2)
        np.testing.assert_array_equal(dataset[0]["label"], labels)

    def test_rejects_unknown_label(self):
        with tempfile.TemporaryDirectory() as root:
            path = root + "/scan.npz"
            np.savez(path, xyz=np.zeros((1, 3)), label=np.array([4]))
            with self.assertRaises(ValueError):
                load_npz_dataset(path)


class TestSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = World.generate(seed=7)
        cls.sim = LidarSimulator(cls.world, SensorConfig(), seed=7)
        cls.frame = cls.sim.scan(0.6)

    def test_scan_is_dense_and_in_range(self):
        f = self.frame
        self.assertGreater(len(f), 20000)
        r = f.range
        self.assertGreaterEqual(r.min(), self.sim.cfg.min_range - 0.2)
        self.assertLessEqual(r.max(), self.sim.cfg.max_range + 0.5)

    def test_all_four_classes_appear(self):
        counts = np.bincount(self.frame.label.astype(int), minlength=NUM_CLASSES)
        self.assertTrue(np.all(counts > 0), f"missing classes: {counts}")

    def test_ground_returns_sit_on_the_height_field(self):
        f = self.frame
        c, s = np.cos(f.yaw), np.sin(f.yaw)
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        w = f.xyz @ r.T + f.origin
        g = f.label <= CLASS_ROUGH_TERRAIN
        truth = self.world.ground_height(w[g, 0], w[g, 1])
        err = np.abs(w[g, 2] - truth)
        self.assertLess(float(np.median(err)), 0.05)

    def test_overhang_is_above_the_vehicle(self):
        f = self.frame
        if not np.any(f.overhang):
            self.skipTest("no overhanging geometry in view")
        self.assertGreater(float(np.median(f.xyz[f.overhang, 2])), 0.8)

    def test_sequence_advances(self):
        frames = list(self.sim.sequence(3))
        xs = [f.origin[0] for f in frames]
        self.assertTrue(xs[0] < xs[1] < xs[2])


class TestFeatures(unittest.TestCase):
    def test_shape_and_finiteness(self):
        world = World.generate(seed=2)
        sim = LidarSimulator(world, SensorConfig(), seed=2)
        f = sim.scan(0.2)
        b = compute_features(f.xyz, f.intensity)
        self.assertEqual(b.features.shape, (len(f), NUM_FEATURES))
        self.assertTrue(np.all(np.isfinite(b.features)))

    def test_vertical_structure_is_not_planar_ground(self):
        """A wall and a road must separate on the normal-z feature."""
        rng = np.random.default_rng(0)
        road = np.stack([rng.uniform(-3, 3, 4000), rng.uniform(-3, 3, 4000),
                         np.full(4000, -1.7)], axis=1).astype(np.float32)
        wall = np.stack([np.full(4000, 4.0), rng.uniform(-3, 3, 4000),
                         rng.uniform(-1.7, 1.5, 4000)], axis=1).astype(np.float32)
        xyz = np.vstack([road, wall])
        b = compute_features(xyz, np.full(xyz.shape[0], 0.4, dtype=np.float32))
        nz = b.features[:, 10]
        self.assertGreater(float(nz[:4000].mean()), float(nz[4000:].mean()) + 0.3)


class TestSegmenters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = World.generate(seed=7)
        cls.sim = LidarSimulator(cls.world, SensorConfig(), seed=7)
        cls.frame = cls.sim.scan(0.8)

    def test_geometric_backend_is_useful(self):
        seg = GeometricSegmenter()
        lab, probs = seg.predict(self.frame.xyz, self.frame.intensity)
        self.assertEqual(lab.shape, (len(self.frame),))
        self.assertAlmostEqual(float(probs.sum(axis=1).mean()), 1.0, places=4)
        gt = self.frame.label
        # obstacles are the easy class and must be found reliably even
        # without a network
        obst = gt == CLASS_STATIC_OBSTACLE
        self.assertGreater(float((lab[obst] == CLASS_STATIC_OBSTACLE).mean()), 0.7)

    def test_ground_model_finds_the_surface(self):
        g = estimate_ground(self.frame.xyz)
        gz = g.sample(self.frame.xyz[:, 0], self.frame.xyz[:, 1])
        ground = self.frame.label <= CLASS_ROUGH_TERRAIN
        err = np.abs(self.frame.xyz[ground, 2] - gz[ground])
        self.assertLess(float(np.median(err)), 0.12)

    def test_segmenter_always_returns_something(self):
        seg = Segmenter()
        lab, probs, bundle = seg.predict(self.frame.xyz, self.frame.intensity)
        self.assertEqual(lab.shape[0], len(self.frame))
        self.assertEqual(probs.shape, (len(self.frame), NUM_CLASSES))
        self.assertIn("inference_ms", seg.last_timing)

    @unittest.skipUnless(torch_available(), "PyTorch not installed")
    def test_torch_and_numpy_backends_agree_broadly(self):
        from avr25d.config import ModelConfig
        a = Segmenter(ModelConfig())
        b = Segmenter(ModelConfig(backend="numpy"))
        la, _, _ = a.predict(self.frame.xyz, self.frame.intensity)
        lb, _, _ = b.predict(self.frame.xyz, self.frame.intensity)
        if a.net is None:
            self.skipTest("no checkpoint available")
        self.assertGreater(float((la == lb).mean()), 0.6)


if __name__ == "__main__":
    unittest.main()
