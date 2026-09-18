"""The structural guarantees of the variable-resolution grid.

These are the claims the design rests on, so they are tested numerically
rather than asserted in a comment.
"""

import unittest

import numpy as np

from avr25d.config import NUM_CLASSES, PipelineConfig
from avr25d.grid.vrgrid import (BYTES_PER_CELL, VariableResolutionGrid,
                                pack_keys, unpack_keys)


def random_scan(n=20000, seed=0, radius=95.0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(1.5, radius, n)
    th = rng.uniform(0, 2 * np.pi, n)
    x, y = r * np.cos(th), r * np.sin(th)
    z = 0.2 * np.sin(x / 9.0) + 0.05 * rng.standard_normal(n)
    probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    probs[:, 0] = 1.0
    return x, y, z, probs, np.full(n, 0.4, dtype=np.float32)


class TestKeys(unittest.TestCase):
    def test_roundtrip(self):
        ix = np.array([-100000, -1, 0, 1, 100000])
        iy = np.array([5, -7, 0, 99999, -100000])
        a, b = unpack_keys(pack_keys(ix, iy))
        np.testing.assert_array_equal(a, ix)
        np.testing.assert_array_equal(b, iy)

    def test_keys_are_unique(self):
        rng = np.random.default_rng(1)
        ix = rng.integers(-50000, 50000, 5000)
        iy = rng.integers(-50000, 50000, 5000)
        keys = pack_keys(ix, iy)
        pairs = {(int(a), int(b)) for a, b in zip(ix, iy)}
        self.assertEqual(len(np.unique(keys)), len(pairs))


class TestLeafLevels(unittest.TestCase):
    def setUp(self):
        self.g = VariableResolutionGrid(PipelineConfig())

    def test_level_is_a_property_of_the_cell_not_the_point(self):
        """Two points in the same leaf cell always resolve to that leaf.

        This is the invariant that stops a cell straddling a ring boundary
        from being split between two resolutions - the classic way a
        multi-resolution projection loses points.  Each sample is jittered
        anywhere inside its own leaf and must come back with the same
        level, including the samples deliberately placed on a ring edge.
        """
        rng = np.random.default_rng(3)
        x, y, *_ = random_scan(n=8000, seed=3)
        # bias half the samples onto the ring boundaries, where it hurts
        edges = np.array(self.g.lod.ring_radii[1:])
        th = rng.uniform(0, 2 * np.pi, 4000)
        re = rng.choice(edges, 4000) + rng.uniform(-0.3, 0.3, 4000)
        x = np.concatenate([x, re * np.cos(th)])
        y = np.concatenate([y, re * np.sin(th)])

        lv = self.g.leaf_level(x, y, (0.0, 0.0))
        r = np.array([self.g.lod.cell_size(int(l)) for l in lv])
        x0, y0 = np.floor(x / r) * r, np.floor(y / r) * r
        for _ in range(6):
            jx = x0 + rng.uniform(1e-6, 1 - 1e-6, x.shape[0]) * r
            jy = y0 + rng.uniform(1e-6, 1 - 1e-6, y.shape[0]) * r
            again = self.g.leaf_level(jx, jy, (0.0, 0.0))
            bad = int((again != lv).sum())
            self.assertEqual(bad, 0, f"{bad} cells resolved to two levels")

    def test_coarse_cells_only_far_away(self):
        x, y, *_ = random_scan(seed=4)
        lv = self.g.leaf_level(x, y, (0.0, 0.0))
        for L in range(1, self.g.lod.n_levels):
            sel = lv == L
            if not np.any(sel):
                continue
            d = self.g.cell_min_distance(x[sel], y[sel], L, (0.0, 0.0))
            self.assertTrue(np.all(d >= self.g.lod.ring_radii[L] - 1e-9))

    def test_finest_level_is_the_fallback(self):
        lv = self.g.leaf_level(np.array([0.3]), np.array([0.2]), (0.0, 0.0))
        self.assertEqual(int(lv[0]), 0)


class TestProjection(unittest.TestCase):
    def setUp(self):
        self.g = VariableResolutionGrid(PipelineConfig())
        self.x, self.y, self.z, self.p, self.i = random_scan(seed=5)
        self.g.update(self.x, self.y, self.z, self.p, self.i, (0.0, 0.0), 1.7,
                      ground_mask=np.ones(self.x.shape[0], dtype=bool),
                      frame_index=0)

    def test_every_point_lands_in_exactly_one_leaf(self):
        lvl, row = self.g.lookup(self.x, self.y)
        self.assertTrue(np.all(lvl >= 0), "some points were not mapped")
        leaf = self.g.leaf_level(self.x, self.y, (0.0, 0.0))
        np.testing.assert_array_equal(lvl, leaf)

    def test_no_cell_is_stored_at_two_levels(self):
        for fine in range(self.g.lod.n_levels - 1):
            store = self.g.levels[fine]
            if len(store) == 0:
                continue
            ix, iy = unpack_keys(store.keys)
            for coarse in range(fine + 1, self.g.lod.n_levels):
                other = self.g.levels[coarse]
                if len(other) == 0:
                    continue
                anc = pack_keys(ix >> (coarse - fine), iy >> (coarse - fine))
                self.assertEqual(int((other.rows(np.unique(anc)) >= 0).sum()), 0)

    def test_elevation_tracks_the_surface(self):
        lvl, row = self.g.lookup(self.x, self.y)
        est = self.g.elevation_at(self.x, self.y)
        self.assertLess(float(np.nanmean(np.abs(est - self.z))), 0.08)

    def test_memory_accounting(self):
        self.assertEqual(self.g.memory_bytes(),
                         self.g.cell_count() * BYTES_PER_CELL)
        self.assertGreater(self.g.cell_count(), 1000)


class TestKalman(unittest.TestCase):
    def test_variance_shrinks_with_observations(self):
        g = VariableResolutionGrid(PipelineConfig())
        rng = np.random.default_rng(11)
        x = np.full(1, 3.03)
        y = np.full(1, 1.02)
        probs = np.zeros((1, NUM_CLASSES), dtype=np.float32)
        probs[:, 0] = 1.0
        inten = np.full(1, 0.3, dtype=np.float32)
        first = None
        for k in range(12):
            z = 0.4 + 0.02 * rng.standard_normal(1)
            g.update(x, y, z, probs, inten, (0.0, 0.0), 1.7,
                     np.ones(1, dtype=bool), frame_index=k)
            lvl, row = g.lookup(np.array([3.03]), np.array([1.02]))
            var = float(g.levels[int(lvl[0])].var[int(row[0])])
            if first is None:
                first = var
        self.assertLess(var, first)
        est = float(g.elevation_at(np.array([3.03]), np.array([1.02]))[0])
        self.assertAlmostEqual(est, 0.4, delta=0.03)


class TestRebalance(unittest.TestCase):
    def test_migration_conserves_the_surface(self):
        """Coarsening and refining must not move the mapped elevation."""
        g = VariableResolutionGrid(PipelineConfig())
        x, y, z, p, i = random_scan(n=30000, seed=9)
        g.update(x, y, z, p, i, (0.0, 0.0), 1.7,
                 np.ones(x.shape[0], dtype=bool), frame_index=0)
        before = g.elevation_at(x, y)
        moved = g.rebalance((25.0, 0.0))
        after = g.elevation_at(x, y)
        self.assertGreater(moved["coarsened"] + moved["refined"], 0)
        ok = np.isfinite(before) & np.isfinite(after)
        self.assertLess(float(np.mean(np.abs(before[ok] - after[ok]))), 0.05)

    def test_levels_stay_disjoint_after_rebalance(self):
        g = VariableResolutionGrid(PipelineConfig())
        x, y, z, p, i = random_scan(n=20000, seed=13)
        g.update(x, y, z, p, i, (0.0, 0.0), 1.7,
                 np.ones(x.shape[0], dtype=bool), frame_index=0)
        for sx in (10.0, 20.0, 35.0):
            g.rebalance((sx, 0.0))
        seen = set()
        for L, store in enumerate(g.levels):
            ix, iy = unpack_keys(store.keys)
            for k in range(L + 1, g.lod.n_levels):
                anc = set(pack_keys(ix >> (k - L), iy >> (k - L)).tolist())
                self.assertFalse(anc & set(g.levels[k].keys.tolist()))
            seen |= set(store.keys.tolist())


if __name__ == "__main__":
    unittest.main()
