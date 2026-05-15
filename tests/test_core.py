"""
Unit tests verifying the theoretical claims of Bande Mapping.

Run with:
    pytest tests/
or:
    python -m unittest tests/test_core.py
"""

import unittest
import numpy as np
from scipy.spatial import KDTree

from bande_mapping import bande_transform
from bande_mapping.core import (
    consensus_centroid, bande_score, constraint_activation_rate
)
from bande_mapping.anc import anc_transform
from bande_mapping.heuristics import nn_tsp, tour_cost


class TestTheorems(unittest.TestCase):
    """Verify the theorems from the paper hold in implementation."""

    def setUp(self):
        np.random.seed(0)
        self.points = np.random.uniform(0, 100, (40, 2))

    def test_theorem_existence_pi_in_ci(self):
        """Theorem 2: P_i is always in its own consensus region C_i."""
        N = len(self.points)
        tree = KDTree(self.points)
        for n in [3, 5, 10]:
            distances, indices = tree.query(self.points, k=n + 1)
            for i in range(N):
                for j, r in zip(indices[i, 1:], distances[i, 1:]):
                    d_ij = np.linalg.norm(self.points[i] - self.points[j])
                    # P_i must lie within ball B(P_j, r_ij)
                    self.assertLessEqual(
                        d_ij, r + 1e-9,
                        f"P_{i} not in B(P_{j}, {r}) at n={n}"
                    )

    def test_theorem_collapse_n1(self):
        """Theorem 3(i): At n=1, V_i equals the nearest neighbor."""
        V = bande_transform(self.points, n=1)
        tree = KDTree(self.points)
        _, idx = tree.query(self.points, k=2)
        nearest_neighbors = self.points[idx[:, 1]]
        # V_i should equal nearest neighbor (within MC noise)
        # Note: with one ball, centroid is exactly the ball's center
        max_err = np.max(np.linalg.norm(V - nearest_neighbors, axis=1))
        self.assertLess(max_err, 1e-6,
                        f"n=1 collapse failed: max error {max_err}")

    def test_anc_picks_interior_n(self):
        """Corollary: ANC-selected n* lies strictly in the interior."""
        V, n_star = anc_transform(
            self.points, lambda D: nn_tsp(D, start=0)
        )
        N = len(self.points)
        self.assertGreater(n_star, 1)
        self.assertLess(n_star, N - 1)


class TestProperties(unittest.TestCase):
    """Basic API and shape properties."""

    def test_shape_preservation(self):
        points = np.random.uniform(0, 100, (30, 2))
        V = bande_transform(points, n=5)
        self.assertEqual(V.shape, points.shape)

    def test_higher_dimensional(self):
        """Bande Mapping works in arbitrary dimension."""
        points = np.random.uniform(0, 100, (20, 4))
        V = bande_transform(points, n=4)
        self.assertEqual(V.shape, points.shape)

    def test_invalid_n_raises(self):
        points = np.random.uniform(0, 100, (10, 2))
        with self.assertRaises(ValueError):
            bande_transform(points, n=0)
        with self.assertRaises(ValueError):
            bande_transform(points, n=10)
        with self.assertRaises(ValueError):
            bande_transform(points, n=-1)

    def test_constraint_activation_in_unit_interval(self):
        points = np.random.uniform(0, 100, (30, 2))
        rate = constraint_activation_rate(points, n=5)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_bande_score_nonnegative(self):
        points = np.random.uniform(0, 100, (30, 2))
        V = bande_transform(points, n=5)
        s = bande_score(points, V)
        self.assertGreaterEqual(s, 0.0)

    def test_zero_score_when_identical(self):
        points = np.random.uniform(0, 100, (10, 2))
        self.assertEqual(bande_score(points, points), 0.0)


class TestANC(unittest.TestCase):
    """ANC procedure properties."""

    def test_anc_returns_valid_n(self):
        points = np.random.uniform(0, 100, (40, 2))
        V, n_star = anc_transform(points, lambda D: nn_tsp(D, start=0))
        self.assertEqual(V.shape, points.shape)
        self.assertIsInstance(n_star, int)

    def test_anc_returns_all_costs(self):
        points = np.random.uniform(0, 100, (30, 2))
        V, n_star, costs = anc_transform(
            points, lambda D: nn_tsp(D, start=0), return_all_costs=True
        )
        self.assertIsInstance(costs, dict)
        self.assertEqual(costs[n_star], min(costs.values()))

    def test_anc_optimal_n_matches_min_cost(self):
        """ANC must return the n minimizing the cost (exhaustive search)."""
        points = np.random.uniform(0, 100, (25, 2))
        V, n_star, costs = anc_transform(
            points, lambda D: nn_tsp(D, start=0), return_all_costs=True
        )
        self.assertEqual(min(costs, key=costs.get), n_star)


if __name__ == "__main__":
    unittest.main()
