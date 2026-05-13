"""Unit tests for feature_normalization module."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Allow running directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_normalization import normalize_features


class TestNormalizeFeatures(unittest.TestCase):
    def setUp(self) -> None:
        """Create a sample feature matrix."""
        rng = np.random.RandomState(42)
        self.X = rng.rand(50, 12) * np.array([
            5, 3, 2, 1, 0.5, 0.3, 0.1,  # Hu moments (small)
            50000, 800,                    # area, perimeter (large)
            1.5, 0.8, 0.6,                # ratios (small)
        ])

    def test_standard_scaler_mean_zero(self) -> None:
        X_scaled, scaler = normalize_features(self.X, method="standard")
        self.assertEqual(X_scaled.shape, self.X.shape)
        # After StandardScaler, mean should be ≈ 0
        means = X_scaled.mean(axis=0)
        for m in means:
            self.assertAlmostEqual(m, 0.0, places=10)

    def test_standard_scaler_std_one(self) -> None:
        X_scaled, _ = normalize_features(self.X, method="standard")
        stds = X_scaled.std(axis=0)
        for s in stds:
            self.assertAlmostEqual(s, 1.0, places=5)

    def test_minmax_scaler_range(self) -> None:
        X_scaled, _ = normalize_features(self.X, method="minmax")
        self.assertEqual(X_scaled.shape, self.X.shape)
        # All values should be in [0, 1]
        self.assertGreaterEqual(X_scaled.min(), -1e-10)
        self.assertLessEqual(X_scaled.max(), 1.0 + 1e-10)

    def test_invalid_method(self) -> None:
        with self.assertRaises(ValueError):
            normalize_features(self.X, method="invalid")

    def test_scaler_is_reusable(self) -> None:
        """The returned scaler should be usable for transforming new data."""
        X_scaled, scaler = normalize_features(self.X, method="standard")
        # Transform a single sample
        single = self.X[0:1]
        single_scaled = scaler.transform(single)
        self.assertEqual(single_scaled.shape, (1, 12))


if __name__ == "__main__":
    unittest.main()
