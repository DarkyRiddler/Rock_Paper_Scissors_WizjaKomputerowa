"""Unit tests for feature_extraction module."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2  # type: ignore
import numpy as np

# Allow running directly:  python tests/test_feature_extraction.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_extraction import (
    FEATURE_NAMES,
    NUM_FEATURES,
    compute_area,
    compute_aspect_ratio,
    compute_circularity,
    compute_convexity,
    compute_hu_moments,
    compute_perimeter,
    extract_feature_vector,
    extract_features_from_image,
)


def _make_rectangle_contour(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Create a synthetic rectangle contour."""
    return np.array([
        [[x, y]],
        [[x + w, y]],
        [[x + w, y + h]],
        [[x, y + h]],
    ], dtype=np.int32)


def _make_circle_contour(cx: int, cy: int, radius: int, n_points: int = 64) -> np.ndarray:
    """Create a synthetic circle contour."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    points = np.stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
    ], axis=-1)
    return points.reshape(-1, 1, 2).astype(np.int32)


class TestComputeArea(unittest.TestCase):
    def test_rectangle_area(self) -> None:
        contour = _make_rectangle_contour(10, 10, 100, 50)
        area = compute_area(contour)
        self.assertAlmostEqual(area, 5000.0, delta=1.0)

    def test_area_positive(self) -> None:
        contour = _make_circle_contour(100, 100, 30)
        area = compute_area(contour)
        self.assertGreater(area, 0)


class TestComputePerimeter(unittest.TestCase):
    def test_rectangle_perimeter(self) -> None:
        contour = _make_rectangle_contour(0, 0, 100, 50)
        perimeter = compute_perimeter(contour)
        self.assertAlmostEqual(perimeter, 300.0, delta=1.0)


class TestComputeHuMoments(unittest.TestCase):
    def test_shape(self) -> None:
        contour = _make_rectangle_contour(0, 0, 50, 50)
        hu = compute_hu_moments(contour)
        self.assertEqual(hu.shape, (7,))

    def test_finite_values(self) -> None:
        contour = _make_circle_contour(50, 50, 30)
        hu = compute_hu_moments(contour)
        self.assertTrue(np.all(np.isfinite(hu)))

    def test_nonzero(self) -> None:
        contour = _make_rectangle_contour(0, 0, 100, 50)
        hu = compute_hu_moments(contour)
        # At least first moment should be non-zero for a real shape
        self.assertNotEqual(hu[0], 0.0)


class TestComputeAspectRatio(unittest.TestCase):
    def test_square(self) -> None:
        contour = _make_rectangle_contour(0, 0, 100, 100)
        ar = compute_aspect_ratio(contour)
        self.assertAlmostEqual(ar, 1.0, delta=0.01)

    def test_wide_rectangle(self) -> None:
        contour = _make_rectangle_contour(0, 0, 200, 100)
        ar = compute_aspect_ratio(contour)
        self.assertAlmostEqual(ar, 2.0, delta=0.01)


class TestComputeConvexity(unittest.TestCase):
    def test_convex_shape(self) -> None:
        # A rectangle is convex => convexity ≈ 1.0
        contour = _make_rectangle_contour(0, 0, 100, 50)
        conv = compute_convexity(contour)
        self.assertAlmostEqual(conv, 1.0, delta=0.05)

    def test_range(self) -> None:
        contour = _make_circle_contour(50, 50, 30)
        conv = compute_convexity(contour)
        self.assertGreaterEqual(conv, 0.0)
        self.assertLessEqual(conv, 1.0)


class TestComputeCircularity(unittest.TestCase):
    def test_circle_high_circularity(self) -> None:
        contour = _make_circle_contour(100, 100, 50, n_points=128)
        circ = compute_circularity(contour)
        # A circle should have circularity close to 1.0
        self.assertGreater(circ, 0.9)

    def test_rectangle_lower_circularity(self) -> None:
        contour = _make_rectangle_contour(0, 0, 200, 50)
        circ = compute_circularity(contour)
        # An elongated rectangle has much lower circularity
        self.assertLess(circ, 0.7)


class TestExtractFeatureVector(unittest.TestCase):
    def test_shape(self) -> None:
        contour = _make_rectangle_contour(0, 0, 100, 50)
        vec = extract_feature_vector(contour)
        self.assertEqual(vec.shape, (NUM_FEATURES,))
        self.assertEqual(len(vec), 12)

    def test_dtype(self) -> None:
        contour = _make_circle_contour(50, 50, 40)
        vec = extract_feature_vector(contour)
        self.assertEqual(vec.dtype, np.float64)

    def test_feature_names_count(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), NUM_FEATURES)
        self.assertEqual(NUM_FEATURES, 12)


class TestExtractFeaturesFromImage(unittest.TestCase):
    def test_on_real_image(self) -> None:
        """Test on a real dataset image if available."""
        sample_dir = PROJECT_ROOT / "rps-cv-images" / "rock"
        if not sample_dir.exists():
            self.skipTest("Dataset not available")

        images = sorted(sample_dir.glob("*.png"))
        if not images:
            self.skipTest("No PNG images found")

        bgr = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        self.assertIsNotNone(bgr)

        feat = extract_features_from_image(bgr)
        # Should succeed on a dataset image
        self.assertIsNotNone(feat)
        self.assertEqual(feat.shape, (12,))
        self.assertTrue(np.all(np.isfinite(feat)))

    def test_invalid_input(self) -> None:
        with self.assertRaises(TypeError):
            extract_features_from_image("not_an_array")  # type: ignore


if __name__ == "__main__":
    unittest.main()
