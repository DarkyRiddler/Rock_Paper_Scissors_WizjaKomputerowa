import sys
import unittest
from pathlib import Path

import numpy as np

# Allow running this file directly:
#   python tests/test_image_preprocessing.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_preprocessing import (
    HSVRange,
    calibrate_hsv_range_from_samples,
    gaussian_blur,
    hsv_in_range_mask,
    rgb_to_hsv,
)


class TestImagePreprocessing(unittest.TestCase):
    def test_rgb_to_hsv_red(self) -> None:
        # 1x1 pure red in RGB
        rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
        hsv = rgb_to_hsv(rgb)

        self.assertEqual(hsv.shape, (1, 1, 3))
        h, s, v = (int(x) for x in hsv[0, 0])

        # OpenCV uses H in [0..179]. Red is around 0.
        self.assertTrue(h <= 5 or h >= 175)
        self.assertGreaterEqual(s, 250)
        self.assertGreaterEqual(v, 250)

    def test_gaussian_blur_preserves_shape(self) -> None:
        img = np.zeros((20, 30, 3), dtype=np.uint8)
        img[10, 15] = (255, 255, 255)

        blurred5 = gaussian_blur(img, (5, 5))
        blurred7 = gaussian_blur(img, (7, 7))

        self.assertEqual(blurred5.shape, img.shape)
        self.assertEqual(blurred7.shape, img.shape)
        # Blur should spread the impulse
        self.assertNotEqual(int(blurred5[10, 15, 0]), 255)

    def test_hsv_in_range_mask(self) -> None:
        # Create an HSV image with two pixels
        hsv = np.zeros((1, 2, 3), dtype=np.uint8)
        hsv[0, 0] = (50, 200, 200)
        hsv[0, 1] = (10, 200, 200)

        hsv_range = HSVRange(lower=(40, 100, 100), upper=(60, 255, 255))
        mask = hsv_in_range_mask(hsv, hsv_range)

        self.assertEqual(mask.shape, (1, 2))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(mask[0, 0]), 255)
        self.assertEqual(int(mask[0, 1]), 0)

    def test_calibrate_hsv_range_from_samples_basic(self) -> None:
        hsv1 = np.zeros((2, 2, 3), dtype=np.uint8)
        hsv2 = np.zeros((2, 2, 3), dtype=np.uint8)

        hsv1[:] = (40, 100, 100)
        hsv2[:] = (60, 200, 200)

        hsv_range = calibrate_hsv_range_from_samples(
            [hsv1, hsv2], lower_percentile=0.0, upper_percentile=100.0, margin=(0, 0, 0)
        )

        self.assertEqual(hsv_range.lower, (40, 100, 100))
        self.assertEqual(hsv_range.upper, (60, 200, 200))


if __name__ == "__main__":
    unittest.main()
