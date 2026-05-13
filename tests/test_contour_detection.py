import sys
import unittest
from pathlib import Path

import cv2  # type: ignore
import numpy as np

# Allow running this file directly:
#   python tests/test_contour_detection.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contour_detection import (  # noqa: E402
    GREEN_BG_HSV,
    apply_morphology,
    detect_hand_contour,
    filter_contours_by_area,
    find_contours,
    get_main_contour,
    invert_mask,
)
from image_preprocessing import HSVRange  # noqa: E402


class TestContourDetection(unittest.TestCase):
    def test_invert_mask(self) -> None:
        mask = np.array([[0, 255]], dtype=np.uint8)
        inv = invert_mask(mask)
        self.assertTrue(np.array_equal(inv, np.array([[255, 0]], dtype=np.uint8)))

    def test_apply_morphology_preserves_shape(self) -> None:
        mask = np.zeros((50, 60), dtype=np.uint8)
        mask[10:40, 20:50] = 255
        out = apply_morphology(mask, kernel_size=5)
        self.assertEqual(out.shape, mask.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_find_contours_rectangle(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (20, 30), (80, 70), 255, thickness=-1)
        contours = find_contours(mask)
        self.assertEqual(len(contours), 1)

    def test_filter_contours_by_area(self) -> None:
        small = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], dtype=np.int32)
        big = np.array([[[0, 0]], [[0, 100]], [[100, 100]], [[100, 0]]], dtype=np.int32)

        filtered = filter_contours_by_area([small, big], min_area=2000.0)
        self.assertEqual(len(filtered), 1)
        self.assertTrue(np.array_equal(filtered[0], big))

    def test_get_main_contour(self) -> None:
        c1 = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], dtype=np.int32)
        c2 = np.array([[[0, 0]], [[0, 20]], [[20, 20]], [[20, 0]]], dtype=np.int32)
        main = get_main_contour([c1, c2])
        self.assertIsNotNone(main)
        self.assertTrue(np.array_equal(main, c2))

    def test_detect_hand_contour_real_image(self) -> None:
        # Try multiple images to reduce flakiness.
        images_dir = PROJECT_ROOT / "rps-cv-images" / "rock"
        if not images_dir.exists():
            self.skipTest("Dataset folder rps-cv-images/rock not found")

        paths = sorted(images_dir.glob("*.png"))
        if not paths:
            self.skipTest("No .png files found in rps-cv-images/rock")

        hsv_range = HSVRange(lower=(35, 50, 50), upper=(85, 255, 255))

        found_good = False
        for path in paths[:25]:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            contour = detect_hand_contour(
                rgb,
                hsv_range,
                blur_kernel=(5, 5),
                morph_kernel_size=5,
                min_contour_area=1000.0,
            )
            if contour is None:
                continue
            area = float(cv2.contourArea(contour))
            if area > 5000.0:
                found_good = True
                break

        self.assertTrue(found_good, "No hand contour with area > 5000 found in first 25 rock images")


if __name__ == "__main__":
    unittest.main()
