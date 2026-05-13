from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np


HSVBound = Tuple[int, int, int]


@dataclass(frozen=True)
class HSVRange:
    lower: HSVBound
    upper: HSVBound

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array(self.lower, dtype=np.uint8)
        upper = np.array(self.upper, dtype=np.uint8)
        return lower, upper


def rgb_to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to HSV (OpenCV HSV ranges).

    Expects shape (H, W, 3) and dtype uint8.
    """
    _validate_color_image(image_rgb)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)


def bgr_to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 image (as returned by cv2.imread) to HSV."""
    _validate_color_image(image_bgr)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """Apply Gaussian blur.

    Typical kernel sizes for this project: (5, 5) or (7, 7).
    """
    if kernel_size not in {(5, 5), (7, 7)}:
        raise ValueError("kernel_size must be (5, 5) or (7, 7)")
    return cv2.GaussianBlur(image, kernel_size, sigmaX=0)


def hsv_in_range_mask(hsv_image: np.ndarray, hsv_range: HSVRange) -> np.ndarray:
    """Create a binary mask selecting pixels within the HSV range."""
    _validate_hsv_image(hsv_image)
    lower, upper = hsv_range.as_arrays()
    return cv2.inRange(hsv_image, lower, upper)


def preprocess_to_hsv_mask(
    image_rgb: np.ndarray,
    hsv_range: HSVRange,
    blur_kernel: Tuple[int, int] = (5, 5),
) -> np.ndarray:
    """RGB -> (optional blur) -> HSV -> binary mask."""
    _validate_color_image(image_rgb)
    blurred = gaussian_blur(image_rgb, blur_kernel)
    hsv = rgb_to_hsv(blurred)
    return hsv_in_range_mask(hsv, hsv_range)


def calibrate_hsv_range_from_samples(
    hsv_images: Sequence[np.ndarray],
    *,
    sample_mask: Optional[np.ndarray] = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    margin: HSVBound = (2, 20, 20),
) -> HSVRange:
    """Calibrate HSV thresholds from sample HSV images.

    This is a simple heuristic: it collects HSV pixels (optionally within a mask)
    and computes per-channel percentiles, then expands by a margin.

    Notes:
    - OpenCV HSV: H in [0, 179], S in [0, 255], V in [0, 255]
    """
    if not hsv_images:
        raise ValueError("hsv_images must not be empty")

    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")

    pixels_list: list[np.ndarray] = []

    for hsv in hsv_images:
        _validate_hsv_image(hsv)

        if sample_mask is not None:
            if sample_mask.shape[:2] != hsv.shape[:2]:
                raise ValueError("sample_mask must match image height/width")
            mask_u8 = (sample_mask > 0).astype(np.uint8)
            selected = hsv[mask_u8.astype(bool)]
        else:
            selected = hsv.reshape(-1, 3)

        if selected.size:
            pixels_list.append(selected)

    if not pixels_list:
        raise ValueError("No pixels available for calibration (mask may be empty)")

    pixels = np.concatenate(pixels_list, axis=0).astype(np.float32)

    lows = np.percentile(pixels, lower_percentile, axis=0)
    highs = np.percentile(pixels, upper_percentile, axis=0)

    margin_arr = np.array(margin, dtype=np.float32)
    lows = lows - margin_arr
    highs = highs + margin_arr

    # Clamp to OpenCV HSV bounds
    lows[0] = np.clip(lows[0], 0, 179)
    highs[0] = np.clip(highs[0], 0, 179)
    lows[1:] = np.clip(lows[1:], 0, 255)
    highs[1:] = np.clip(highs[1:], 0, 255)

    lower = tuple(int(x) for x in np.round(lows))  # type: ignore[assignment]
    upper = tuple(int(x) for x in np.round(highs))  # type: ignore[assignment]

    return HSVRange(lower=lower, upper=upper)


def _validate_color_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")


def _validate_hsv_image(hsv: np.ndarray) -> None:
    _validate_color_image(hsv)
    # Nothing else: HSV values are still uint8 3-channel.
