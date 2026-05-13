from __future__ import annotations

from typing import List, Tuple

import cv2  # type: ignore
import numpy as np

from image_preprocessing import HSVRange, preprocess_to_hsv_mask


GREEN_BG_HSV = HSVRange(lower=(35, 50, 50), upper=(85, 255, 255))


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Invert a binary mask.

    The mask produced by image preprocessing detects GREEN BACKGROUND
    (white=background, black=hand). After inversion: white=hand, black=background.

    Args:
        mask: Single-channel binary mask, dtype uint8.

    Returns:
        Inverted mask (uint8) with same shape.
    """
    _validate_mask(mask)
    return cv2.bitwise_not(mask)


def apply_morphology(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply morphological operations to clean up a binary mask.

    Order:
      1) Erosion (cv2.erode) — remove small noise
      2) Dilation (cv2.dilate) — restore object size
      3) Closing (MORPH_CLOSE) — fill small holes
      4) Opening (MORPH_OPEN) — remove small artifacts

    Kernel:
      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    Args:
        mask: Single-channel binary mask, dtype uint8.
        kernel_size: Odd integer >= 3 recommended.

    Returns:
        Cleaned mask (uint8) with same shape.
    """
    _validate_mask(mask)
    if not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be int")
    if kernel_size <= 0:
        raise ValueError("kernel_size must be > 0")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    eroded = cv2.erode(mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return opened


def find_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Detect contours on a binary mask.

    Uses cv2.findContours with:
      - mode: cv2.RETR_EXTERNAL (external contours only)
      - method: cv2.CHAIN_APPROX_SIMPLE

    Args:
        mask: Single-channel binary mask, dtype uint8.

    Returns:
        List of contours (each contour is an ndarray).
    """
    _validate_mask(mask)

    result = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV compatibility: (contours, hierarchy) or (image, contours, hierarchy)
    contours = result[0] if len(result) == 2 else result[1]
    return list(contours)


def filter_contours_by_area(contours: list[np.ndarray], min_area: float = 1000.0) -> list[np.ndarray]:
    """Filter contours by area.

    Args:
        contours: List of contours.
        min_area: Minimum contour area to keep.

    Returns:
        Filtered list of contours.
    """
    if min_area < 0:
        raise ValueError("min_area must be >= 0")

    kept: list[np.ndarray] = []
    for contour in contours:
        if contour is None:
            continue
        area = float(cv2.contourArea(contour))
        if area >= min_area:
            kept.append(contour)
    return kept


def get_main_contour(contours: list[np.ndarray]) -> np.ndarray | None:
    """Return the largest contour by area.

    Args:
        contours: List of contours.

    Returns:
        Largest contour or None if the list is empty.
    """
    if not contours:
        return None

    best_contour: np.ndarray | None = None
    best_area = -1.0
    for contour in contours:
        if contour is None:
            continue
        area = float(cv2.contourArea(contour))
        if area > best_area:
            best_area = area
            best_contour = contour

    return best_contour


def detect_hand_contour(
    image_rgb: np.ndarray,
    hsv_range: HSVRange,
    blur_kernel: tuple[int, int] = (5, 5),
    morph_kernel_size: int = 5,
    min_contour_area: float = 1000.0,
) -> np.ndarray | None:
    """End-to-end pipeline: RGB image -> main hand contour.

    Steps:
      1) preprocess_to_hsv_mask() from image_preprocessing (mask of green background)
      2) invert_mask() -> white=hand
      3) apply_morphology() -> cleaned mask
      4) find_contours()
      5) filter_contours_by_area()
      6) get_main_contour()

    Args:
        image_rgb: RGB image (H, W, 3), dtype uint8.
        hsv_range: HSV range for detecting green background.
        blur_kernel: Gaussian blur kernel, typically (5,5) or (7,7).
        morph_kernel_size: Kernel size for morphology.
        min_contour_area: Minimum area to keep.

    Returns:
        Hand contour (ndarray) or None.
    """
    _validate_rgb_image(image_rgb)

    bg_mask = preprocess_to_hsv_mask(image_rgb, hsv_range, blur_kernel)
    hand_mask = invert_mask(bg_mask)
    cleaned = apply_morphology(hand_mask, kernel_size=morph_kernel_size)

    contours = find_contours(cleaned)
    contours = filter_contours_by_area(contours, min_area=min_contour_area)
    return get_main_contour(contours)


def _validate_mask(mask: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy array")
    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel (H, W) array")
    if mask.dtype != np.uint8:
        raise ValueError("mask must be uint8")


def _validate_rgb_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
