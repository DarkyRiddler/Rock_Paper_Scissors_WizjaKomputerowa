"""Feature extraction module for hand gesture classification.

Extracts geometric features from hand contours:
  - 7 Hu invariant moments (log-transformed)
  - Area, Perimeter
  - Aspect Ratio, Convexity, Circularity

Author: Michał Kumięga (Tydzień 3-4)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2  # type: ignore
import numpy as np

from contour_detection import GREEN_BG_HSV, detect_hand_contour
from image_preprocessing import HSVRange


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------


def compute_area(contour: np.ndarray) -> float:
    """Compute the area of a contour using cv2.contourArea().

    Args:
        contour: Contour array from cv2.findContours(), shape (N, 1, 2).

    Returns:
        Area in pixels.
    """
    _validate_contour(contour)
    return float(cv2.contourArea(contour))


def compute_perimeter(contour: np.ndarray, closed: bool = True) -> float:
    """Compute the perimeter (arc length) of a contour.

    Args:
        contour: Contour array, shape (N, 1, 2).
        closed: Whether the contour is closed.

    Returns:
        Perimeter in pixels.
    """
    _validate_contour(contour)
    return float(cv2.arcLength(contour, closed))


def compute_hu_moments(contour: np.ndarray) -> np.ndarray:
    """Compute the 7 Hu invariant moments of a contour.

    The moments are log-transformed: sign(hu) * log10(|hu|).
    This is the standard approach because raw Hu moments span many
    orders of magnitude.

    Args:
        contour: Contour array, shape (N, 1, 2).

    Returns:
        1-D array of shape (7,) with log-transformed Hu moments.
    """
    _validate_contour(contour)
    moments = cv2.moments(contour)
    hu_raw = cv2.HuMoments(moments).flatten()  # shape (7,)

    # Log-transform: sign(h) * log10(|h|), with protection against log(0)
    hu_log = np.zeros(7, dtype=np.float64)
    for i, h in enumerate(hu_raw):
        if abs(h) > 0:
            hu_log[i] = -np.sign(h) * np.log10(abs(h))
        else:
            hu_log[i] = 0.0

    return hu_log


def compute_aspect_ratio(contour: np.ndarray) -> float:
    """Compute the aspect ratio of the bounding rectangle.

    Aspect Ratio = width / height of the minimum bounding rectangle.
    A closed fist will have AR ≈ 1.0, an open hand will be taller.

    Args:
        contour: Contour array, shape (N, 1, 2).

    Returns:
        Aspect ratio (width / height). Returns 0.0 if height is 0.
    """
    _validate_contour(contour)
    _, _, w, h = cv2.boundingRect(contour)
    if h == 0:
        return 0.0
    return float(w) / float(h)


def compute_convexity(contour: np.ndarray) -> float:
    """Compute the convexity (solidity) of a contour.

    Convexity = contour_area / convex_hull_area.
    A perfect convex shape has convexity = 1.0.
    Scissors gesture has lower convexity due to the gap between fingers.

    Args:
        contour: Contour array, shape (N, 1, 2).

    Returns:
        Convexity in range [0, 1]. Returns 0.0 if hull area is 0.
    """
    _validate_contour(contour)
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return float(area) / float(hull_area)


def compute_circularity(contour: np.ndarray) -> float:
    """Compute the circularity of a contour.

    Circularity = 4π * area / perimeter².
    A perfect circle has circularity = 1.0.

    Args:
        contour: Contour array, shape (N, 1, 2).

    Returns:
        Circularity value. Returns 0.0 if perimeter is 0.
    """
    _validate_contour(contour)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "hu_0", "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6",
    "area", "perimeter",
    "aspect_ratio", "convexity", "circularity",
]
"""Names for each element in the feature vector (12 features total)."""

NUM_FEATURES: int = len(FEATURE_NAMES)


def extract_feature_vector(contour: np.ndarray) -> np.ndarray:
    """Extract the full feature vector from a single contour.

    The vector consists of 12 features:
      [hu_0 .. hu_6, area, perimeter, aspect_ratio, convexity, circularity]

    Args:
        contour: Contour array, shape (N, 1, 2).

    Returns:
        1-D float64 array of shape (12,).
    """
    _validate_contour(contour)

    hu = compute_hu_moments(contour)           # 7
    area = compute_area(contour)               # 1
    perimeter = compute_perimeter(contour)      # 1
    aspect_ratio = compute_aspect_ratio(contour)  # 1
    convexity = compute_convexity(contour)     # 1
    circularity = compute_circularity(contour) # 1

    return np.array(
        [*hu, area, perimeter, aspect_ratio, convexity, circularity],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Convenience: image → feature vector (uses contour_detection pipeline)
# ---------------------------------------------------------------------------


def extract_features_from_image(
    image_bgr: np.ndarray,
    hsv_range: HSVRange = GREEN_BG_HSV,
) -> Optional[np.ndarray]:
    """Extract features from a BGR image (as returned by cv2.imread).

    This is a convenience function that runs the full pipeline:
      BGR → RGB → detect_hand_contour() → extract_feature_vector()

    Args:
        image_bgr: BGR image (H, W, 3) uint8 (OpenCV imread format).
        hsv_range: HSV range for green background segmentation.

    Returns:
        Feature vector of shape (12,), or None if no hand contour was found.
    """
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (H, W, 3)")
    if image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must be uint8")

    # Convert BGR → RGB (contour_detection expects RGB)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    contour = detect_hand_contour(image_rgb, hsv_range)

    if contour is None:
        return None

    return extract_feature_vector(contour)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_contour(contour: np.ndarray) -> None:
    """Validate that *contour* looks like a cv2 contour array."""
    if not isinstance(contour, np.ndarray):
        raise TypeError("contour must be a numpy array")
    if contour.ndim < 2:
        raise ValueError("contour must have at least 2 dimensions")
    if contour.shape[0] < 3:
        raise ValueError("contour must have at least 3 points")
