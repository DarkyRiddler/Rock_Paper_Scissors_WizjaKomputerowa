"""Feature normalization and dataset construction module.

Builds a feature matrix from all images in the dataset and applies
normalization (StandardScaler or MinMaxScaler) so that k-NN distance
metrics are not dominated by high-magnitude features like Area.

Author: Michał Kumięga (Tydzień 5-6)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2  # type: ignore
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore

from contour_detection import GREEN_BG_HSV
from feature_extraction import (
    FEATURE_NAMES,
    NUM_FEATURES,
    extract_features_from_image,
)
from image_preprocessing import HSVRange


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = ["rock", "paper", "scissors"]
"""Default class names (folder names in the dataset)."""

CLASS_LABELS: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
"""Mapping: class name -> integer label."""


def build_feature_matrix(
    image_dir: Path | str,
    hsv_range: HSVRange = GREEN_BG_HSV,
    class_names: Optional[list[str]] = None,
    *,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix X and label vector y from the image dataset.

    Iterates over all PNG images in ``image_dir/{class_name}/``,
    extracts features from each, and returns the full matrix.

    Args:
        image_dir: Root directory containing class subfolders.
        hsv_range: HSV range for green background segmentation.
        class_names: List of class folder names. Defaults to CLASS_NAMES.
        verbose: If True, print progress to stdout.

    Returns:
        Tuple of (X, y, file_paths):
          - X: Feature matrix of shape (n_samples, 12), dtype float64.
          - y: Label vector of shape (n_samples,), dtype int.
          - file_paths: List of file paths (for error analysis).

    Raises:
        FileNotFoundError: If a class folder does not exist.
    """
    image_dir = Path(image_dir)
    if class_names is None:
        class_names = CLASS_NAMES

    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    paths_list: list[str] = []

    for class_name in class_names:
        class_dir = image_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        label = CLASS_LABELS.get(class_name)
        if label is None:
            label = class_names.index(class_name)

        image_paths = sorted(class_dir.glob("*.png"))

        success = 0
        skipped = 0
        for path in image_paths:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                skipped += 1
                continue

            feat = extract_features_from_image(bgr, hsv_range)
            if feat is None:
                skipped += 1
                continue

            features_list.append(feat)
            labels_list.append(label)
            paths_list.append(str(path))
            success += 1

        if verbose:
            total = len(image_paths)
            print(
                f"  [{class_name:>10s}] "
                f"{success}/{total} OK, {skipped} skipped",
                flush=True,
            )

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels_list, dtype=np.int64)

    if verbose:
        print(f"\nFeature matrix: {X.shape}, Labels: {y.shape}", flush=True)

    return X, y, paths_list


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_features(
    X: np.ndarray,
    method: str = "standard",
) -> Tuple[np.ndarray, object]:
    """Normalize the feature matrix.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        method: 'standard' for StandardScaler (mean=0, std=1)
                'minmax'   for MinMaxScaler (range [0, 1]).

    Returns:
        Tuple of (X_scaled, scaler):
          - X_scaled: Normalized feature matrix, same shape.
          - scaler: Fitted scaler object (for later use on test/new data).

    Raises:
        ValueError: If method is not 'standard' or 'minmax'.
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'standard' or 'minmax'.")

    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
