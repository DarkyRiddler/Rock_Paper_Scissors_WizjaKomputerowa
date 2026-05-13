"""Prediction module for hand gesture classification.

Loads a trained k-NN model and scaler, then predicts the gesture
class for new images.

Usage:
    python predict.py --image path/to/image.png
    python predict.py --image-dir path/to/folder/

Author: Michał Kumięga (Tydzień 7-8)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2  # type: ignore
import joblib  # type: ignore
import numpy as np

from contour_detection import GREEN_BG_HSV
from feature_extraction import extract_features_from_image
from feature_normalization import CLASS_NAMES
from image_preprocessing import HSVRange


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_path: Path | str = Path("models/knn_model.pkl"),
    scaler_path: Path | str = Path("models/scaler.pkl"),
) -> Tuple[object, object]:
    """Load a trained k-NN model and its scaler.

    Args:
        model_path: Path to the pickled KNeighborsClassifier.
        scaler_path: Path to the pickled scaler (StandardScaler/MinMaxScaler).

    Returns:
        Tuple of (model, scaler).

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_gesture(
    image_bgr: np.ndarray,
    model: object,
    scaler: object,
    hsv_range: HSVRange = GREEN_BG_HSV,
    class_names: Optional[list[str]] = None,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Predict the gesture class for a single BGR image.

    Args:
        image_bgr: BGR image (H, W, 3) uint8 (as returned by cv2.imread).
        model: Trained KNeighborsClassifier.
        scaler: Fitted scaler object.
        hsv_range: HSV range for green background segmentation.
        class_names: List of class names. Defaults to CLASS_NAMES.

    Returns:
        Tuple of (label, probabilities):
          - label: Predicted class name (e.g. 'rock'), or None if detection failed.
          - probabilities: Array of shape (n_classes,) with class probabilities,
            or None if detection failed.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    feat = extract_features_from_image(image_bgr, hsv_range)
    if feat is None:
        return None, None

    # Reshape for single-sample prediction and normalize
    feat_2d = feat.reshape(1, -1)
    feat_scaled = scaler.transform(feat_2d)

    # Predict class and probabilities
    label_idx = int(model.predict(feat_scaled)[0])
    proba = model.predict_proba(feat_scaled)[0]

    label_name = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
    return label_name, proba


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predykcja gestu dłoni z obrazu (Kamień/Papier/Nożyce)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image", type=Path,
        help="Ścieżka do pojedynczego obrazu PNG",
    )
    group.add_argument(
        "--image-dir", type=Path,
        help="Katalog z obrazami do predykcji",
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/knn_model.pkl"),
        help="Ścieżka do modelu",
    )
    parser.add_argument(
        "--scaler", type=Path, default=Path("models/scaler.pkl"),
        help="Ścieżka do scalera",
    )
    args = parser.parse_args()

    model, scaler = load_model(args.model, args.scaler)

    if args.image:
        _predict_single(args.image, model, scaler)
    else:
        _predict_directory(args.image_dir, model, scaler)

    return 0


def _predict_single(path: Path, model: object, scaler: object) -> None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"BŁĄD: Nie można wczytać obrazu: {path}")
        return

    label, proba = predict_gesture(bgr, model, scaler)
    if label is None:
        print(f"{path.name}: Nie wykryto dłoni")
    else:
        proba_str = ", ".join(
            f"{CLASS_NAMES[i]}={p:.1%}" for i, p in enumerate(proba)
        )
        print(f"{path.name}: {label.upper()} [{proba_str}]")


def _predict_directory(dir_path: Path, model: object, scaler: object) -> None:
    if not dir_path.exists():
        print(f"BŁĄD: Katalog nie istnieje: {dir_path}")
        return

    paths = sorted(dir_path.glob("*.png"))
    if not paths:
        print(f"Brak plików PNG w: {dir_path}")
        return

    print(f"Predykcja dla {len(paths)} obrazów z {dir_path}\n")
    for path in paths:
        _predict_single(path, model, scaler)


if __name__ == "__main__":
    raise SystemExit(main())
