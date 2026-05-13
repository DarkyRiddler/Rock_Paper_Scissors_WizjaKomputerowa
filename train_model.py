"""Train a k-NN classifier for hand gesture recognition.

This script:
  1) Builds a feature matrix from the Kaggle RPS dataset
  2) Splits into train/test (80/20, stratified)
  3) Tunes the k parameter via cross-validation
  4) Trains the final model
  5) Evaluates: accuracy, precision, recall, F1, confusion matrix
  6) Exports model + scaler to models/

Usage:
    python train_model.py
    python train_model.py --data-dir rps-cv-images --method standard

Author: Michał Kumięga (Tydzień 7-8)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2  # type: ignore
import joblib  # type: ignore
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore

from feature_normalization import (
    CLASS_NAMES,
    build_feature_matrix,
    normalize_features,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("rps-cv-images")

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def train(
    data_dir: Path,
    method: str = "standard",
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    output_dir: Path = Path("models"),
    results_dir: Path = Path("results"),
) -> None:
    """Full training pipeline."""


    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build feature matrix
    # ------------------------------------------------------------------
    print("=" * 60)
    print("ETAP 1: Budowanie macierzy cech")
    print("=" * 60)
    t0 = time.time()

    X, y, file_paths = build_feature_matrix(data_dir)
    print(f"\nCzas ekstrakcji cech: {time.time() - t0:.1f}s")
    print(f"Liczba próbek: {len(y)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {np.sum(y == i)} próbek")

    # ------------------------------------------------------------------
    # 2. Normalization
    # ------------------------------------------------------------------
    print(f"\nETAP 2: Normalizacja cech (metoda: {method})")
    X_scaled, scaler = normalize_features(X, method=method)
    print(f"  Kształt macierzy: {X_scaled.shape}")

    # ------------------------------------------------------------------
    # 3. Train/test split
    # ------------------------------------------------------------------
    print(f"\nETAP 3: Podział na zbiory (test_size={test_size})")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"  Treningowy: {X_train.shape[0]} próbek")
    print(f"  Testowy:    {X_test.shape[0]} próbek")

    # ------------------------------------------------------------------
    # 4. Trening finalnego modelu (k=5)
    # ------------------------------------------------------------------
    best_k = 5
    print(f"\nETAP 4: Trening finalnego modelu (k={best_k})")
    model = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Ewaluacja na zbiorze testowym
    # ------------------------------------------------------------------
    print("\nETAP 5: Ewaluacja na zbiorze testowym")
    print("=" * 60)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f} (macro)")
    print(f"  Recall:    {rec:.4f} (macro)")
    print(f"  F1-score:  {f1:.4f} (macro)")

    print(f"\nRaport klasyfikacji:")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Macierz pomyłek:")
    print(cm)
    _plot_confusion_matrix(cm, CLASS_NAMES, results_dir / "confusion_matrix.png")

    # ------------------------------------------------------------------
    # 6. Save report
    # ------------------------------------------------------------------
    report_path = results_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RAPORT KLASYFIKACJI - Kamień, Papier, Nożyce (k-NN)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Najlepsze k: {best_k}\n")
        f.write(f"Metoda normalizacji: {method}\n")
        f.write(f"Podział: {1-test_size:.0%} trening / {test_size:.0%} test\n")
        f.write(f"Liczba próbek: {len(y)} (train: {len(y_train)}, test: {len(y_test)})\n\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f} (macro)\n")
        f.write(f"Recall:    {rec:.4f} (macro)\n")
        f.write(f"F1-score:  {f1:.4f} (macro)\n\n")
        f.write("Raport klasyfikacji:\n")
        f.write(report)
        f.write(f"\nMacierz pomyłek:\n{cm}\n")

    print(f"\nRaport zapisany: {report_path}")

    # ------------------------------------------------------------------
    # 7. Export model + scaler
    # ------------------------------------------------------------------
    model_path = output_dir / "knn_model.pkl"
    scaler_path = output_dir / "scaler.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nModel zapisany:  {model_path}")
    print("\nTrening zakonczony pomyslnie!")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------



def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predykcja", fontsize=12)
    ax.set_ylabel("Prawda", fontsize=12)
    ax.set_title("Macierz pomyłek (Confusion Matrix)", fontsize=14)

    # Write values in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Wykres zapisany: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trenowanie klasyfikatora k-NN dla gestów dłoni (RPS)"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Katalog z danymi (podfoldery rock/paper/scissors)",
    )
    parser.add_argument(
        "--method", type=str, default="standard",
        choices=["standard", "minmax"],
        help="Metoda normalizacji: 'standard' lub 'minmax'",
    )
    parser.add_argument(
        "--test-size", type=float, default=DEFAULT_TEST_SIZE,
        help="Proporcja zbioru testowego (default: 0.20)",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        method=args.method,
        test_size=args.test_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
