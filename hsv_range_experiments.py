from __future__ import annotations

import argparse
from pathlib import Path

import cv2  # type: ignore
import numpy as np

from image_preprocessing import (
    HSVRange,
    bgr_to_hsv,
    gaussian_blur,
    hsv_in_range_mask,
)


def parse_hsv_triplet(text: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("HSV triplet must be 'H,S,V'")
    try:
        h, s, v = (int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError("HSV triplet must contain integers") from e
    return h, s, v


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Test different HSV ranges on sample images and save binary masks. "
            "Input images are read with OpenCV (BGR)."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("rps-cv-images"),
        help="Directory containing class subfolders (paper/rock/scissors)",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="rock",
        choices=["rock", "paper", "scissors"],
        help="Which class folder to sample images from",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many images to process",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        default=5,
        choices=[5, 7],
        help="Gaussian blur kernel size (5 or 7 => (5,5)/(7,7))",
    )
    parser.add_argument(
        "--lower",
        type=parse_hsv_triplet,
        required=True,
        help="Lower HSV bound, e.g. 35,50,50",
    )
    parser.add_argument(
        "--upper",
        type=parse_hsv_triplet,
        required=True,
        help="Upper HSV bound, e.g. 85,255,255",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments_out"),
        help="Where to save output masks",
    )

    args = parser.parse_args()

    class_dir = args.images_dir / args.class_name
    if not class_dir.exists():
        raise SystemExit(f"Missing folder: {class_dir}")

    paths = sorted(class_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"No .png images found in: {class_dir}")

    out_dir = args.out_dir / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    hsv_range = HSVRange(lower=args.lower, upper=args.upper)
    kernel = (args.kernel, args.kernel)

    for path in paths[: args.limit]:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"SKIP: could not read {path}")
            continue

        blurred = gaussian_blur(bgr, kernel)
        hsv = bgr_to_hsv(blurred)
        mask = hsv_in_range_mask(hsv, hsv_range)

        # A tiny bit of reporting (ratio of selected pixels)
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        print(f"{path.name}: selected={ratio:.3f}")

        out_path = out_dir / f"{path.stem}_mask_H{args.lower[0]}-{args.upper[0]}_K{args.kernel}.png"
        cv2.imwrite(str(out_path), mask)

    print(f"Saved masks to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
