"""Quick environment sanity-check for OpenCV + NumPy.

Run:
  python test_setup.py

It prints versions and performs a simple BGR->HSV conversion.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        print(
            "ERROR: Missing dependency: cv2 (OpenCV). Install with: pip install opencv-python",
            file=sys.stderr,
            flush=True,
        )
        return 1

    try:
        import numpy as np
    except ModuleNotFoundError:
        print(
            "ERROR: Missing dependency: numpy. Install with: pip install numpy",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"OpenCV wersja: {cv2.__version__}", flush=True)
    print(f"NumPy wersja: {np.__version__}", flush=True)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    print("OK: cv2.cvtColor(BGR->HSV) dziala git", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
