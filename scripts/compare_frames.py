"""Compare two uint8 frame stacks ([T, H, W, 3] .npy): identical or not, PSNR, max and mean absolute difference, per-frame minimum PSNR.

    python scripts/compare_frames.py a.npy b.npy
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return math.inf if mse == 0 else 10 * math.log10(255**2 / mse)


def main() -> int:
    a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
    if a.shape != b.shape:
        print(json.dumps({"shapes": [list(a.shape), list(b.shape)]}))
        return 1
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    per_frame = [psnr(x, y) for x, y in zip(a, b)]
    finite = [p for p in per_frame if p != math.inf]
    print(json.dumps({"identical": bool((diff == 0).all()), "psnr_db": round(psnr(a, b), 2) if psnr(a, b) != math.inf else "inf",
                      "max_abs": int(diff.max()), "mean_abs": round(float(diff.mean()), 3),
                      "min_frame_psnr_db": round(min(finite), 2) if finite else "inf"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
