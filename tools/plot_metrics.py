from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numbers(rows: list[dict], x_key: str, y_key: str):
    pairs = []
    for row in rows:
        try:
            x, y = float(row[x_key]), float(row[y_key])
        except (TypeError, ValueError, KeyError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            pairs.append((x, y))
    pairs.sort(key=lambda pair: pair[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Stage 4 plots from evaluate.py output.")
    parser.add_argument("--metrics", type=Path, required=True,
                        help="Path to per_image_metrics.csv")
    parser.add_argument("--outdir", type=Path, default=Path("report/figures"))
    args = parser.parse_args()
    rows = load_rows(args.metrics)
    if not rows:
        print("Metrics CSV contains no rows.")
        return 1
    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1. Residuals against the measured perspective-side imbalance.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in [("a3_fourth_corner_error_px", "A3 fourth-corner error"),
                       ("a4_rms_px", "A4 RMS residual"),
                       ("h4_rms_px", "H4 mapping residual")]:
        x, y = numbers(rows, "obliqueness", key)
        if x:
            ax.scatter(x, y, label=label)
    ax.set_xlabel("Obliqueness (opposite-side ratio - 1)")
    ax.set_ylabel("Corner residual (pixels)")
    ax.set_title("Affine / projective corner residual vs obliqueness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "corner_residual_vs_obliqueness.png", dpi=180)
    plt.close(fig)

    # 2. Grid-line straightness/spacing diagnostic.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in [("grid_rms_A3_px", "A3"), ("grid_rms_A4_px", "A4"),
                       ("grid_rms_H4_px", "H4")]:
        x, y = numbers(rows, "obliqueness", key)
        if x:
            ax.scatter(x, y, label=label)
    ax.set_xlabel("Obliqueness (opposite-side ratio - 1)")
    ax.set_ylabel("Grid-line deviation RMS (pixels)")
    ax.set_title("Rectified grid-line deviation vs obliqueness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "grid_straightness_vs_obliqueness.png", dpi=180)
    plt.close(fig)

    # 3. OCR cell accuracy comparison for the three rectification models.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in [("cell_accuracy_A3", "A3"), ("cell_accuracy_A4", "A4"),
                       ("cell_accuracy_H4", "H4")]:
        x, y = numbers(rows, "obliqueness", key)
        if x:
            ax.scatter(x, y, label=label)
    ax.set_xlabel("Obliqueness (opposite-side ratio - 1)")
    ax.set_ylabel("Cell accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("OCR cell accuracy vs obliqueness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "ocr_accuracy_vs_obliqueness.png", dpi=180)
    plt.close(fig)

    # 4. Grid spacing error, measuring differences between adjacent detected lines.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in [("grid_spacing_rms_A3_px", "A3"),
                       ("grid_spacing_rms_A4_px", "A4"),
                       ("grid_spacing_rms_H4_px", "H4")]:
        x, y = numbers(rows, "obliqueness", key)
        if x:
            ax.scatter(x, y, label=label)
    ax.set_xlabel("Obliqueness (opposite-side ratio - 1)")
    ax.set_ylabel("Grid spacing deviation RMS (pixels)")
    ax.set_title("Grid spacing deviation vs obliqueness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "grid_spacing_vs_obliqueness.png", dpi=180)
    plt.close(fig)
    print(f"Saved Stage 4 plots to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
