from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def make_grid(seed: int = 7) -> tuple[np.ndarray, list[list[int]]]:
    rng = np.random.default_rng(seed)
    grid = np.zeros((9, 9), dtype=int)
    # A fixed valid pattern is used only to generate repeatable software tests.
    base = np.array([[((r * 3 + r // 3 + c) % 9) + 1 for c in range(9)] for r in range(9)])
    keep = rng.random((9, 9)) < 0.42
    values = base * keep
    image = np.full((450, 450, 3), 255, np.uint8)
    for i in range(10):
        p = min(449, int(round(i * 50)))
        thick = 3 if i % 3 == 0 else 1
        cv2.line(image, (p, 0), (p, 449), (0, 0, 0), thick, cv2.LINE_AA)
        cv2.line(image, (0, p), (449, p), (0, 0, 0), thick, cv2.LINE_AA)
    for r in range(9):
        for c in range(9):
            value = int(values[r, c])
            if value:
                cv2.putText(image, str(value), (c * 50 + 17, r * 50 + 37),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
    return image, values.tolist()


def generate(outdir: Path, count: int = 1) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    puzzle, grid = make_grid()
    h, w = 720, 900
    photo = np.full((h, w, 3), 235, np.uint8)
    cv2.rectangle(photo, (15, 15), (w - 16, h - 16), (218, 218, 218), -1)
    cv2.putText(photo, "SYNTHETIC QA IMAGE - NOT A PERSONAL DATASET", (24, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    for index in range(count):
        shift = index * 5
        dst = np.array([[155 + shift, 92], [670, 122 + shift],
                        [635 - shift, 615], [112, 574 - shift]], dtype=np.float32)
        src = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(puzzle, H, (w, h), borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(np.full((450, 450), 255, np.uint8), H, (w, h))
        image = photo.copy()
        image[mask > 0] = warped[mask > 0]
        name = f"synthetic_{index + 1:02d}"
        cv2.imwrite(str(outdir / f"{name}.png"), image)
        annotation = {
            "image": f"{name}.png",
            "conditions": ["C2"],
            "corners": dst.astype(float).tolist(),
            "grid": grid,
            "synthetic_for_software_testing_only": True,
        }
        (outdir / f"{name}.json").write_text(json.dumps(annotation, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("examples/synthetic"))
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    generate(args.outdir, max(1, args.count))
