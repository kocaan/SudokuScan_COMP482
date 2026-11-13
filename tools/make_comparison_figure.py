from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the required side-by-side rectification figure for the most oblique photo.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("report/figures/most_oblique_rectifications.png"))
    args = parser.parse_args()
    results = []
    for path in args.results.rglob("*_result.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("verdict") != "not_found" and data.get("obliqueness") is not None:
                results.append((float(data["obliqueness"]), path))
        except Exception:
            continue
    if not results:
        print("No successful result JSON files with obliqueness found.")
        return 1
    obliqueness, result_path = max(results, key=lambda item: item[0])
    image_stem = result_path.stem.replace("_result", "")
    paths = [result_path.parent / f"{image_stem}_rect_{model}.png" for model in ("A3", "A4", "H4")]
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if any(image is None for image in images):
        print("One or more A3/A4/H4 rectification images are missing.")
        return 1
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, image, model in zip(axes, images, ("A3", "A4", "H4")):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(model)
        ax.axis("off")
    fig.suptitle(f"Most oblique image: {image_stem} (obliqueness={obliqueness:.3f})")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
