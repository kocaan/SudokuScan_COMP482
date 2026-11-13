from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sudokuscan.config import ScanConfig
from sudokuscan.corners import (detect_outer_corners, _contour_quads,
                                _extract_harris_candidates, _proposal_grid_evidence)
from sudokuscan.preprocessing import preprocess_image, resize_for_detection
from sudokuscan.geometry import order_points


def main() -> int:
    parser = argparse.ArgumentParser(description="Harris parameter study on annotated images.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("parameter_sweep.csv"))
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of images")
    args = parser.parse_args()

    image_paths = sorted(p for p in args.images.rglob("*")
                         if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        image_paths = image_paths[:args.limit]
    settings = list(itertools.product([2, 3, 4], [3, 5], [0.04, 0.06], [0.005, 0.01, 0.02]))
    rows = []
    for image_path in image_paths:
        relative_image = image_path.relative_to(args.images)
        annotation_path = args.annotations / relative_image.parent / f"{image_path.stem}.json"
        if not annotation_path.exists():
            print(f"Skipping {image_path.name}: annotation not found")
            continue
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        gt = np.asarray(annotation["corners"], dtype=np.float32).reshape(4, 2)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        _gray, normalized, binary = preprocess_image(image)
        base_cfg = ScanConfig()
        small_gray, scale = resize_for_detection(normalized, base_cfg.max_detection_side)
        small_binary = cv2.resize(binary, (small_gray.shape[1], small_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        # These proposals depend on the image but not on Harris parameters, so
        # compute them once per photo rather than once for every parameter tuple.
        proposals = _contour_quads(small_gray, small_binary, base_cfg)
        grid_evidence = [_proposal_grid_evidence(small_gray, proposal) for proposal in proposals]
        for block, ksize, k, threshold in settings:
            cfg = ScanConfig(harris_block_size=block, harris_ksize=ksize,
                             harris_k=k, response_threshold=threshold)
            # Candidate count is a required study output even when automatic
            # outer-corner selection fails for a setting. Detection otherwise
            # returns None and would silently record zero candidates.
            candidate_data = _extract_harris_candidates(small_gray, cfg)
            candidates, _response = candidate_data
            candidate_count = int(len(candidates))
            detected = detect_outer_corners(
                small_gray, small_binary, cfg,
                precomputed_candidates_response=candidate_data,
                precomputed_proposals=proposals,
                precomputed_grid_evidence=grid_evidence,
            )
            error_pct = None
            if detected is not None:
                predicted = order_points(detected.corners / scale)
                side = np.mean([np.linalg.norm(gt[1] - gt[0]), np.linalg.norm(gt[2] - gt[1]),
                                np.linalg.norm(gt[3] - gt[2]), np.linalg.norm(gt[0] - gt[3])])
                error_pct = float(np.mean(np.linalg.norm(predicted - gt, axis=1)) / max(side, 1e-9) * 100)
            rows.append({"image": str(relative_image), "blockSize": block, "ksize": ksize,
                         "k": k, "response_threshold": threshold,
                         "candidate_count": candidate_count,
                         "detection_succeeded": detected is not None,
                         "corner_error_percent": error_pct})
            print(f"{image_path.name}: b={block} ks={ksize} k={k} t={threshold} error={error_pct}")

    if not rows:
        print("No annotated images available for the sweep.")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} parameter/image rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
