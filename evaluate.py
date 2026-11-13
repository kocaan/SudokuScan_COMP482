from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _grid_accuracy(gt: np.ndarray, pred_value) -> float | None:
    if pred_value is None:
        return None
    pred = np.asarray(pred_value, dtype=int)
    if pred.shape != (9, 9):
        return None
    return float(np.mean(gt == pred))


def evaluate_pair(annotation: dict, result: dict) -> dict:
    gt_grid = np.asarray(annotation["grid"], dtype=int)
    if gt_grid.shape != (9, 9):
        raise ValueError("Expected a 9x9 ground-truth grid")
    pred_value = result.get("grid", [[0] * 9 for _ in range(9)])
    pred_grid = np.asarray(pred_value, dtype=int)
    if pred_grid.shape != (9, 9):
        raise ValueError("Expected a 9x9 predicted grid")
    cell_accuracy = float(np.mean(gt_grid == pred_grid))
    gt_empty = (gt_grid == 0)
    pred_empty = (pred_grid == 0)
    tp = int(np.count_nonzero(gt_empty & pred_empty))
    fp = int(np.count_nonzero(~gt_empty & pred_empty))
    fn = int(np.count_nonzero(gt_empty & ~pred_empty))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    empty_f1 = 2 * precision * recall / max(1e-12, precision + recall)
    exact = bool(np.array_equal(gt_grid, pred_grid))

    corner_error = None
    if annotation.get("corners") and result.get("corners"):
        gt = np.asarray(annotation["corners"], dtype=float).reshape(4, 2)
        pred = np.asarray(result["corners"], dtype=float).reshape(4, 2)
        side = np.mean([np.linalg.norm(gt[1] - gt[0]), np.linalg.norm(gt[2] - gt[1]),
                        np.linalg.norm(gt[3] - gt[2]), np.linalg.norm(gt[0] - gt[3])])
        corner_error = float(np.mean(np.linalg.norm(gt - pred, axis=1)) / max(side, 1e-9) * 100.0)

    runtime = result.get("runtime_s")
    try:
        runtime_value = float(runtime) if runtime is not None else None
    except (TypeError, ValueError):
        runtime_value = None
    robustness = bool(
        runtime_value is not None
        and math.isfinite(runtime_value)
        and 0.0 <= runtime_value <= 20.0
        and "processing_error" not in result
    )
    line_metrics = result.get("grid_line_metrics", {})
    return {
        "corner_error_percent": corner_error,
        "obliqueness": result.get("obliqueness"),
        "a3_fourth_corner_error_px": result.get("a3_corner_err_px"),
        "a4_rms_px": result.get("a4_rms_px"),
        "h4_rms_px": result.get("h4_rms_px"),
        "grid_rms_A3_px": line_metrics.get("A3", {}).get("rms_px"),
        "grid_rms_A4_px": line_metrics.get("A4", {}).get("rms_px"),
        "grid_rms_H4_px": line_metrics.get("H4", {}).get("rms_px"),
        "grid_max_A3_px": line_metrics.get("A3", {}).get("max_abs_px"),
        "grid_max_A4_px": line_metrics.get("A4", {}).get("max_abs_px"),
        "grid_max_H4_px": line_metrics.get("H4", {}).get("max_abs_px"),
        "grid_spacing_rms_A3_px": line_metrics.get("A3", {}).get("spacing_rms_px"),
        "grid_spacing_rms_A4_px": line_metrics.get("A4", {}).get("spacing_rms_px"),
        "grid_spacing_rms_H4_px": line_metrics.get("H4", {}).get("spacing_rms_px"),
        "grid_spacing_max_A3_px": line_metrics.get("A3", {}).get("spacing_max_abs_px"),
        "grid_spacing_max_A4_px": line_metrics.get("A4", {}).get("spacing_max_abs_px"),
        "grid_spacing_max_H4_px": line_metrics.get("H4", {}).get("spacing_max_abs_px"),
        "cell_accuracy": cell_accuracy,
        "cell_accuracy_A3": _grid_accuracy(gt_grid, result.get("grid_A3")),
        "cell_accuracy_A4": _grid_accuracy(gt_grid, result.get("grid_A4")),
        "cell_accuracy_H4": cell_accuracy,
        "cell_accuracy_H4_fused": cell_accuracy,
        "cell_accuracy_H4_prepared": _grid_accuracy(gt_grid, result.get("grid_H4_prepared")),
        "cell_accuracy_H4_adaptive": _grid_accuracy(gt_grid, result.get("grid_H4_adaptive")),
        "cell_accuracy_H4_raw": _grid_accuracy(gt_grid, result.get("grid_H4_raw")),
        "cell_accuracy_H4_fallback": _grid_accuracy(gt_grid, result.get("grid_H4_fallback")),
        "cell_accuracy_Hmany": _grid_accuracy(gt_grid, result.get("grid_Hmany")),
        "empty_cell_f1": float(empty_f1),
        "grid_exact_match": exact,
        "runtime_s": runtime,
        "robust": robustness,
        "verdict": result.get("verdict"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SudokuScan result JSON files against Appendix A annotations.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("evaluation"))
    parser.add_argument("--max-confusion-examples", type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    by_condition = defaultdict(list)
    confusion_counts = Counter()
    confusion_examples = []
    for annotation_path in sorted(args.annotations.rglob("*.json")):
        annotation = load_json(annotation_path)
        relative_annotation = annotation_path.relative_to(args.annotations)
        image_name = Path(annotation.get("image", annotation_path.stem)).stem
        # Keep the annotation folder structure aligned with recursive batch
        # output, while remaining compatible with the original flat dataset.
        result_path = args.results / relative_annotation.parent / f"{image_name}_result.json"
        result_missing = not result_path.exists()
        if result_missing:
            # A missing output is a failed processing attempt, not an image to
            # silently remove from the robustness denominator. Use the same
            # all-zero grid the CLI writes for a failed image and mark it as a
            # missing-result failure. This keeps cell metrics interpretable
            # while ensuring robustness is reduced for every omitted output.
            print(f"Missing result (counting as failed): {result_path}")
            result = {
                "image": annotation.get("image", image_name),
                "corners": [],
                "grid": [[0] * 9 for _ in range(9)],
                "runtime_s": None,
                "verdict": "missing_result",
                "processing_error": "Result JSON was not produced",
            }
        else:
            result = load_json(result_path)
        metrics = evaluate_pair(annotation, result)
        row = {"image": str(relative_annotation.with_suffix("")),
               "conditions": ";".join(annotation.get("conditions", [])),
               "result_missing": result_missing,
               **metrics}
        rows.append(row)
        for condition in annotation.get("conditions", []):
            by_condition[condition].append(metrics)

        gt = np.asarray(annotation["grid"], dtype=int)
        pred = np.asarray(result.get("grid", [[0] * 9 for _ in range(9)]), dtype=int)
        for r in range(9):
            for c in range(9):
                actual, guessed = int(gt[r, c]), int(pred[r, c])
                if actual != guessed:
                    confusion_counts[f"{actual}->{guessed}"] += 1
                    if len(confusion_examples) < args.max_confusion_examples:
                        confusion_examples.append({
                            "image": image_name,
                            "result_parent": (str(result_path.parent.relative_to(args.results))
                                              if not result_missing else ""),
                            "row": r, "col": c,
                            "ground_truth": actual, "prediction": guessed,
                        })

    if not rows:
        print("No matching annotation/result pairs found.")
        return 1
    csv_path = args.out / "per_image_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for condition, values in sorted(by_condition.items()):
        summary[condition] = {
            "images": len(values),
            "mean_corner_error_percent": float(np.mean([v["corner_error_percent"] for v in values if v["corner_error_percent"] is not None])) if any(v["corner_error_percent"] is not None for v in values) else None,
            "mean_cell_accuracy_H4": float(np.mean([v["cell_accuracy_H4"] for v in values])),
            "mean_cell_accuracy_A3": float(np.mean([v["cell_accuracy_A3"] for v in values if v["cell_accuracy_A3"] is not None])) if any(v["cell_accuracy_A3"] is not None for v in values) else None,
            "mean_cell_accuracy_A4": float(np.mean([v["cell_accuracy_A4"] for v in values if v["cell_accuracy_A4"] is not None])) if any(v["cell_accuracy_A4"] is not None for v in values) else None,
            "mean_cell_accuracy_H4_fused": float(np.mean([v["cell_accuracy_H4_fused"] for v in values])),
            "mean_cell_accuracy_H4_prepared": float(np.mean([v["cell_accuracy_H4_prepared"] for v in values if v["cell_accuracy_H4_prepared"] is not None])) if any(v["cell_accuracy_H4_prepared"] is not None for v in values) else None,
            "mean_cell_accuracy_H4_adaptive": float(np.mean([v["cell_accuracy_H4_adaptive"] for v in values if v["cell_accuracy_H4_adaptive"] is not None])) if any(v["cell_accuracy_H4_adaptive"] is not None for v in values) else None,
            "mean_cell_accuracy_H4_raw": float(np.mean([v["cell_accuracy_H4_raw"] for v in values if v["cell_accuracy_H4_raw"] is not None])) if any(v["cell_accuracy_H4_raw"] is not None for v in values) else None,
            "mean_cell_accuracy_H4_fallback": float(np.mean([v["cell_accuracy_H4_fallback"] for v in values if v["cell_accuracy_H4_fallback"] is not None])) if any(v["cell_accuracy_H4_fallback"] is not None for v in values) else None,
            "mean_cell_accuracy_Hmany": float(np.mean([v["cell_accuracy_Hmany"] for v in values if v["cell_accuracy_Hmany"] is not None])) if any(v["cell_accuracy_Hmany"] is not None for v in values) else None,
            "mean_grid_rms_A3_px": float(np.mean([v["grid_rms_A3_px"] for v in values if v["grid_rms_A3_px"] is not None])) if any(v["grid_rms_A3_px"] is not None for v in values) else None,
            "mean_grid_rms_A4_px": float(np.mean([v["grid_rms_A4_px"] for v in values if v["grid_rms_A4_px"] is not None])) if any(v["grid_rms_A4_px"] is not None for v in values) else None,
            "mean_grid_rms_H4_px": float(np.mean([v["grid_rms_H4_px"] for v in values if v["grid_rms_H4_px"] is not None])) if any(v["grid_rms_H4_px"] is not None for v in values) else None,
            "mean_grid_spacing_rms_A3_px": float(np.mean([v["grid_spacing_rms_A3_px"] for v in values if v["grid_spacing_rms_A3_px"] is not None])) if any(v["grid_spacing_rms_A3_px"] is not None for v in values) else None,
            "mean_grid_spacing_rms_A4_px": float(np.mean([v["grid_spacing_rms_A4_px"] for v in values if v["grid_spacing_rms_A4_px"] is not None])) if any(v["grid_spacing_rms_A4_px"] is not None for v in values) else None,
            "mean_grid_spacing_rms_H4_px": float(np.mean([v["grid_spacing_rms_H4_px"] for v in values if v["grid_spacing_rms_H4_px"] is not None])) if any(v["grid_spacing_rms_H4_px"] is not None for v in values) else None,
            "mean_empty_cell_f1": float(np.mean([v["empty_cell_f1"] for v in values])),
            "grid_exact_match_rate": float(np.mean([v["grid_exact_match"] for v in values])),
            "robustness_rate": float(np.mean([v["robust"] for v in values])),
        }
    (args.out / "condition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "digit_confusions.json").write_text(json.dumps(confusion_counts.most_common(), indent=2), encoding="utf-8")
    (args.out / "confusion_examples.json").write_text(json.dumps(confusion_examples, indent=2), encoding="utf-8")

    # Copy raw/prepared crop pairs for representative errors when the scanner
    # saved its H4 cell crops. These are useful for figures in the report.
    example_dir = args.out / "confusion_examples"
    copied = 0
    for item in confusion_examples:
        # Result/cell folders mirror the source directory tree in recursive
        # batch mode. Keep that parent path so duplicate image basenames in
        # different puzzle folders do not copy crops from the wrong scan.
        source_dir = args.results / item.get("result_parent", "") / f"{item['image']}_cells"
        stem = f"r{item['row'] + 1}_c{item['col'] + 1}"
        for suffix in ("_crop.png", "_prepared.png", "_adaptive.png"):
            source = source_dir / f"{stem}{suffix}"
            if source.exists():
                relative_parent = Path(item.get("result_parent", ""))
                destination_dir = example_dir / relative_parent
                destination_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination_dir / f"{item['image']}_{stem}{suffix}")
                copied += 1
    print(f"Wrote {csv_path}, condition summary, confusion counts and {copied} crop examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
