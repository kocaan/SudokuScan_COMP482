from __future__ import annotations

"""Run SudokuScan on an annotated personal dataset and create report-ready diagnostics.

Expected layouts (both supported):

  data/personal/images/photo_01.jpg
  data/personal/annotations/photo_01.json

or a combined directory containing matching image/JSON pairs. Annotation JSON
files follow Appendix A: image, conditions, corners (TL,TR,BR,BL), and 9x9 grid.
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CONDITIONS = ("C1", "C2", "C3", "C4", "C5")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON annotation {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Annotation must be a JSON object: {path}")
    return value


def _validate_annotation(path: Path, annotation: dict[str, Any]) -> None:
    if "image" not in annotation or "grid" not in annotation or "corners" not in annotation:
        raise ValueError(f"Not an Appendix A annotation (image/grid/corners missing): {path}")
    if "conditions" not in annotation:
        raise ValueError(f"Annotation conditions field is missing: {path}")
    grid = np.asarray(annotation["grid"])
    if grid.shape != (9, 9):
        raise ValueError(f"Annotation grid must be exactly 9x9: {path}")
    try:
        numeric = grid.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Annotation grid must contain integers 0..9: {path}") from exc
    if not np.array_equal(grid, numeric) or np.any(numeric < 0) or np.any(numeric > 9):
        raise ValueError(f"Annotation grid values must be integers 0..9: {path}")
    corners = np.asarray(annotation["corners"], dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError(f"Annotation corners must contain four finite [x,y] pairs: {path}")
    conditions = annotation.get("conditions", [])
    if not isinstance(conditions, list) or any(item not in CONDITIONS for item in conditions):
        raise ValueError(f"conditions must be a list containing only C1..C5: {path}")


def _resolve_image(images_root: Path, annotation_path: Path,
                   annotations_root: Path, annotation: dict[str, Any]) -> Path | None:
    relative = annotation_path.relative_to(annotations_root)
    image_value = str(annotation.get("image", annotation_path.stem))
    image_name = Path(image_value).name
    candidates = [
        images_root / relative.parent / image_name,
        images_root / image_name,
    ]
    if not Path(image_name).suffix:
        candidates.extend(images_root / relative.parent / f"{annotation_path.stem}{ext}"
                          for ext in sorted(IMAGE_EXTENSIONS))
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    matches = [path for path in images_root.rglob(f"{annotation_path.stem}.*")
               if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if len(matches) == 1:
        return matches[0]
    return None


def _discover_dataset(images_root: Path, annotations_root: Path) -> list[dict[str, Any]]:
    if not images_root.is_dir():
        raise ValueError(f"Image folder does not exist: {images_root}")
    if not annotations_root.is_dir():
        raise ValueError(f"Annotation folder does not exist: {annotations_root}")
    items: list[dict[str, Any]] = []
    invalid: list[str] = []
    for annotation_path in sorted(annotations_root.rglob("*.json")):
        try:
            annotation = _read_json(annotation_path)
            # Ignore unrelated JSON files such as exports and result files.
            if "grid" not in annotation or "corners" not in annotation:
                continue
            _validate_annotation(annotation_path, annotation)
            image_path = _resolve_image(images_root, annotation_path, annotations_root, annotation)
            if image_path is None:
                invalid.append(f"{annotation_path}: matching image not found")
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                invalid.append(f"{image_path}: OpenCV could not read image")
                continue
            corners = np.asarray(annotation["corners"], dtype=float)
            image_height, image_width = image.shape[:2]
            if (np.any(corners[:, 0] < 0) or np.any(corners[:, 0] >= image_width)
                    or np.any(corners[:, 1] < 0) or np.any(corners[:, 1] >= image_height)):
                invalid.append(
                    f"{annotation_path}: corner coordinates are outside the original image "
                    f"bounds {image_width}x{image_height}"
                )
                continue
            items.append({
                "annotation_path": annotation_path,
                "relative_annotation": annotation_path.relative_to(annotations_root),
                "annotation": annotation,
                "image_path": image_path,
                "image_size": [int(image.shape[1]), int(image.shape[0])],
            })
        except ValueError as exc:
            invalid.append(str(exc))
    if invalid:
        raise ValueError("Dataset annotation problems:\n  - " + "\n  - ".join(invalid))
    if not items:
        raise ValueError(
            f"No valid Appendix A annotations were found in {annotations_root}. "
            "Each JSON must contain image, corners and a 9x9 grid."
        )
    return items


def _run(command: list[str], log_path: Path, *, cwd: Path = ROOT,
         timeout_s: float = 600.0) -> int:
    """Run a tool with a hard timeout and retain partial output on failure.

    Long OCR sweeps can trigger slow Tesseract segmentation modes. They must
    never hang the complete dataset report indefinitely. Exit code 124 denotes
    a timeout; other child exit codes are preserved.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_line = "$ " + subprocess.list2cmdline(command) + "\n\n"
    try:
        completed = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=False,
            timeout=max(1.0, float(timeout_s)),
        )
        output_text = completed.stdout or ""
        error_text = completed.stderr or ""
        log_path.write_text(
            command_line + output_text
            + ("\n--- STDERR ---\n" + error_text if error_text else ""),
            encoding="utf-8",
        )
        return int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        log_path.write_text(
            command_line + stdout
            + ("\n--- STDERR ---\n" + stderr if stderr else "")
            + f"\n\nERROR: command exceeded timeout of {timeout_s:.1f} seconds.\n",
            encoding="utf-8",
        )
        return 124


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _value(row: dict[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
        return 1.0 if raw.strip().lower() == "true" else 0.0
    return _float(raw)


def _mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = [value for row in rows if (value := _value(row, key)) is not None]
    return float(np.mean(values)) if values else None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_confusion_and_cell_diagnostics(
    items: list[dict[str, Any]], results_root: Path, output_root: Path,
    max_error_crops: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    matrix = np.zeros((10, 10), dtype=np.int64)
    cell_rows: list[dict[str, Any]] = []
    error_counts = Counter()
    examples: list[tuple[dict[str, Any], Path, Path, Path]] = []

    for item in items:
        annotation = item["annotation"]
        relative = item["relative_annotation"]
        image_stem = Path(annotation.get("image", relative.stem)).stem
        result_path = results_root / relative.parent / f"{image_stem}_result.json"
        result: dict[str, Any] = {}
        if result_path.is_file():
            try:
                result = _read_json(result_path)
            except ValueError:
                result = {}
        predicted = np.asarray(result.get("grid", [[0] * 9 for _ in range(9)]), dtype=int)
        if predicted.shape != (9, 9):
            predicted = np.zeros((9, 9), dtype=int)
        truth = np.asarray(annotation["grid"], dtype=int)
        cells_dir = result_path.parent / f"{image_stem}_cells"
        metadata: dict[tuple[int, int], dict[str, Any]] = {}
        meta_path = cells_dir / "cells.json"
        if meta_path.is_file():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                metadata = {(int(entry["row"]), int(entry["col"])): entry
                            for entry in payload if "row" in entry and "col" in entry}
            except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
                metadata = {}

        for row in range(9):
            for col in range(9):
                actual, guess = int(truth[row, col]), int(predicted[row, col])
                matrix[actual, guess] += 1
                if actual == guess:
                    error_type = "correct"
                elif actual == 0:
                    error_type = "false_digit_in_empty_cell"
                elif guess == 0:
                    error_type = "missed_printed_digit"
                else:
                    error_type = "digit_substitution"
                error_counts[error_type] += int(error_type != "correct")
                cell_meta = metadata.get((row, col), {})
                cell_rows.append({
                    "image": str(relative.with_suffix("")),
                    "row": row + 1, "col": col + 1,
                    "ground_truth": actual, "prediction": guess,
                    "correct": actual == guess, "error_type": error_type,
                    "is_filled_by_precheck": cell_meta.get("is_filled"),
                    "otsu_digit": cell_meta.get("digit_prepared"),
                    "adaptive_digit": cell_meta.get("digit_adaptive"),
                    "raw_digit": cell_meta.get("digit_raw"),
                    "fallback_digit": cell_meta.get("digit_fallback"),
                    "otsu_confidence": cell_meta.get("confidence_prepared"),
                    "adaptive_confidence": cell_meta.get("confidence_adaptive"),
                    "raw_confidence": cell_meta.get("confidence_raw"),
                    "fallback_confidence": cell_meta.get("confidence_fallback"),
                    "ink_fraction": cell_meta.get("ink_fraction"),
                })
                if error_type != "correct" and len(examples) < max_error_crops:
                    stem = f"r{row + 1}_c{col + 1}"
                    examples.append((
                        {"image": image_stem, "relative": str(relative),
                         "row": row + 1, "col": col + 1,
                         "truth": actual, "prediction": guess,
                         "error_type": error_type, "meta": cell_meta},
                        cells_dir / f"{stem}_crop.png",
                        cells_dir / f"{stem}_prepared.png",
                        cells_dir / f"{stem}_adaptive.png",
                    ))

    error_dir = output_root / "error_examples"
    error_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for descriptor, crop_path, prepared_path, adaptive_path in examples:
        panels = []
        for path in (crop_path, prepared_path, adaptive_path):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
            if image is not None:
                panels.append(cv2.resize(image, (180, 180), interpolation=cv2.INTER_AREA))
            else:
                panels.append(np.full((180, 180), 245, dtype=np.uint8))
        canvas = np.full((230, 540, 3), 255, dtype=np.uint8)
        for idx, panel in enumerate(panels):
            canvas[42:222, idx * 180:(idx + 1) * 180] = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        title = (f"{descriptor['image']} r{descriptor['row']}c{descriptor['col']} "
                 f"GT={descriptor['truth']} PRED={descriptor['prediction']} "
                 f"({descriptor['error_type']})")
        cv2.putText(canvas, title[:85], (8, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (20, 20, 20), 1, cv2.LINE_AA)
        for idx, label in enumerate(("Raw crop", "Otsu", "Adaptive")):
            cv2.putText(canvas, label, (idx * 180 + 8, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (30, 30, 30), 1, cv2.LINE_AA)
        name = f"{written + 1:03d}_{descriptor['image']}_r{descriptor['row']}_c{descriptor['col']}.png"
        cv2.imwrite(str(error_dir / name), canvas)
        written += 1

    _write_csv(output_root / "cell_diagnostics.csv", cell_rows)
    _write_csv(output_root / "error_type_counts.csv",
               [{"error_type": key, "count": value} for key, value in sorted(error_counts.items())],
               ["error_type", "count"])
    _write_csv(output_root / "confusion_matrix.csv",
               [{"ground_truth": row, **{f"pred_{col}": int(matrix[row, col]) for col in range(10)}}
                for row in range(10)])
    (output_root / "error_type_counts.json").write_text(
        json.dumps(dict(error_counts), indent=2), encoding="utf-8")
    return matrix, cell_rows, dict(error_counts)


def _make_figures(metrics_rows: list[dict[str, str]], matrix: np.ndarray,
                  output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    # Confusion matrix including the empty-cell class (0).
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Cell count")
    ax.set(xticks=range(10), yticks=range(10),
           xticklabels=["0 (empty)"] + [str(x) for x in range(1, 10)],
           yticklabels=["0 (empty)"] + [str(x) for x in range(1, 10)],
           xlabel="Predicted class", ylabel="Ground-truth class",
           title="Sudoku cell confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = max(1, int(matrix.max()) * 0.55)
    for row in range(10):
        for col in range(10):
            value = int(matrix[row, col])
            if value:
                ax.text(col, row, str(value), ha="center", va="center",
                        color="white" if value > threshold else "black", fontsize=8)
    fig.tight_layout()
    path = output_dir / "cell_confusion_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(path.name)

    # Per-image accuracy, with the assignment's three rectification models.
    names = [Path(row["image"]).name for row in metrics_rows]
    x = np.arange(len(names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
    for offset, key, label in [(-width, "cell_accuracy_A3", "A3"),
                               (0.0, "cell_accuracy_A4", "A4"),
                               (width, "cell_accuracy_H4", "H4")]:
        values = [_float(row.get(key)) or 0.0 for row in metrics_rows]
        ax.bar(x + offset, values, width, label=label)
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Cell accuracy")
    ax.set_title("OCR cell accuracy by image and rectification model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "accuracy_by_image.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(path.name)

    # Corner error and OCR accuracy against obliqueness.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key, label in [("cell_accuracy_A3", "A3"), ("cell_accuracy_A4", "A4"),
                       ("cell_accuracy_H4", "H4")]:
        pairs = [(_float(row.get("obliqueness")), _float(row.get(key))) for row in metrics_rows]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if pairs:
            ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], label=label)
    ax.set_xlabel("Obliqueness (opposite-side ratio minus 1)")
    ax.set_ylabel("Cell accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("OCR accuracy versus viewing obliqueness")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "ocr_accuracy_vs_obliqueness.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(path.name)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    pairs = [(_float(row.get("obliqueness")), _float(row.get("corner_error_percent")))
             for row in metrics_rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if pairs:
        ax.scatter([p[0] for p in pairs], [p[1] for p in pairs])
    ax.set_xlabel("Obliqueness (opposite-side ratio minus 1)")
    ax.set_ylabel("Mean corner error (% of mean side length)")
    ax.set_title("Corner localization error versus obliqueness")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_dir / "corner_error_vs_obliqueness.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(path.name)
    return files


def _write_report_summary(
    items: list[dict[str, Any]], metrics_rows: list[dict[str, str]],
    error_counts: dict[str, int], output_root: Path, figures: list[str],
    psm_rows: list[dict[str, Any]] | None = None,
    parameter_csv: Path | None = None,
    quality: dict[str, Any] | None = None,
) -> None:
    count = len(metrics_rows)
    conditions = Counter(condition for item in items
                         for condition in item["annotation"].get("conditions", []))
    means = {
        "H4 accuracy": _mean(metrics_rows, "cell_accuracy_H4"),
        "A3 accuracy": _mean(metrics_rows, "cell_accuracy_A3"),
        "A4 accuracy": _mean(metrics_rows, "cell_accuracy_A4"),
        "H4 prepared/Otsu accuracy": _mean(metrics_rows, "cell_accuracy_H4_prepared"),
        "H4 adaptive accuracy": _mean(metrics_rows, "cell_accuracy_H4_adaptive"),
        "H4 raw-crop accuracy": _mean(metrics_rows, "cell_accuracy_H4_raw"),
        "H4 fallback-only accuracy": _mean(metrics_rows, "cell_accuracy_H4_fallback"),
        "Empty-cell F1": _mean(metrics_rows, "empty_cell_f1"),
        "Corner error (%)": _mean(metrics_rows, "corner_error_percent"),
        "Runtime (s)": _mean(metrics_rows, "runtime_s"),
        "Robustness rate": _mean(metrics_rows, "robust"),
        "Exact-grid rate": _mean(metrics_rows, "grid_exact_match"),
    }
    lines = [
        "# SudokuScan personal-dataset diagnosis",
        "",
        f"**Annotated images evaluated:** {count}",
        "",
        "> These measurements were generated from the matching original images and manually verified Appendix A annotations. "
        "Do not report them as personal-dataset results unless the photographs and annotations are genuinely yours and have been checked.",
        "",
        "## Aggregate results",
        "",
        "| Metric | Macro mean across images |",
        "|---|---:|",
    ]
    for key, value in means.items():
        rendered = "N/A" if value is None else (f"{value * 100:.2f}%" if "accuracy" in key.lower() or "rate" in key.lower() or "f1" in key.lower() else f"{value:.4f}")
        lines.append(f"| {key} | {rendered} |")
    if quality is not None:
        lines += ["", "## Dataset requirement checks", "", "| Requirement | Result |", "|---|---|"]
        for key, passed in quality["checks"].items():
            if passed is None:
                result_text = "Not machine-verifiable (add optional puzzle_id fields)"
            else:
                result_text = "PASS" if passed else "NOT MET"
            lines.append(f"| {key.replace('_', ' ')} | {result_text} |")
        lines.append(f"| Unannotated image files | {len(quality['unannotated_images'])} |")
        if quality["unannotated_images"]:
            lines += ["", "Images without annotations:"]
            lines.extend(f"- `{name}`" for name in quality["unannotated_images"])
    lines += ["", "## Capture-condition coverage", "", "| Condition | Annotated photographs |", "|---|---:|"]
    for condition in CONDITIONS:
        lines.append(f"| {condition} | {conditions.get(condition, 0)} |")
    missing = [condition for condition in CONDITIONS if not conditions.get(condition)]
    if missing:
        lines += ["", f"**Missing coverage:** {', '.join(missing)}. Capture and annotate these conditions before claiming complete C1-C5 coverage."]
    lines += ["", "## Recognition error diagnosis", "", "| Error type | Count |", "|---|---:|"]
    for key, label in [("missed_printed_digit", "Printed digit missed (GT digit → 0)"),
                       ("digit_substitution", "Wrong digit substituted"),
                       ("false_digit_in_empty_cell", "False digit in an empty cell")]:
        lines.append(f"| {label} | {error_counts.get(key, 0)} |")
    lines += ["", "## Per-image results", "",
              "| Image | Conditions | Corner error % | A3 | A4 | H4 | Empty F1 | Exact | Runtime s |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics_rows:
        def fmt(key: str, percent: bool = False) -> str:
            value = _value(row, key)
            if value is None:
                return "N/A"
            return f"{value * 100:.2f}" if percent else f"{value:.3f}"
        lines.append(
            f"| {Path(row['image']).name} | {row.get('conditions','')} | {fmt('corner_error_percent')} | "
            f"{fmt('cell_accuracy_A3', True)} | {fmt('cell_accuracy_A4', True)} | "
            f"{fmt('cell_accuracy_H4', True)} | {fmt('empty_cell_f1', True)} | "
            f"{fmt('grid_exact_match', True)} | {fmt('runtime_s')} |"
        )
    if psm_rows:
        lines += ["", "## Primary Tesseract PSM comparison", "",
                  "These runs disable the secondary fallback so only the primary segmentation mode changes. "
                  "A timed-out or failed mode is marked in the status column and must not be treated as a valid comparison.",
                  "", "| PSM | Status | Mean H4 accuracy | Mean empty F1 | Exact-grid rate | Mean runtime (s) |",
                  "|---:|---|---:|---:|---:|---:|"]
        for row in psm_rows:
            status = "OK" if int(row.get("scanner_exit_code", 0)) == 0 else (
                "TIMEOUT" if int(row.get("scanner_exit_code", 0)) == 124 else
                f"ERROR {row.get('scanner_exit_code', 'N/A')}"
            )
            accuracy = f"{row['mean_h4_accuracy']:.2%}" if status == "OK" else "partial/N/A"
            empty_f1 = f"{row['mean_empty_f1']:.2%}" if status == "OK" else "partial/N/A"
            exact = f"{row['exact_grid_rate']:.2%}" if status == "OK" else "partial/N/A"
            runtime = f"{row['mean_runtime_s']:.3f}" if row['mean_runtime_s'] else "N/A"
            lines.append(f"| {row['psm']} | {status} | {accuracy} | {empty_f1} | {exact} | {runtime} |")
        completed_psms = [row for row in psm_rows if int(row.get("scanner_exit_code", 0)) == 0]
        if completed_psms:
            best = max(completed_psms, key=lambda row: row["mean_h4_accuracy"])
            lines += ["", f"Highest mean H4 cell accuracy among completed modes in this dataset: **PSM {best['psm']}**. "
                      "Use this as dataset-specific evidence, not a universal optimum.",
                      f"For a follow-up scan, set `--ocr-psm {best['psm']}`. Keep a different secondary mode (for example PSM 7 unless it is the selected primary mode) and rerun the diagnosis to verify that the combined configuration improves ground-truth accuracy."]
        else:
            lines += ["", "No PSM run completed successfully; do not draw a conclusion from this comparison."]
    if parameter_csv is not None:
        lines += ["", "## Harris parameter study", "",
                  f"The full parameter-study CSV is `{parameter_csv.name}`. Use it with the generated parameter plot "
                  "to discuss candidate count and corner error on the annotated subset."]
    lines += ["", "## Generated figures", ""]
    for name in figures:
        lines.append(f"- `figures/{name}`")
    h4_text = "N/A" if means["H4 accuracy"] is None else f"{means['H4 accuracy'] * 100:.2f}%"
    empty_f1_text = "N/A" if means["Empty-cell F1"] is None else f"{means['Empty-cell F1'] * 100:.2f}%"
    corner_text = "N/A" if means["Corner error (%)"] is None else f"{means['Corner error (%)']:.3f}%"
    runtime_text = "N/A" if means["Runtime (s)"] is None else f"{means['Runtime (s)']:.3f} s"
    lines += ["", "## Suggested report text", "",
              "The following values are generated directly from the annotated dataset. Replace bracketed wording only after inspecting the diagnostic figures and failure crops.", "",
              f"The system was evaluated on {count} annotated photographs. The macro-averaged H4 cell accuracy was {h4_text}, "
              f"with an empty-cell F1 of {empty_f1_text}. Mean normalized corner error was {corner_text} "
              f"and mean runtime was {runtime_text} per image. The most frequent error categories were missed digits "
              f"({error_counts.get('missed_printed_digit', 0)}), digit substitutions "
              f"({error_counts.get('digit_substitution', 0)}) and false positives in empty cells "
              f"({error_counts.get('false_digit_in_empty_cell', 0)}).",
              "",
              "**Interpretation warning:** Do not use OCR confidence as accuracy. All reported accuracy values above are computed by comparison with the manually annotated 9x9 ground-truth grids.",
              ]
    (output_root / "REPORT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_evaluator(annotations: Path, results: Path, out: Path, log: Path) -> list[dict[str, str]]:
    command = [sys.executable, str(ROOT / "evaluate.py"), "--annotations", str(annotations),
               "--results", str(results), "--out", str(out), "--max-confusion-examples", "250"]
    code = _run(command, log)
    if code != 0:
        raise RuntimeError(f"Dataset evaluation failed; inspect {log}")
    metrics = out / "per_image_metrics.csv"
    if not metrics.is_file():
        raise RuntimeError(f"Evaluator did not create {metrics}")
    return _read_csv(metrics)



def _write_dataset_quality(items: list[dict[str, Any]], images_root: Path,
                           annotations_root: Path, output_root: Path) -> dict[str, Any]:
    annotated_paths = {item["image_path"].resolve() for item in items}
    all_images = [path for path in images_root.rglob("*")
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    unannotated = [str(path.relative_to(images_root)) for path in all_images
                   if path.resolve() not in annotated_paths]
    condition_counts = Counter(condition for item in items
                               for condition in item["annotation"].get("conditions", []))
    puzzle_ids = {str(item["annotation"].get("puzzle_id")) for item in items
                  if item["annotation"].get("puzzle_id") not in (None, "")}
    quality = {
        "annotated_photographs": len(items),
        "image_files_found": len(all_images),
        "unannotated_images": unannotated,
        "condition_counts": {key: int(condition_counts.get(key, 0)) for key in CONDITIONS},
        "checks": {
            "at_least_10_annotated_photographs": len(items) >= 10,
            "C1_covered": condition_counts.get("C1", 0) >= 1,
            "C2_at_least_3_oblique_photographs": condition_counts.get("C2", 0) >= 3,
            "C3_covered": condition_counts.get("C3", 0) >= 1,
            "C4_covered": condition_counts.get("C4", 0) >= 1,
            "C5_covered": condition_counts.get("C5", 0) >= 1,
            "at_least_3_puzzle_ids": len(puzzle_ids) >= 3 if puzzle_ids else None,
        },
        "distinct_puzzle_ids_provided": sorted(puzzle_ids),
        "manual_checks": [
            "Verify photographs are original, unedited phone captures with EXIF retained.",
            "Verify every photograph visibly contains the handwritten student-number/date card outside the grid.",
            "Verify there are at least three genuinely different puzzle sources. The annotation format does not require puzzle_id; add an optional puzzle_id field to each JSON if you want the script to count them.",
            "Verify C2 photographs represent different oblique angles, not repeated copies of one viewpoint.",
        ],
    }
    (output_root / "dataset_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    manifest = []
    for item in items:
        manifest.append({
            "image": str(item["image_path"].relative_to(images_root)),
            "annotation": str(item["relative_annotation"]),
            "width_px": item["image_size"][0],
            "height_px": item["image_size"][1],
            "conditions": ";".join(item["annotation"].get("conditions", [])),
            "puzzle_id": item["annotation"].get("puzzle_id", ""),
            "ground_truth_givens": int(np.count_nonzero(np.asarray(item["annotation"]["grid"], dtype=int))),
        })
    _write_csv(output_root / "dataset_manifest.csv", manifest)
    return quality


def run_diagnostics(args: argparse.Namespace) -> int:
    images = args.images.resolve()
    annotations = (args.annotations or args.images).resolve()
    output = args.out.resolve()
    if output == images or images in output.parents:
        raise ValueError("Diagnosis output must be outside the original image folder")
    if output == annotations or annotations in output.parents:
        raise ValueError("Diagnosis output must be outside the annotation folder")
    output.mkdir(parents=True, exist_ok=True)
    items = _discover_dataset(images, annotations)
    # The assignment allows up to 20 seconds per image. Leave additional
    # process-startup and filesystem margin while keeping a stuck OCR mode
    # from hanging the diagnosis indefinitely.
    batch_timeout_s = (args.command_timeout if args.command_timeout > 0
                       else max(90.0, 24.0 * len(items) + 30.0))
    quality = _write_dataset_quality(items, images, annotations, output)
    print(f"Found {len(items)} valid annotated image(s).")
    if quality["unannotated_images"]:
        print(f"WARNING: {len(quality['unannotated_images'])} image(s) do not have matching annotations.")
    failed_checks = [name for name, passed in quality["checks"].items() if passed is False]
    if failed_checks:
        print("Dataset requirement checks not met: " + ", ".join(failed_checks))

    results = args.results.resolve() if args.results else output / "results"
    if args.results is None or args.scan:
        print(f"Scanning {len(items)} annotated image(s)...", flush=True)
        command = [sys.executable, str(ROOT / "sudoku_scan.py"), "--batch", str(images),
                   "--outdir", str(results), "--ocr-psm", str(args.ocr_psm), "--quiet"]
        if args.no_fallback:
            command.append("--no-ocr-fallback")
        else:
            command += ["--ocr-fallback-psm", str(args.fallback_psm)]
        code = _run(command, output / "scanner.log", timeout_s=batch_timeout_s)
        if code != 0:
            print(f"WARNING: scanner returned {code}; failed images remain in the evaluation denominator. See {output / 'scanner.log'}")
        else:
            print("Initial scanner batch completed.", flush=True)

    metrics_dir = output / "evaluation"
    metrics_rows = _run_evaluator(annotations, results, metrics_dir, output / "evaluation.log")
    condition_summary_path = metrics_dir / "condition_summary.json"
    if condition_summary_path.is_file():
        condition_summary = json.loads(condition_summary_path.read_text(encoding="utf-8"))
        condition_rows = [{"condition": condition, **metrics}
                          for condition, metrics in sorted(condition_summary.items())]
        _write_csv(output / "condition_metrics.csv", condition_rows)
    matrix, cell_rows, error_counts = _make_confusion_and_cell_diagnostics(
        items, results, output, args.max_error_crops
    )

    figures = _make_figures(metrics_rows, matrix, output / "figures")
    # Existing assignment plotting workflow supplies the remaining Stage 4 plots.
    plot_code = _run([
        sys.executable, str(ROOT / "tools" / "plot_metrics.py"),
        "--metrics", str(metrics_dir / "per_image_metrics.csv"),
        "--outdir", str(output / "figures"),
    ], output / "plot_metrics.log")
    if plot_code == 0:
        figures.extend(name for name in [
            "corner_residual_vs_obliqueness.png", "grid_straightness_vs_obliqueness.png",
            "ocr_accuracy_vs_obliqueness.png", "grid_spacing_vs_obliqueness.png",
        ] if name not in figures and (output / "figures" / name).is_file())
    comparison_path = output / "figures" / "most_oblique_rectifications.png"
    comparison_code = _run([
        sys.executable, str(ROOT / "tools" / "make_comparison_figure.py"),
        "--results", str(results), "--out", str(comparison_path),
    ], output / "comparison_figure.log")
    if comparison_code == 0 and comparison_path.is_file():
        figures.append(comparison_path.name)

    psm_rows: list[dict[str, Any]] = []
    if args.psm_sweep:
        sweep_root = output / "psm_sweep"
        for psm in (6, 7, 8, 10, 13):
            mode_root = sweep_root / f"psm_{psm}"
            if mode_root.exists():
                shutil.rmtree(mode_root)
            psm_results = mode_root / "results"
            psm_metrics = mode_root / "evaluation"
            print(f"[PSM sweep] Running primary Tesseract PSM {psm} on {len(items)} image(s)...", flush=True)
            started = time.perf_counter()
            code = _run([
                sys.executable, str(ROOT / "sudoku_scan.py"), "--batch", str(images),
                "--outdir", str(psm_results), "--ocr-psm", str(psm),
                "--no-ocr-fallback", "--quiet",
            ], mode_root / "scanner.log", timeout_s=batch_timeout_s)
            elapsed = time.perf_counter() - started
            if code == 124:
                print(f"[PSM sweep] PSM {psm} timed out after {elapsed:.1f}s; partial results will be measured.", flush=True)
            elif code != 0:
                print(f"[PSM sweep] PSM {psm} scanner returned {code}; partial results will be measured.", flush=True)
            try:
                sweep_metrics = _run_evaluator(
                    annotations, psm_results, psm_metrics,
                    mode_root / "evaluation.log",
                )
            except RuntimeError as exc:
                print(f"[PSM sweep] PSM {psm} could not be evaluated: {exc}", flush=True)
                sweep_metrics = []
            psm_rows.append({
                "psm": psm,
                "scanner_exit_code": code,
                "scanner_timed_out": code == 124,
                "mean_h4_accuracy": _mean(sweep_metrics, "cell_accuracy_H4") or 0.0,
                "mean_empty_f1": _mean(sweep_metrics, "empty_cell_f1") or 0.0,
                "exact_grid_rate": _mean(sweep_metrics, "grid_exact_match") or 0.0,
                "mean_runtime_s": _mean(sweep_metrics, "runtime_s") or 0.0,
            })
            print(f"[PSM sweep] PSM {psm} completed in {elapsed:.1f}s (scanner exit {code}).", flush=True)
        _write_csv(output / "psm_sweep_summary.csv", psm_rows)
        if psm_rows:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.bar([str(row["psm"]) for row in psm_rows],
                   [row["mean_h4_accuracy"] * 100 for row in psm_rows])
            ax.set_xlabel("Tesseract primary PSM")
            ax.set_ylabel("Mean H4 cell accuracy (%)")
            ax.set_title("Dataset-measured Tesseract PSM comparison")
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(output / "figures" / "psm_comparison.png", dpi=180)
            plt.close(fig)
            figures.append("psm_comparison.png")

    parameter_csv = None
    if args.parameter_study:
        if len(items) < 6:
            raise ValueError(
                f"The assignment's Harris parameter study needs at least 6 annotated images; found {len(items)}."
            )
        parameter_csv = output / "harris_parameter_sweep.csv"
        limit_args = ["--limit", str(args.parameter_limit)] if args.parameter_limit > 0 else []
        code = _run([
            sys.executable, str(ROOT / "tools" / "parameter_sweep.py"),
            "--images", str(images), "--annotations", str(annotations),
            "--out", str(parameter_csv), *limit_args,
        ], output / "parameter_sweep.log")
        if code != 0:
            print(f"WARNING: Harris parameter study returned {code}; see {output / 'parameter_sweep.log'}")
        elif parameter_csv.is_file():
            plot_path = output / "figures" / "harris_parameter_study.png"
            plot_code = _run([
                sys.executable, str(ROOT / "tools" / "plot_parameter_sweep.py"),
                "--csv", str(parameter_csv), "--out", str(plot_path),
            ], output / "parameter_plot.log")
            if plot_code == 0 and plot_path.is_file():
                figures.append(plot_path.name)

    run_summary = {
        "images_evaluated": len(items),
        "results_directory": str(results),
        "annotations_directory": str(annotations),
        "images_directory": str(images),
        "primary_ocr_psm": args.ocr_psm,
        "fallback_ocr_psm": None if args.no_fallback else args.fallback_psm,
        "psm_sweep_performed": bool(args.psm_sweep),
        "parameter_study_performed": bool(args.parameter_study),
        "error_cells": int(sum(error_counts.values())),
        "correct_cells": int(sum(int(row["correct"]) for row in cell_rows)),
        "total_cells": int(len(cell_rows)),
    }
    (output / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    _write_report_summary(items, metrics_rows, error_counts, output, figures,
                          psm_rows if args.psm_sweep else None, parameter_csv, quality)

    mean_accuracy = _mean(metrics_rows, "cell_accuracy_H4")
    print("\nDiagnosis completed.")
    print(f"Images: {len(items)} | H4 cell accuracy: {mean_accuracy:.2%}" if mean_accuracy is not None
          else f"Images: {len(items)} | H4 cell accuracy: N/A")
    print(f"Correct cells: {run_summary['correct_cells']}/{run_summary['total_cells']}")
    print(f"Errors: {error_counts}")
    print(f"Report-ready summary: {output / 'REPORT_SUMMARY.md'}")
    print(f"Per-cell diagnosis: {output / 'cell_diagnostics.csv'}")
    print(f"Figures: {output / 'figures'}")
    return 0


def _select_folders() -> tuple[Path, Path, Path] | None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as exc:
        print(f"Tkinter folder picker is unavailable: {exc}")
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    messagebox.showinfo("SudokuScan diagnostics", "Select the folder containing your original photographs.", parent=root)
    images = filedialog.askdirectory(title="Select personal dataset images")
    if not images:
        root.destroy()
        return None
    messagebox.showinfo("SudokuScan diagnostics", "Select the folder containing matching Appendix A JSON annotations. Select the same folder if images and JSON files are together.", parent=root)
    annotations = filedialog.askdirectory(title="Select annotation JSON folder")
    if not annotations:
        root.destroy()
        return None
    messagebox.showinfo("SudokuScan diagnostics", "Select where report tables, plots and error crops should be saved.", parent=root)
    output = filedialog.askdirectory(title="Select diagnosis output folder")
    root.destroy()
    if not output:
        return None
    return Path(images), Path(annotations), Path(output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan and diagnose a personal Sudoku dataset against Appendix A ground-truth annotations."
    )
    parser.add_argument("--images", type=Path, help="Original personal image folder (recursive)")
    parser.add_argument("--annotations", type=Path,
                        help="Appendix A JSON folder; defaults to --images for a combined directory")
    parser.add_argument("--results", type=Path,
                        help="Use existing scanner results instead of running a new batch")
    parser.add_argument("--out", type=Path, default=Path("report/personal_diagnosis"))
    parser.add_argument("--ocr-psm", type=int, choices=(6, 7, 8, 10, 13), default=10)
    parser.add_argument("--fallback-psm", type=int, choices=(6, 7, 8, 10, 13), default=7)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--scan", action="store_true", help="Re-run scanner even when --results is supplied")
    parser.add_argument("--psm-sweep", action="store_true",
                        help="Compare PSM 6/7/8/10/13 against ground truth; this runs five additional batches")
    parser.add_argument("--parameter-study", action="store_true",
                        help="Run the 36-setting Harris parameter sweep")
    parser.add_argument("--parameter-limit", type=int, default=6,
                        help="Number of annotated images for Harris study; assignment requires at least 6")
    parser.add_argument("--max-error-crops", type=int, default=40)
    parser.add_argument(
        "--command-timeout", type=float, default=0.0,
        help="Maximum seconds for each scanner batch (default: 24 seconds per image plus 30 seconds, minimum 90).",
    )
    parser.add_argument("--select", action="store_true", help="Open Windows folder pickers for images/annotations/output")
    args = parser.parse_args()

    if args.select:
        selected = _select_folders()
        if selected is None:
            print("Folder selection cancelled.")
            return 2
        args.images, args.annotations, args.out = selected
    if args.images is None:
        parser.error("--images is required unless --select is used")
    if args.parameter_study and args.parameter_limit not in (0,) and args.parameter_limit < 6:
        parser.error("--parameter-limit must be at least 6 for the assignment's required study")
    if args.max_error_crops < 0:
        parser.error("--max-error-crops cannot be negative")
    if args.command_timeout < 0:
        parser.error("--command-timeout cannot be negative")
    if not args.no_fallback and args.ocr_psm == args.fallback_psm:
        parser.error("--fallback-psm must differ from --ocr-psm, or use --no-fallback")
    try:
        return run_diagnostics(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"diagnose_dataset.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
