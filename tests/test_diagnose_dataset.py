from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def test_diagnose_dataset_creates_report_tables_plots_and_error_crops(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    source = root / "examples" / "synthetic"
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    results = tmp_path / "results"
    output = tmp_path / "diagnosis"
    images.mkdir()
    annotations.mkdir()
    results.mkdir()

    image = cv2.imread(str(source / "synthetic_01.png"))
    assert image is not None
    assert cv2.imwrite(str(images / "photo_01.png"), image)
    annotation = json.loads((source / "synthetic_01.json").read_text(encoding="utf-8"))
    annotation["image"] = "photo_01.png"
    annotation["conditions"] = ["C2", "C4"]
    (annotations / "photo_01.json").write_text(json.dumps(annotation), encoding="utf-8")

    truth = np.asarray(annotation["grid"], dtype=int)
    prediction = truth.copy()
    missed = next((r, c) for r in range(9) for c in range(9) if truth[r, c] != 0)
    false_positive = next((r, c) for r in range(9) for c in range(9) if truth[r, c] == 0)
    prediction[missed] = 0
    prediction[false_positive] = 7
    prediction_list = prediction.tolist()
    result = {
        "image": "photo_01.png",
        "corners": annotation["corners"],
        "A3": [[1, 0, 0], [0, 1, 0]],
        "A4": [[1, 0, 0], [0, 1, 0]],
        "H4": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "a3_corner_err_px": 2.0,
        "a4_rms_px": 1.0,
        "h4_rms_px": 0.0,
        "grid": prediction_list,
        "grid_A3": prediction_list,
        "grid_A4": prediction_list,
        "grid_H4_prepared": prediction_list,
        "grid_H4_adaptive": prediction_list,
        "grid_H4_raw": prediction_list,
        "grid_H4_fallback": prediction_list,
        "obliqueness": 0.25,
        "grid_line_metrics": {
            model: {"rms_px": 1.0, "spacing_rms_px": 0.5}
            for model in ("A3", "A4", "H4")
        },
        "runtime_s": 2.0,
        "verdict": "valid",
    }
    (results / "photo_01_result.json").write_text(json.dumps(result), encoding="utf-8")
    for model in ("A3", "A4", "H4"):
        assert cv2.imwrite(str(results / f"photo_01_rect_{model}.png"), image[:450, :450])

    cells = results / "photo_01_cells"
    cells.mkdir()
    metadata = []
    for row in range(9):
        for col in range(9):
            metadata.append({"row": row, "col": col, "is_filled": bool(truth[row, col]),
                             "digit_prepared": 0, "digit_adaptive": 0, "digit_raw": 0,
                             "digit_fallback": 0, "confidence_prepared": -1,
                             "confidence_adaptive": -1, "confidence_raw": -1,
                             "confidence_fallback": -1, "ink_fraction": 0.0})
            for suffix in ("crop", "prepared", "adaptive"):
                cv2.imwrite(str(cells / f"r{row+1}_c{col+1}_{suffix}.png"),
                            np.full((50, 50), 255, dtype=np.uint8))
    (cells / "cells.json").write_text(json.dumps(metadata), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(root / "diagnose_dataset.py"),
         "--images", str(images), "--annotations", str(annotations),
         "--results", str(results), "--out", str(output), "--max-error-crops", "5"],
        cwd=root, capture_output=True, text=True, check=False, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    for relative in (
        "REPORT_SUMMARY.md", "dataset_quality.json", "dataset_manifest.csv",
        "cell_diagnostics.csv", "confusion_matrix.csv", "error_type_counts.json",
        "figures/cell_confusion_matrix.png", "figures/accuracy_by_image.png",
        "figures/ocr_accuracy_vs_obliqueness.png", "figures/most_oblique_rectifications.png",
    ):
        assert (output / relative).is_file(), relative
    with (output / "cell_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 81
    assert sum(row["error_type"] != "correct" for row in rows) == 2
    errors = json.loads((output / "error_type_counts.json").read_text(encoding="utf-8"))
    assert errors["missed_printed_digit"] == 1
    assert errors["false_digit_in_empty_cell"] == 1
    assert len(list((output / "error_examples").glob("*.png"))) == 2
    summary = (output / "REPORT_SUMMARY.md").read_text(encoding="utf-8")
    assert "H4 cell accuracy" in summary
    assert "C2" in summary


def test_diagnose_dataset_rejects_out_of_bounds_annotations(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    source_image = cv2.imread(str(root / "examples" / "synthetic" / "synthetic_01.png"))
    cv2.imwrite(str(images / "photo.png"), source_image)
    annotation = json.loads((root / "examples" / "synthetic" / "synthetic_01.json").read_text())
    annotation["image"] = "photo.png"
    annotation["corners"][0][0] = 100000
    (annotations / "photo.json").write_text(json.dumps(annotation))
    completed = subprocess.run(
        [sys.executable, str(root / "diagnose_dataset.py"),
         "--images", str(images), "--annotations", str(annotations),
         "--out", str(tmp_path / "out")],
        cwd=root, capture_output=True, text=True, check=False, timeout=90,
    )
    assert completed.returncode == 2
    assert "outside the original image bounds" in completed.stderr


def test_parameter_study_requires_six_annotated_images(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    source = root / "examples" / "synthetic"
    output = tmp_path / "diagnosis"
    completed = subprocess.run(
        [sys.executable, str(root / "diagnose_dataset.py"),
         "--images", str(source), "--annotations", str(source),
         "--out", str(output), "--parameter-study"],
        cwd=root, capture_output=True, text=True, check=False, timeout=90,
    )
    assert completed.returncode == 2
    assert "needs at least 6 annotated images" in completed.stderr


def test_diagnostic_subprocess_timeout_is_logged_and_returns_124(tmp_path: Path):
    from diagnose_dataset import _run

    log_path = tmp_path / "timed_out_command.log"
    code = _run(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(2)"],
        log_path,
        timeout_s=0.1,
    )
    assert code == 124
    log_text = log_path.read_text(encoding="utf-8")
    assert "started" in log_text
    assert "exceeded timeout" in log_text
