from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_evaluator_counts_missing_result_as_robustness_failure(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    annotations = tmp_path / "annotations"
    results = tmp_path / "results"
    output = tmp_path / "evaluation"
    annotations.mkdir()
    results.mkdir()

    grid = [[0] * 9 for _ in range(9)]
    for name in ("present", "missing"):
        (annotations / f"{name}.json").write_text(
            json.dumps({"image": f"{name}.png", "conditions": ["C1"],
                        "corners": [[0, 0], [450, 0], [450, 450], [0, 450]],
                        "grid": grid}),
            encoding="utf-8",
        )

    (results / "present_result.json").write_text(
        json.dumps({"image": "present.png", "corners": [], "grid": grid,
                    "runtime_s": 1.0, "verdict": "valid"}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(root / "evaluate.py"),
         "--annotations", str(annotations), "--results", str(results),
         "--out", str(output)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Missing result (counting as failed)" in completed.stdout

    with (output / "per_image_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    missing = next(row for row in rows if row["image"] == "missing")
    assert missing["result_missing"] == "True"
    assert missing["robust"] == "False"

    summary = json.loads((output / "condition_summary.json").read_text(encoding="utf-8"))
    assert summary["C1"]["images"] == 2
    assert summary["C1"]["robustness_rate"] == 0.5


def test_evaluator_handles_non_numeric_runtime_as_not_robust():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from evaluate import evaluate_pair

    grid = [[0] * 9 for _ in range(9)]
    annotation = {"grid": grid}
    result = {"grid": grid, "runtime_s": "not-a-number"}
    metrics = evaluate_pair(annotation, result)
    assert metrics["robust"] is False


def test_evaluator_reports_each_h4_preprocessing_variant():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from evaluate import evaluate_pair

    truth = [[0] * 9 for _ in range(9)]
    truth[0][0] = 5
    raw = [[0] * 9 for _ in range(9)]
    prepared = [[0] * 9 for _ in range(9)]
    adaptive = [[0] * 9 for _ in range(9)]
    fused = [[0] * 9 for _ in range(9)]
    prepared[0][0] = 3
    adaptive[0][0] = 5
    raw[0][0] = 5
    fused[0][0] = 5
    metrics = evaluate_pair(
        {"grid": truth},
        {"grid": fused, "grid_H4_prepared": prepared,
         "grid_H4_adaptive": adaptive, "grid_H4_raw": raw},
    )
    assert metrics["cell_accuracy_H4_prepared"] < 1.0
    assert metrics["cell_accuracy_H4_adaptive"] == 1.0
    assert metrics["cell_accuracy_H4_raw"] == 1.0
    assert metrics["cell_accuracy_H4_fused"] == 1.0


def test_evaluator_finds_results_in_recursive_dataset_structure(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    annotations = tmp_path / "annotations"
    results = tmp_path / "results"
    nested = Path("puzzle_a") / "angles"
    (annotations / nested).mkdir(parents=True)
    (results / nested).mkdir(parents=True)
    grid = [[0] * 9 for _ in range(9)]
    (annotations / nested / "photo.json").write_text(
        json.dumps({"image": "photo.jpg", "conditions": ["C2"],
                    "corners": [[0, 0], [450, 0], [450, 450], [0, 450]],
                    "grid": grid}), encoding="utf-8",
    )
    (results / nested / "photo_result.json").write_text(
        json.dumps({"image": "photo.jpg", "corners": [[0, 0], [450, 0], [450, 450], [0, 450]],
                    "grid": grid, "runtime_s": 1.0, "verdict": "valid"}),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(root / "evaluate.py"),
         "--annotations", str(annotations), "--results", str(results),
         "--out", str(tmp_path / "evaluation")],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with (tmp_path / "evaluation" / "per_image_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["result_missing"] == "False"
    assert rows[0]["robust"] == "True"


def test_evaluator_copies_confusion_crops_from_nested_result_folder(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    annotations = tmp_path / "annotations"
    results = tmp_path / "results"
    nested = Path("puzzle_b") / "camera_import"
    annotation_dir = annotations / nested
    result_dir = results / nested
    cells_dir = result_dir / "photo_cells"
    annotation_dir.mkdir(parents=True)
    cells_dir.mkdir(parents=True)

    truth = [[0] * 9 for _ in range(9)]
    truth[0][0] = 8
    pred = [[0] * 9 for _ in range(9)]
    (annotation_dir / "photo.json").write_text(
        json.dumps({"image": "photo.jpg", "conditions": ["C2"],
                    "corners": [[0, 0], [450, 0], [450, 450], [0, 450]],
                    "grid": truth}), encoding="utf-8",
    )
    (result_dir / "photo_result.json").write_text(
        json.dumps({"image": "photo.jpg", "corners": [[0, 0], [450, 0], [450, 450], [0, 450]],
                    "grid": pred, "runtime_s": 1.0, "verdict": "valid"}),
        encoding="utf-8",
    )
    for suffix in ("_crop.png", "_prepared.png", "_adaptive.png"):
        (cells_dir / f"r1_c1{suffix}").write_bytes(b"test crop")

    output = tmp_path / "evaluation"
    completed = subprocess.run(
        [sys.executable, str(root / "evaluate.py"),
         "--annotations", str(annotations), "--results", str(results),
         "--out", str(output)], capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    copied = output / "confusion_examples" / nested
    assert (copied / "photo_r1_c1_crop.png").is_file()
    assert (copied / "photo_r1_c1_prepared.png").is_file()
    assert (copied / "photo_r1_c1_adaptive.png").is_file()


def test_evaluator_reports_fallback_grid_accuracy():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from evaluate import evaluate_pair

    truth = [[0] * 9 for _ in range(9)]
    truth[0][0] = 6
    fallback = [[0] * 9 for _ in range(9)]
    fallback[0][0] = 6
    metrics = evaluate_pair(
        {"grid": truth},
        {"grid": fallback, "grid_H4_fallback": fallback},
    )
    assert metrics["cell_accuracy_H4_fallback"] == 1.0
