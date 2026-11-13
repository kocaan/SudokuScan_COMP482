from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from sudokuscan.config import ScanConfig
from sudokuscan.pipeline import process_image
from sudoku_scan import parse_args


def test_pipeline_writes_required_output_contract(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    image_path = root / "examples" / "synthetic" / "synthetic_01.png"
    assert cv2.imread(str(image_path)) is not None

    result = process_image(image_path, tmp_path, ScanConfig())
    stem = image_path.stem
    required = [
        f"{stem}_grid.txt", f"{stem}_result.json", f"{stem}_harris.png",
        f"{stem}_rect_A3.png", f"{stem}_rect_A4.png", f"{stem}_rect_H4.png",
        f"{stem}_debug.png",
    ]
    assert all((tmp_path / name).is_file() for name in required)

    payload = json.loads((tmp_path / f"{stem}_result.json").read_text(encoding="utf-8"))
    for key in ("corners", "A3", "A4", "H4", "a3_corner_err_px", "a4_rms_px",
                "grid", "verdict", "runtime_s"):
        assert key in payload
    assert len(payload["corners"]) == 4
    assert len(payload["grid"]) == 9
    assert all(len(row) == 9 for row in payload["grid"])
    assert payload["A3"] and payload["A4"] and payload["H4"]
    for variant in ("grid_H4_prepared", "grid_H4_adaptive", "grid_H4_raw", "grid_H4_fused"):
        assert len(payload[variant]) == 9
        assert all(len(row) == 9 for row in payload[variant])
    assert payload["solver"]["status"].startswith("solved_")
    solution = payload["solver"]["solution"]
    assert solution is not None and len(solution) == 9
    assert all(len(row) == 9 and set(row) == set(range(1, 10)) for row in solution)
    assert (tmp_path / f"{stem}_solution.txt").is_file()
    assert (tmp_path / f"{stem}_debug_solved.png").is_file()
    for r in range(9):
        for c in range(9):
            if payload["grid"][r][c] != 0:
                assert solution[r][c] == payload["grid"][r][c]
    crop_dir = tmp_path / f"{stem}_cells"
    crop_metadata = json.loads((crop_dir / "cells.json").read_text(encoding="utf-8"))
    assert len(crop_metadata) == 81
    assert len(list(crop_dir.glob("*_crop.png"))) == 81
    assert len(list(crop_dir.glob("*_prepared.png"))) == 81
    assert len(list(crop_dir.glob("*_adaptive.png"))) == 81
    for model in ("A3", "A4", "H4"):
        metrics = payload["grid_line_metrics"][model]
        assert len(metrics["vertical_positions_px"]) == 10
        assert len(metrics["horizontal_positions_px"]) == 10
        assert len(metrics["vertical_spacing_px"]) == 9
        assert len(metrics["horizontal_spacing_px"]) == 9
        assert metrics["spacing_rms_px"] >= 0


def test_no_puzzle_returns_not_found_without_crashing(tmp_path: Path):
    blank = tmp_path / "blank.png"
    assert cv2.imwrite(str(blank), np.full((320, 320, 3), 255, dtype=np.uint8))
    result = process_image(blank, tmp_path / "out", ScanConfig())
    assert result["verdict"] == "not_found"
    assert (tmp_path / "out" / "blank_grid.txt").is_file()
    payload = json.loads((tmp_path / "out" / "blank_result.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == "not_found"
    assert len(payload["grid"]) == 9


def test_cli_invalid_config_returns_clean_error_code():
    root = Path(__file__).resolve().parents[1]
    image_path = root / "examples" / "synthetic" / "synthetic_01.png"
    completed = subprocess.run(
        [sys.executable, str(root / "sudoku_scan.py"), "--input", str(image_path),
         "--outdir", str(root / ".pytest_cache" / "invalid-config-output"),
         "--response-threshold", "0"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 2
    assert "response_threshold must be in" in completed.stderr
    assert "NameError" not in completed.stderr


def test_cli_uses_configured_harris_threshold_by_default(monkeypatch):
    common = ["sudoku_scan.py", "--input", "photo.png", "--outdir", "out"]
    monkeypatch.setattr("sys.argv", common)
    assert parse_args().response_threshold == ScanConfig().response_threshold == 0.001


def test_solver_is_on_by_default_and_can_be_disabled(monkeypatch):
    common = ["sudoku_scan.py", "--input", "photo.png", "--outdir", "out"]
    monkeypatch.setattr("sys.argv", common)
    assert parse_args().solve is True
    monkeypatch.setattr("sys.argv", common + ["--no-solve"])
    assert parse_args().solve is False


def test_missing_ocr_does_not_write_a_random_sudoku_solution(tmp_path: Path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    image_path = root / "examples" / "synthetic" / "synthetic_01.png"
    monkeypatch.setattr("sudokuscan.recognition._tesseract_executable", lambda: None)

    result = process_image(image_path, tmp_path, ScanConfig())
    assert result["solver"]["status"] == "no_recognized_givens"
    assert result["solver"]["recognized_given_count"] == 0
    assert result["solver"]["solution"] is None
    assert not (tmp_path / "synthetic_01_solution.txt").exists()


def test_recursive_batch_solves_nested_images_and_records_summary(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "photos"
    nested = source / "puzzle_set_b" / "angles"
    nested.mkdir(parents=True)
    image_path = root / "examples" / "synthetic" / "synthetic_01.png"
    assert cv2.imread(str(image_path)) is not None
    # Put the only image below the selected folder, as happens with camera
    # import folders. Option 2 must discover and solve it recursively.
    copied = nested / "angle_01.png"
    copied.write_bytes(image_path.read_bytes())
    output = tmp_path / "results"

    completed = subprocess.run(
        [sys.executable, str(root / "sudoku_scan.py"), "--batch", str(source),
         "--outdir", str(output), "--quiet"],
        cwd=root, capture_output=True, text=True, check=False, timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    result_file = output / "puzzle_set_b" / "angles" / "angle_01_result.json"
    solution_file = output / "puzzle_set_b" / "angles" / "angle_01_solution.txt"
    assert result_file.is_file()
    assert solution_file.is_file()
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    assert payload["solver"]["status"].startswith("solved_")
    summary = json.loads((output / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["images_found"] == 1
    assert summary["processing_failures"] == 0
    assert summary["solver_enabled"] is True


def test_recursive_batch_continues_after_bad_image_and_reports_solver_status(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "photos"
    source.mkdir()
    blank = source / "blank.png"
    assert cv2.imwrite(str(blank), np.full((320, 320, 3), 255, dtype=np.uint8))
    corrupt = source / "corrupt.png"
    corrupt.write_text("not an image", encoding="utf-8")
    output = tmp_path / "results"

    completed = subprocess.run(
        [sys.executable, str(root / "sudoku_scan.py"), "--batch", str(source),
         "--outdir", str(output), "--quiet"],
        cwd=root, capture_output=True, text=True, check=False, timeout=90,
    )
    assert completed.returncode == 1
    blank_result = json.loads((output / "blank_result.json").read_text(encoding="utf-8"))
    corrupt_result = json.loads((output / "corrupt_result.json").read_text(encoding="utf-8"))
    assert blank_result["solver"]["status"] == "puzzle_not_found"
    assert corrupt_result["solver"]["status"] == "not_run_due_to_processing_error"
    summary = json.loads((output / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["images_found"] == 2
    assert summary["processing_failures"] == 1
    assert {item["image"] for item in summary["images"]} == {"blank.png", "corrupt.png"}


def test_batch_output_parent_of_input_does_not_skip_source_images(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "input_photos"
    source.mkdir()
    image_path = root / "examples" / "synthetic" / "synthetic_01.png"
    (source / "photo.png").write_bytes(image_path.read_bytes())
    # An output directory above the source directory is valid and must not be
    # treated as an output tree nested inside the source.
    output_parent = tmp_path
    completed = subprocess.run(
        [sys.executable, str(root / "sudoku_scan.py"), "--batch", str(source),
         "--outdir", str(output_parent), "--quiet"],
        cwd=root, capture_output=True, text=True, check=False, timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_parent / "photo_result.json").is_file()
    summary = json.loads((output_parent / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["images_found"] == 1
