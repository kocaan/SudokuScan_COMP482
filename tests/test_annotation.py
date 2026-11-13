from __future__ import annotations

import json
from pathlib import Path

from sudokuscan.annotation import annotation_path, validate_corners, validate_grid, write_annotation


def test_validate_corners_accepts_clockwise_tilted_quad():
    corners = [[300.0, 500.0], [800.0, 200.0], [1100.0, 700.0], [600.0, 1000.0]]
    ok, message = validate_corners(corners, image_width=1200, image_height=1200)
    assert ok, message


def test_validate_corners_rejects_wrong_order_and_out_of_bounds():
    valid_quad = [[300.0, 500.0], [800.0, 200.0], [1100.0, 700.0], [600.0, 1000.0]]
    # Swapping TR and BR creates a crossed/non-convex click sequence.
    wrong_order = [valid_quad[0], valid_quad[2], valid_quad[1], valid_quad[3]]
    ok, _ = validate_corners(wrong_order, 1200, 1200)
    assert not ok

    out_of_bounds = [[10, 10], [900, 10], [900, 900], [10, 9000]]
    ok, message = validate_corners(out_of_bounds, 1000, 1000)
    assert not ok
    assert "outside" in message


def test_validate_corners_rejects_non_finite_and_too_few_points():
    ok, _ = validate_corners([[0, 0], [100, 0], [100, 100]], 200, 200)
    assert not ok
    ok, _ = validate_corners([[0, 0], [100, 0], [100, float("nan")], [0, 100]], 200, 200)
    assert not ok


def test_validate_grid_normalizes_blank_to_zero_and_rejects_bad_values():
    source = [["0"] * 9 for _ in range(9)]
    source[0][0] = "5"
    source[0][1] = ""
    ok, message, grid = validate_grid(source)
    assert ok, message
    assert grid[0][:2] == [5, 0]

    source[0][0] = "10"
    ok, _, _ = validate_grid(source)
    assert not ok


def test_annotation_path_preserves_nested_relative_structure(tmp_path: Path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    image = images / "set_A" / "angle_01.jpg"
    image.parent.mkdir(parents=True)
    image.touch()
    assert annotation_path(image, images, annotations) == annotations / "set_A" / "angle_01.json"


def test_write_annotation_is_valid_json_and_creates_parent(tmp_path: Path):
    target = tmp_path / "annotations" / "nested" / "photo.json"
    payload = {"image": "photo.jpg", "conditions": ["C2"], "corners": [], "grid": [[0] * 9 for _ in range(9)]}
    write_annotation(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not list(target.parent.glob("*.tmp"))
