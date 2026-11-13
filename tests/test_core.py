from __future__ import annotations

import numpy as np
import pytest

from sudokuscan.config import ScanConfig
from sudokuscan.corners import _candidate_match
from sudokuscan.geometry import affine_least_squares, build_rectifications, order_points
from sudokuscan.solver import reconcile_ocr_candidates, solve_sudoku
from sudokuscan.validation import validate_grid


def test_corner_ordering_and_rectification():
    points = [[20, 10], [220, 30], [210, 230], [10, 210]]
    ordered = order_points(points)
    assert ordered.shape == (4, 2)
    rects = build_rectifications(ordered, 450)
    assert rects["A3"]["matrix"].shape == (2, 3)
    assert rects["A4"]["matrix"].shape == (2, 3)
    assert rects["H4"]["matrix"].shape == (3, 3)
    assert rects["A4"]["estimator_inliers"] == 4
    assert rects["A4"]["estimator_fallback"] is False
    assert rects["H4"]["rms_px"] < 1e-3
    assert rects["A3"]["corner_error_px"] >= 0

    target = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype=np.float32)
    expected_a4 = affine_least_squares(ordered, target)
    np.testing.assert_allclose(rects["A4"]["matrix"], expected_a4, atol=1e-4)


def test_order_points_handles_assignment_rotation_range():
    base = np.array([[-100, -100], [100, -100], [100, 100], [-100, 100]], dtype=np.float32)
    for degrees in (-30, -20, 0, 20, 30):
        angle = np.deg2rad(degrees)
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        points = base @ rotation.T + np.array([300, 300])
        ordered = order_points(points)
        assert ordered.shape == (4, 2)
        assert len(np.unique(np.round(ordered, 2), axis=0)) == 4
        # Cyclic order must have positive image-plane polygon area.
        first = ordered[1] - ordered[0]
        second = ordered[2] - ordered[1]
        cross = float(first[0] * second[1] - first[1] * second[0])
        assert cross > 0


def test_a4_uses_all_four_correspondences_for_least_squares():
    src = np.array([[0, 0], [100, 0], [90, 100], [10, 100]], dtype=np.float32)
    dst = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype=np.float32)
    result = build_rectifications(src, 450)["A4"]
    assert result["estimator_inliers"] == 4
    assert result["rms_px"] > 0
    fitted = result["matrix"]
    mapped = np.asarray([fitted @ np.array([x, y, 1.0]) for x, y in src])
    expected = affine_least_squares(src, dst)
    np.testing.assert_allclose(fitted, expected, atol=1e-4)
    assert np.isfinite(mapped).all()


def test_validity_detects_row_column_and_box_conflicts():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 5
    grid[0][1] = 5  # row and box conflict
    grid[1][2] = 6
    grid[2][2] = 6  # column and box conflict
    grid[1][1] = 8
    grid[2][0] = 8  # box-only conflict
    verdict, conflicts = validate_grid(grid)
    assert verdict == "invalid"
    assert {tuple(item["cell"]) for item in conflicts} == {
        (0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2)
    }


def test_solver_reports_unique_solution():
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    solution, count = solve_sudoku(puzzle)
    assert solution is not None
    assert count == 1
    assert solution[0] == [5, 3, 4, 6, 7, 8, 9, 1, 2]
    digits = set(range(1, 10))
    assert all(set(row) == digits for row in solution)
    assert all({solution[r][c] for r in range(9)} == digits for c in range(9))
    assert all({solution[r][c] for r in range(br, br + 3)
                for c in range(bc, bc + 3)} == digits
               for br in (0, 3, 6) for bc in (0, 3, 6))
    for r in range(9):
        for c in range(9):
            if puzzle[r][c]:
                assert solution[r][c] == puzzle[r][c]


def test_config_rejects_invalid_opencv_parameters():
    with pytest.raises(ValueError):
        ScanConfig(harris_ksize=4)
    with pytest.raises(ValueError):
        ScanConfig(harris_block_size=1)
    with pytest.raises(ValueError):
        ScanConfig(response_threshold=0)


def test_corner_matching_never_reuses_one_harris_candidate():
    cfg = ScanConfig(corner_match_radius_fraction=0.25)
    proposal = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], dtype=np.float32)
    candidates = np.array([[50, 50], [45, 50], [55, 50], [50, 55]], dtype=np.float32)
    response = np.zeros((100, 100), dtype=np.float32)
    response[50, 50] = 100.0
    response[50, 45] = 90.0
    response[50, 55] = 80.0
    response[55, 50] = 70.0
    matched, score, _distances = _candidate_match(proposal, candidates, response, cfg)
    assert matched is not None and score > 0
    assert len(np.unique(np.round(matched, 3), axis=0)) == 4
    for point in matched:
        assert np.any(np.all(np.isclose(candidates, point), axis=1))


def test_solver_rejects_conflicting_givens_and_unsatisfiable_puzzle():
    duplicate = [[0] * 9 for _ in range(9)]
    duplicate[0][0] = 4
    duplicate[0][1] = 4
    assert solve_sudoku(duplicate) == (None, 0)

    # This alteration creates no immediate row/column/box duplicate, but makes
    # the otherwise standard puzzle impossible to complete.
    impossible = [
        [1, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    assert solve_sudoku(impossible) == (None, 0)


def test_solver_validates_shape_values_and_solution_limit():
    with pytest.raises(ValueError):
        solve_sudoku([[0] * 9 for _ in range(8)])
    with pytest.raises(ValueError):
        solve_sudoku([[0] * 8 + [10] if r == 0 else [0] * 9 for r in range(9)])
    with pytest.raises(ValueError):
        solve_sudoku([[0.0] * 9 for _ in range(9)])
    with pytest.raises(ValueError):
        solve_sudoku([[0] * 9 for _ in range(9)], max_solutions=0)


def test_ocr_candidate_reconciliation_uses_sudoku_constraints():
    from types import SimpleNamespace

    cells = [[SimpleNamespace(
        is_filled=False, digit=0, confidence=100.0,
        alternate_digit=0, alternate_confidence=100.0,
        raw_digit=0, raw_confidence=100.0,
    ) for _ in range(9)] for _ in range(9)]
    # The preferred readings duplicate 1 in the first row. The adaptive pass
    # independently reads the second cell as 2, yielding a solvable puzzle.
    cells[0][0] = SimpleNamespace(
        is_filled=True, digit=1, confidence=96.0,
        alternate_digit=1, alternate_confidence=95.0,
        raw_digit=1, raw_confidence=94.0,
    )
    cells[0][1] = SimpleNamespace(
        is_filled=True, digit=1, confidence=96.0,
        alternate_digit=2, alternate_confidence=95.0,
        raw_digit=1, raw_confidence=94.0,
    )
    preferred = [[0] * 9 for _ in range(9)]
    preferred[0][0] = 1
    preferred[0][1] = 1

    repaired, diagnostics = reconcile_ocr_candidates(preferred, cells)
    assert repaired[0][0] == 1
    assert repaired[0][1] == 2
    assert diagnostics["attempted"] is True
    assert [0, 1] in diagnostics["changed_cells"]
    solution, count = solve_sudoku(repaired)
    assert solution is not None and count >= 1
