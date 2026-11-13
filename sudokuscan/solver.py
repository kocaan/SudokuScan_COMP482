from __future__ import annotations

from typing import Any, Optional

from .validation import validate_grid


FULL = (1 << 9) - 1


def _copy_and_validate_shape(grid: list[list[int]]) -> list[list[int]]:
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        raise ValueError("Sudoku grid must be 9x9")
    board: list[list[int]] = []
    for row in grid:
        converted: list[int] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int,)):
                # NumPy integers are accepted without making the module depend
                # on NumPy just for validation.
                try:
                    import numpy as np
                    is_integer = isinstance(value, np.integer)
                except Exception:
                    is_integer = False
                if not is_integer:
                    raise ValueError("Grid values must be integers in the range 0..9")
            integer = int(value)
            if not 0 <= integer <= 9:
                raise ValueError("Grid values must be integers in the range 0..9")
            converted.append(integer)
        board.append(converted)
    return board


def solve_sudoku(grid: list[list[int]], max_solutions: int = 2
                 ) -> tuple[Optional[list[list[int]]], int]:
    """Solve a 9x9 Sudoku using bit masks and minimum-remaining-values search.

    Returns ``(first_solution, count)``. The count is capped at
    ``max_solutions``; a count of one means uniqueness was exhaustively
    established, while two means at least two solutions exist.
    """
    if isinstance(max_solutions, bool) or not isinstance(max_solutions, int) or max_solutions < 1:
        raise ValueError("max_solutions must be a positive integer")
    board = _copy_and_validate_shape(grid)
    row_masks = [0] * 9
    col_masks = [0] * 9
    box_masks = [0] * 9
    for r in range(9):
        for c in range(9):
            value = board[r][c]
            if value == 0:
                continue
            bit = 1 << (value - 1)
            box = (r // 3) * 3 + c // 3
            if row_masks[r] & bit or col_masks[c] & bit or box_masks[box] & bit:
                return None, 0
            row_masks[r] |= bit
            col_masks[c] |= bit
            box_masks[box] |= bit

    first_solution: Optional[list[list[int]]] = None
    solution_count = 0

    def search() -> None:
        nonlocal first_solution, solution_count
        if solution_count >= max_solutions:
            return

        best_cell: tuple[int, int, int] | None = None
        best_mask = 0
        best_count = 10
        for row in range(9):
            for col in range(9):
                if board[row][col] != 0:
                    continue
                box = (row // 3) * 3 + col // 3
                candidates = FULL & ~(row_masks[row] | col_masks[col] | box_masks[box])
                count = candidates.bit_count()
                if count == 0:
                    return
                if count < best_count:
                    best_count = count
                    best_cell = (row, col, box)
                    best_mask = candidates
                    if count == 1:
                        break
            if best_count == 1:
                break

        if best_cell is None:
            solution_count += 1
            if first_solution is None:
                first_solution = [row[:] for row in board]
            return

        row, col, box = best_cell
        remaining = best_mask
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            board[row][col] = bit.bit_length()
            row_masks[row] |= bit
            col_masks[col] |= bit
            box_masks[box] |= bit
            search()
            row_masks[row] ^= bit
            col_masks[col] ^= bit
            box_masks[box] ^= bit
            board[row][col] = 0
            if solution_count >= max_solutions:
                return

    search()
    return first_solution, solution_count


def reconcile_ocr_candidates(
    preferred_grid: list[list[int]],
    cells: list[list[Any]],
    *,
    max_nodes: int = 1000,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Use Sudoku constraints to choose between disagreeing OCR readings.

    A search is only needed when the preferred OCR grid is invalid or cannot
    be completed. Candidate readings come exclusively from the independent
    Otsu, adaptive-threshold and raw Tesseract passes. No digit is fabricated
    by this stage; the Sudoku solver fills only cells left blank by OCR.
    """
    preferred = _copy_and_validate_shape(preferred_grid)
    if not any(value for row in preferred for value in row):
        return preferred, {
            "attempted": False,
            "changed_cells": [],
            "search_nodes": 0,
            "preferred_grid_solvable": False,
            "resolution": "no recognized givens; skipped candidate solving",
        }
    options: list[list[list[tuple[int, float]]]] = [[[] for _ in range(9)] for _ in range(9)]

    source_weights = (("digit", "confidence", 1.0),
                      ("alternate_digit", "alternate_confidence", 0.95),
                      ("raw_digit", "raw_confidence", 0.75),
                      ("fallback_digit", "fallback_confidence", 0.85))
    for row in range(9):
        for col in range(9):
            cell = cells[row][col]
            if not bool(getattr(cell, "is_filled", False)):
                continue
            grouped: dict[int, list[tuple[float, float]]] = {}
            for digit_attr, confidence_attr, weight in source_weights:
                digit = int(getattr(cell, digit_attr, 0) or 0)
                confidence = float(getattr(cell, confidence_attr, -1.0))
                # Do not re-introduce weak one-off OCR guesses during Sudoku
                # constraint reconciliation. Empty/noisy cells must remain blank
                # unless a recognizer supplied reasonably strong evidence.
                if 1 <= digit <= 9 and confidence >= 55.0:
                    grouped.setdefault(digit, []).append((confidence, weight))
            for digit, votes in grouped.items():
                weight_sum = sum(weight for _confidence, weight in votes)
                weighted_confidence = sum(conf * weight for conf, weight in votes) / max(1e-9, weight_sum)
                agreement_bonus = 7.0 * max(0, len(votes) - 1)
                options[row][col].append((digit, weighted_confidence + agreement_bonus))
            options[row][col].sort(key=lambda item: item[1], reverse=True)

    preferred_verdict, _ = validate_grid(preferred)
    preferred_solution, preferred_count = solve_sudoku(preferred, max_solutions=2)
    if preferred_verdict == "valid" and preferred_solution is not None:
        return preferred, {
            "attempted": False,
            "changed_cells": [],
            "search_nodes": 0,
            "preferred_grid_solvable": True,
            "solution_count_capped_at_2": preferred_count,
        }

    # Only cells with multiple OCR alternatives can change the reading. Cells
    # with one candidate are fixed; empty cells remain unknown and are solved.
    variables: list[tuple[int, int, list[tuple[int, float]]]] = []
    fixed: list[tuple[int, int, int]] = []
    for row in range(9):
        for col in range(9):
            choices = options[row][col]
            if not choices:
                continue
            if len(choices) == 1:
                fixed.append((row, col, choices[0][0]))
            else:
                variables.append((row, col, choices))

    # Branch the smallest ambiguity sets first. A deterministic node cap and
    # first-feasible stopping rule prevent noisy pages from consuming the
    # image time budget.
    variables.sort(key=lambda item: (len(item[2]), -item[2][0][1], item[0], item[1]))
    board = [[0] * 9 for _ in range(9)]
    row_masks = [0] * 9
    col_masks = [0] * 9
    box_masks = [0] * 9
    base_score = 0.0

    def place(row: int, col: int, digit: int) -> bool:
        bit = 1 << (digit - 1)
        box = (row // 3) * 3 + col // 3
        if row_masks[row] & bit or col_masks[col] & bit or box_masks[box] & bit:
            return False
        board[row][col] = digit
        row_masks[row] |= bit
        col_masks[col] |= bit
        box_masks[box] |= bit
        return True

    for row, col, digit in fixed:
        if not place(row, col, digit):
            return preferred, {
                "attempted": True, "changed_cells": [], "search_nodes": 0,
                "preferred_grid_solvable": False,
                "resolution": "no consistent assignment among available OCR alternatives",
            }
        base_score += options[row][col][0][1]

    best_score = float("-inf")
    best_grid: list[list[int]] | None = None
    best_solution_count = 0
    best_nodes = 0
    nodes = 0
    found_solution = False

    def search(index: int, score: float) -> None:
        nonlocal nodes, best_score, best_grid, best_solution_count, best_nodes, found_solution
        if nodes >= max_nodes or found_solution:
            return
        nodes += 1
        if index == len(variables):
            candidate = [line[:] for line in board]
            solution, count = solve_sudoku(candidate, max_solutions=2)
            if solution is None:
                return
            # Variables and each variable's alternatives are visited in OCR
            # confidence order. Stop at the first completable reading set to
            # bound runtime on difficult/ambiguous images.
            best_score = score + (0.25 if count == 1 else 0.0)
            best_grid = candidate
            best_solution_count = count
            best_nodes = nodes
            found_solution = True
            return

        row, col, choices = variables[index]
        for digit, confidence_score in choices:
            if place(row, col, digit):
                search(index + 1, score + confidence_score)
                bit = 1 << (digit - 1)
                box = (row // 3) * 3 + col // 3
                row_masks[row] ^= bit
                col_masks[col] ^= bit
                box_masks[box] ^= bit
                board[row][col] = 0
            if nodes >= max_nodes:
                break

    search(0, base_score)
    if best_grid is None:
        return preferred, {
            "attempted": True, "changed_cells": [], "search_nodes": nodes,
            "preferred_grid_solvable": False,
            "resolution": "no solvable combination found within OCR candidate set",
        }

    changed = [
        [row, col]
        for row in range(9) for col in range(9)
        if preferred[row][col] != best_grid[row][col]
    ]
    return best_grid, {
        "attempted": True,
        "changed_cells": changed,
        "search_nodes": nodes,
        "preferred_grid_solvable": False,
        "solution_count_capped_at_2": best_solution_count,
        "resolution": "selected OCR alternatives that produce a solvable Sudoku",
    }
