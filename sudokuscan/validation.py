from __future__ import annotations

from collections import defaultdict


def validate_grid(grid: list[list[int]]) -> tuple[str, list[dict]]:
    """Check duplicate givens in every row, column and 3x3 box."""
    conflicts: set[tuple[int, int]] = set()

    def check_group(cells: list[tuple[int, int]]) -> None:
        positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r, c in cells:
            value = int(grid[r][c])
            if 1 <= value <= 9:
                positions[value].append((r, c))
        for locations in positions.values():
            if len(locations) > 1:
                conflicts.update(locations)

    for r in range(9):
        check_group([(r, c) for c in range(9)])
    for c in range(9):
        check_group([(r, c) for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            check_group([(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)])

    ordered = sorted(conflicts)
    details = [{"row": r, "col": c, "cell": [r, c]} for r, c in ordered]
    return ("invalid" if conflicts else "valid"), details
