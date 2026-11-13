"""Utilities for creating and validating Appendix A Sudoku annotations."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

CORNER_LABELS = ("TL", "TR", "BR", "BL")
CONDITIONS = ("C1", "C2", "C3", "C4", "C5")


def validate_corners(
    corners: Sequence[Sequence[float]], image_width: int, image_height: int
) -> tuple[bool, str]:
    """Validate TL, TR, BR, BL points in image coordinates (y increases down)."""
    if len(corners) != 4:
        return False, "Exactly four corners are required (TL, TR, BR, BL)."
    try:
        points = [(float(p[0]), float(p[1])) for p in corners]
    except (TypeError, ValueError, IndexError):
        return False, "Each corner must contain numeric [x, y] coordinates."
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
        return False, "Corner coordinates must be finite numbers."
    for label, (x, y) in zip(CORNER_LABELS, points):
        if not (0 <= x < image_width and 0 <= y < image_height):
            return False, f"{label} is outside the image bounds ({image_width}x{image_height})."

    # Consecutive edge cross-products must be positive for the required
    # clockwise order in image coordinates (where +y points down).
    crosses: list[float] = []
    for i in range(4):
        ax, ay = points[i]
        bx, by = points[(i + 1) % 4]
        cx, cy = points[(i + 2) % 4]
        crosses.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    if any(value <= 1e-6 for value in crosses):
        return False, "Corners are crossed, collinear, or not clockwise TL → TR → BR → BL."

    area2 = sum(
        points[i][0] * points[(i + 1) % 4][1]
        - points[(i + 1) % 4][0] * points[i][1]
        for i in range(4)
    )
    if area2 * 0.5 < max(100.0, image_width * image_height * 0.0001):
        return False, "The selected quadrilateral is unusually small; check that you clicked the outer grid corners."

    edge_lengths = [
        math.hypot(
            points[(i + 1) % 4][0] - points[i][0],
            points[(i + 1) % 4][1] - points[i][1],
        )
        for i in range(4)
    ]
    if min(edge_lengths) < 10:
        return False, "Two selected corners are too close together."
    return True, "OK"


def validate_grid(grid: Sequence[Sequence[int | str]]) -> tuple[bool, str, list[list[int]]]:
    """Return normalized 9x9 integer grid; empty/blank cells become 0."""
    if len(grid) != 9:
        return False, "The Sudoku must contain exactly 9 rows.", []
    normalized: list[list[int]] = []
    for row_index, row in enumerate(grid, start=1):
        if len(row) != 9:
            return False, f"Row {row_index} must contain exactly 9 cells.", []
        out_row: list[int] = []
        for col_index, value in enumerate(row, start=1):
            text = str(value).strip()
            if text == "":
                text = "0"
            if text not in "0123456789" or len(text) != 1:
                return False, f"Cell R{row_index}C{col_index} must be a single digit 0–9.", []
            out_row.append(int(text))
        normalized.append(out_row)
    return True, "OK", normalized


def annotation_path(image_path: Path, images_root: Path, annotations_root: Path) -> Path:
    """Map an image to a same-relative-path JSON annotation."""
    relative = image_path.resolve().relative_to(images_root.resolve())
    return (annotations_root / relative).with_suffix(".json")


def write_annotation(path: Path, payload: dict) -> None:
    """Atomically write JSON so an interruption cannot truncate an annotation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
