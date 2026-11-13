from __future__ import annotations

import cv2
import numpy as np

from .geometry import order_points


def draw_debug(rectified: np.ndarray, grid: list[list[int]], conflicts: list[dict],
               orientation_degrees: int = 0) -> np.ndarray:
    vis = rectified.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    for i in range(10):
        x = int(round(i * w / 9))
        y = int(round(i * h / 9))
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(vis, (x, 0), (x, h - 1), (40, 180, 40), thickness, cv2.LINE_AA)
        cv2.line(vis, (0, y), (w - 1, y), (40, 180, 40), thickness, cv2.LINE_AA)
    conflict_set = {(int(x["row"]), int(x["col"])) for x in conflicts}
    for r in range(9):
        for c in range(9):
            x0, x1 = int(c * w / 9), int((c + 1) * w / 9)
            y0, y1 = int(r * h / 9), int((r + 1) * h / 9)
            if (r, c) in conflict_set:
                cv2.rectangle(vis, (x0 + 1, y0 + 1), (x1 - 2, y1 - 2), (0, 0, 255), 2)
            value = int(grid[r][c])
            if value:
                text = str(value)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = min(w, h) / 650.0
                (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
                tx = x0 + (x1 - x0 - tw) // 2
                ty = y0 + (y1 - y0 + th) // 2
                color = (0, 0, 255) if (r, c) in conflict_set else (20, 20, 220)
                cv2.putText(vis, text, (tx, ty), font, scale, color, 1, cv2.LINE_AA)
    cv2.putText(vis, f"Orientation: {orientation_degrees} deg", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
    return vis


def make_ar_overlay(original: np.ndarray, rectified_grid: list[list[int]],
                    source_to_target: np.ndarray, size: int = 450) -> np.ndarray:
    """Draw recognized/solved digits in rectified space and map them to the photo."""
    overlay = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(9):
        for c in range(9):
            value = int(rectified_grid[r][c])
            if value == 0:
                continue
            center = (int((c + 0.5) * size / 9), int((r + 0.5) * size / 9))
            cv2.putText(overlay, str(value), center, cv2.FONT_HERSHEY_SIMPLEX,
                        0.82, (0, 0, 255), 2, cv2.LINE_AA)
    inverse = np.linalg.inv(source_to_target)
    h, w = original.shape[:2]
    warped = cv2.warpPerspective(overlay, inverse, (w, h), flags=cv2.INTER_LINEAR)
    alpha_source = cv2.cvtColor(cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    alpha = cv2.warpPerspective(alpha_source, inverse, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY) > 8
    result = original.copy()
    # Blend only where the projected glyph strokes exist.
    result[mask] = cv2.addWeighted(original[mask], 0.30, warped[mask], 0.70, 0)
    return result


def render_corner_overlay(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    vis = image.copy()
    for i, (x, y) in enumerate(order_points(corners)):
        cv2.circle(vis, (int(round(x)), int(round(y))), 10, (0, 255, 0), 3)
        cv2.putText(vis, ["TL", "TR", "BR", "BL"][i],
                    (int(x) + 9, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis
