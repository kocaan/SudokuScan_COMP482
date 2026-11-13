from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import ScanConfig
from .corners import detect_outer_corners, render_harris_debug
from .geometry import (build_rectifications, grid_line_deviations,
                       obliqueness_measure, order_points)
from .preprocessing import preprocess_image, resize_for_detection
from .recognition import best_orientation, recognize_grid
from .solver import solve_sudoku
from .validation import validate_grid
from .visualization import draw_debug, make_ar_overlay


def _matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).tolist()


def _save_grid(path: Path, grid: list[list[int]]) -> None:
    path.write_text("\n".join("".join(str(int(v)) for v in row) for row in grid) + "\n", encoding="utf-8")


def _many_point_homography(candidates_scaled: np.ndarray, scale: float,
                           base_h: np.ndarray, target_size: int) -> tuple[np.ndarray | None, dict]:
    """Bonus B3: Harris grid-intersection candidates -> ideal grid coordinates."""
    if len(candidates_scaled) < 4:
        return None, {"candidate_matches": 0, "inliers": 0}
    original_points = candidates_scaled.astype(np.float32) / float(scale)
    rectified = cv2.perspectiveTransform(original_points.reshape(1, -1, 2), base_h)[0]
    nodes = np.array([[x, y] for y in np.linspace(0, target_size, 10)
                      for x in np.linspace(0, target_size, 10)], dtype=np.float32)
    distances = np.linalg.norm(rectified[:, None, :] - nodes[None, :, :], axis=2)
    nearest_node = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(len(rectified)), nearest_node]
    candidates: dict[int, tuple[float, int]] = {}
    for idx, (node, dist) in enumerate(zip(nearest_node, nearest_dist)):
        if dist > 8.0:
            continue
        prior = candidates.get(int(node))
        if prior is None or float(dist) < prior[0]:
            candidates[int(node)] = (float(dist), idx)
    if len(candidates) < 4:
        return None, {"candidate_matches": len(candidates), "inliers": 0}
    indices = [item[1] for item in candidates.values()]
    src = original_points[indices].reshape(-1, 1, 2)
    dst = nodes[[int(node) for node in candidates.keys()]].reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0,
                                 maxIters=3000, confidence=0.995)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < 4:
        return None, {"candidate_matches": len(candidates), "inliers": inliers}
    return H, {"candidate_matches": len(candidates), "inliers": inliers}


def _not_found_result(image_path: Path, outdir: Path, runtime: float,
                      *, solve: bool = True) -> dict[str, Any]:
    base = image_path.stem
    grid = [[0] * 9 for _ in range(9)]
    _save_grid(outdir / f"{base}_grid.txt", grid)
    result = {
        "image": image_path.name,
        "corners": [], "A3": [], "A4": [], "H4": [],
        "a3_corner_err_px": None, "a4_rms_px": None,
        "grid": grid, "verdict": "not_found", "runtime_s": runtime,
    }
    if solve:
        result["solver"] = {
            "status": "puzzle_not_found", "solution": None,
            "solution_count_capped_at_2": 0, "unique": False,
            "givens_valid": False, "recognized_given_count": 0,
            "solve_time_s": 0.0, "solution_file": None,
        }
    (outdir / f"{base}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def process_image(image_path: str | Path, outdir: str | Path,
                  cfg: ScanConfig | None = None, *, solve: bool = True,
                  ar_overlay: bool = False, many_point: bool = False,
                  orientation_search: bool = False) -> dict[str, Any]:
    cfg = cfg or ScanConfig()
    image_path, outdir = Path(image_path), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Keep OCR inside an 18-second image-level budget, leaving approximately
    # two seconds for geometry, diagnostics and output serialization.
    ocr_deadline = started + 18.0
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    original_gray, normalized, binary = preprocess_image(image)
    small_gray, scale = resize_for_detection(normalized, cfg.max_detection_side)
    small_binary = cv2.resize(binary, (small_gray.shape[1], small_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    detection = detect_outer_corners(small_gray, small_binary, cfg)
    if detection is None:
        runtime = time.perf_counter() - started
        return _not_found_result(image_path, outdir, runtime, solve=solve)

    corners_original = order_points(detection.corners / scale)
    rectifications = build_rectifications(corners_original, cfg.target_size)
    size = cfg.target_size
    source_pts = corners_original.astype(np.float32)
    matrices = {name: value["matrix"] for name, value in rectifications.items()}
    rectified: dict[str, np.ndarray] = {
        "A3": cv2.warpAffine(image, matrices["A3"], (size, size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)),
        "A4": cv2.warpAffine(image, matrices["A4"], (size, size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)),
        "H4": cv2.warpPerspective(image, matrices["H4"], (size, size), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)),
    }

    base = image_path.stem
    for model, warped in rectified.items():
        cv2.imwrite(str(outdir / f"{base}_rect_{model}.png"), warped)

    # The assignment deliverables remain 450x450. For OCR, warp the original
    # photograph directly to a 2x/3x target instead of enlarging an already
    # downsampled 450px image. This retains source detail in small printed glyphs.
    ocr_scale = int(cfg.ocr_scale)
    if ocr_scale == 1:
        ocr_rectified = dict(rectified)
    else:
        scale_matrix = np.array(
            [[ocr_scale, 0.0, 0.0], [0.0, ocr_scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        ocr_size = size * ocr_scale
        ocr_rectified: dict[str, np.ndarray] = {}
        for model in ("A3", "A4", "H4"):
            matrix = matrices[model]
            if model == "H4":
                scaled_matrix = scale_matrix @ matrix
                ocr_rectified[model] = cv2.warpPerspective(
                    image, scaled_matrix, (ocr_size, ocr_size),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
                )
            else:
                scaled_affine = matrix.astype(np.float64) * float(ocr_scale)
                ocr_rectified[model] = cv2.warpAffine(
                    image, scaled_affine, (ocr_size, ocr_size),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
                )

    # Same OCR implementation and thresholds for all three rectifications.
    grids: dict[str, list[list[int]]] = {}
    ocr_stats: dict[str, dict] = {}
    cell_results = None
    for model in ("A3", "A4", "H4"):
        if model == "H4" and orientation_search:
            chosen_img, grid, stats, cells, angle = best_orientation(
                ocr_rectified[model], cfg, deadline=ocr_deadline)
            ocr_rectified[model] = chosen_img
            ocr_stats[model] = {**stats, "orientation_degrees": angle,
                                "orientation_trials": [0, 90, 180, 270],
                                "orientation_trial_ocr_calls": 4}
            cell_results = cells
        else:
            # All three models use the same Otsu + adaptive + raw OCR ensemble;
            # this keeps the A3/A4/H4 comparison methodologically fair.
            grid, stats, cells = recognize_grid(ocr_rectified[model], cfg,
                                                compare_raw=True,
                                                deadline=ocr_deadline)
            ocr_stats[model] = stats
            if model == "H4":
                cell_results = cells
        grids[model] = grid

    final_grid = grids["H4"]
    verdict, conflicts = validate_grid(final_grid)
    raw_grid_h4 = ([[int(cell.raw_digit) for cell in row] for row in cell_results]
                   if cell_results is not None else [[0] * 9 for _ in range(9)])
    prepared_grid_h4 = ([[int(cell.digit) for cell in row] for row in cell_results]
                        if cell_results is not None else [[0] * 9 for _ in range(9)])
    adaptive_grid_h4 = ([[int(cell.alternate_digit) for cell in row] for row in cell_results]
                        if cell_results is not None else [[0] * 9 for _ in range(9)])
    fallback_grid_h4 = (
        [[int(cell.fallback_digit) for cell in row] for row in cell_results]
        if cell_results is not None and ocr_stats.get("H4", {}).get("ocr_fallback_psm") is not None
        else None
    )

    # Preserve representative H4 crops so the report can show raw/prepared OCR
    # examples and diagnosed confusions rather than relying on screenshots alone.
    if cell_results is not None:
        crop_dir = outdir / f"{base}_cells"
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_metadata = []
        for row_idx, row in enumerate(cell_results):
            for col_idx, cell in enumerate(row):
                # Save all 81 cells, including those classified as empty. This
                # lets the evaluator inspect false-empty mistakes as well as
                # OCR digit confusions.
                stem = f"r{row_idx + 1}_c{col_idx + 1}"
                cv2.imwrite(str(crop_dir / f"{stem}_crop.png"), cell.crop)
                cv2.imwrite(str(crop_dir / f"{stem}_prepared.png"), cell.prepared)
                crop_metadata.append({
                    "row": row_idx, "col": col_idx,
                    "is_filled": bool(cell.is_filled),
                    "digit_prepared": int(cell.digit),
                    "digit_adaptive": int(cell.alternate_digit),
                    "digit_raw": int(cell.raw_digit),
                    "digit_selected": int(cell.selected_digit),
                    "digit_fallback": int(cell.fallback_digit),
                    "confidence_fallback": float(cell.fallback_confidence),
                    "confidence_prepared": float(cell.confidence),
                    "confidence_adaptive": float(cell.alternate_confidence),
                    "confidence_raw": float(cell.raw_confidence),
                    "ink_fraction": float(cell.ink_fraction),
                })
                if cell.alternate_prepared is not None:
                    cv2.imwrite(str(crop_dir / f"{stem}_adaptive.png"), cell.alternate_prepared)
        (crop_dir / "cells.json").write_text(json.dumps(crop_metadata, indent=2), encoding="utf-8")

    orientation = int(ocr_stats.get("H4", {}).get("orientation_degrees", 0))
    solved_grid = None
    solution_count = None
    solve_time = None
    recognized_given_count = sum(value != 0 for row in final_grid for value in row)
    if solve:
        solve_start = time.perf_counter()
        if recognized_given_count > 0:
            solved_grid, solution_count = solve_sudoku(final_grid, max_solutions=2)
        else:
            # Never present an arbitrary completion of an all-empty OCR grid
            # as the solution to a photographed puzzle (e.g. Tesseract missing).
            solved_grid, solution_count = None, 0
        solve_time = time.perf_counter() - solve_start
        if solved_grid is not None:
            _save_grid(outdir / f"{base}_solution.txt", solved_grid)
            solved_debug = draw_debug(rectified["H4"], solved_grid, [],
                                      orientation_degrees=orientation)
            cv2.imwrite(str(outdir / f"{base}_debug_solved.png"), solved_debug)

    effective_H = matrices["H4"].copy()
    if orientation == 90:  # cv2.rotate(...CLOCKWISE): (x,y) -> (size-1-y,x)
        rotation = np.array([[0, -1, size - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        effective_H = rotation @ effective_H
    elif orientation == 180:
        rotation = np.array([[-1, 0, size - 1], [0, -1, size - 1], [0, 0, 1]], dtype=np.float64)
        effective_H = rotation @ effective_H
    elif orientation == 270:
        rotation = np.array([[0, 1, 0], [-1, 0, size - 1], [0, 0, 1]], dtype=np.float64)
        effective_H = rotation @ effective_H

    debug = draw_debug(rectified["H4"], final_grid, conflicts, orientation)
    cv2.imwrite(str(outdir / f"{base}_debug.png"), debug)

    # Save the response map and Harris candidates in original-image coordinates.
    harris_vis = render_harris_debug(small_gray, detection.response,
                                     detection.candidates, detection.corners)
    if scale < 1.0:
        harris_vis = cv2.resize(harris_vis, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(outdir / f"{base}_harris.png"), harris_vis)

    many_result = None
    many_grid = None
    many_ocr_stats = None
    many_grid_metrics = None
    if many_point:
        H_many, many_diag = _many_point_homography(detection.candidates, scale,
                                                  matrices["H4"], size)
        many_result = {"diagnostics": many_diag, "matrix": _matrix_list(H_many) if H_many is not None else None}
        if H_many is not None:
            many_warp = cv2.warpPerspective(image, H_many, (size, size), flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            cv2.imwrite(str(outdir / f"{base}_rect_Hmany.png"), many_warp)
            if ocr_scale > 1:
                many_scaled_matrix = np.array(
                    [[ocr_scale, 0.0, 0.0], [0.0, ocr_scale, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ) @ H_many
                many_ocr_warp = cv2.warpPerspective(
                    image, many_scaled_matrix, (size * ocr_scale, size * ocr_scale),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
                )
            else:
                many_ocr_warp = many_warp
            many_grid, many_ocr_stats, _many_cells = recognize_grid(
                many_ocr_warp, cfg, deadline=ocr_deadline)
            mapped_outer = cv2.perspectiveTransform(corners_original.astype(np.float32).reshape(1, 4, 2), H_many)[0]
            ideal_outer = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
            many_diag["outer_corner_rms_px"] = float(np.sqrt(np.mean(np.sum((mapped_outer - ideal_outer) ** 2, axis=1))))
            many_diag["h4_outer_corner_rms_px"] = float(rectifications["H4"]["rms_px"])
            many_grid_metrics = grid_line_deviations(many_warp, size)

    if ar_overlay:
        # A first valid completion is still a valid solved board when the puzzle
        # has multiple completions; uniqueness is reported separately in JSON.
        overlay_grid = solved_grid if solved_grid is not None else final_grid
        cv2.imwrite(str(outdir / f"{base}_ar.png"),
                    make_ar_overlay(image, overlay_grid, effective_H, size))

    _save_grid(outdir / f"{base}_grid.txt", final_grid)
    runtime = time.perf_counter() - started
    corners_list = corners_original.astype(float).tolist()
    result: dict[str, Any] = {
        "image": image_path.name,
        "corners": corners_list,
        "A3": _matrix_list(matrices["A3"]),
        "A4": _matrix_list(matrices["A4"]),
        "H4": _matrix_list(matrices["H4"]),
        "a3_corner_err_px": float(rectifications["A3"]["corner_error_px"]),
        "a3_br_lands_at": cv2.transform(source_pts[2:3].reshape(1, 1, 2), matrices["A3"])[0, 0].astype(float).tolist(),
        "a4_rms_px": float(rectifications["A4"]["rms_px"]),
        "a4_estimator_inliers": rectifications["A4"]["estimator_inliers"],
        "a4_estimator_fallback": rectifications["A4"]["estimator_fallback"],
        "h4_rms_px": float(rectifications["H4"]["rms_px"]),
        "grid": final_grid,
        "grid_A3": grids["A3"],
        "grid_A4": grids["A4"],
        "grid_H4_prepared": prepared_grid_h4,
        "grid_H4_adaptive": adaptive_grid_h4,
        "grid_H4_raw": raw_grid_h4,
        "grid_H4_fallback": fallback_grid_h4,
        "grid_H4_fused": final_grid,
        "ocr": ocr_stats,
        "ocr_internal_scale": ocr_scale,
        "ocr_internal_size_px": int(size * ocr_scale),
        "obliqueness": obliqueness_measure(corners_original),
        "grid_line_metrics": {name: grid_line_deviations(rectified[name], size) for name in rectified},
        "corner_detection": detection.diagnostics,
        "corner_detection_score": float(detection.score),
        "verdict": verdict,
        "conflicts": conflicts,
        "runtime_s": float(runtime),
    }
    if solve:
        if recognized_given_count == 0:
            solver_status = "no_recognized_givens"
        elif solved_grid is None and verdict == "invalid":
            solver_status = "invalid_givens"
        elif solved_grid is None:
            solver_status = "unsatisfiable"
        elif solution_count == 1:
            solver_status = "solved_unique"
        else:
            solver_status = "solved_multiple_solutions"
        result["solver"] = {
            "status": solver_status,
            "solution": solved_grid,
            "solution_count_capped_at_2": solution_count,
            "unique": solution_count == 1,
            "givens_valid": verdict == "valid",
            "recognized_given_count": recognized_given_count,
            "solve_time_s": solve_time,
            "solution_file": f"{base}_solution.txt" if solved_grid is not None else None,
        }
    if many_result is not None:
        result["many_point_homography"] = many_result
        result["grid_Hmany"] = many_grid
        result["ocr_Hmany"] = many_ocr_stats
        result["grid_line_metrics"]["Hmany"] = many_grid_metrics
    (outdir / f"{base}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
