from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import ScanConfig


@dataclass
class CellRecognition:
    # `digit` is the primary, Otsu-prepared reading. Keep it separate from the
    # fused reading so the report can compare preparation methods honestly.
    digit: int
    confidence: float
    is_filled: bool
    ink_fraction: float
    raw_digit: int
    raw_confidence: float
    crop: np.ndarray
    prepared: np.ndarray
    alternate_digit: int = 0
    alternate_confidence: float = -1.0
    alternate_prepared: np.ndarray | None = None
    selected_digit: int = 0
    fallback_digit: int = 0
    fallback_confidence: float = -1.0


def _cell_crop(cell: np.ndarray, cfg: ScanConfig) -> np.ndarray:
    h, w = cell.shape[:2]
    margin = max(1, int(round(min(h, w) * cfg.cell_margin_fraction)))
    x1, x2 = min(margin, w // 3), max(w - margin, w - (w // 3))
    y1, y2 = min(margin, h // 3), max(h - margin, h - (h // 3))
    crop = cell[y1:y2, x1:x2]
    return crop if crop.size else cell.copy()


def _gray(cell: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell.copy()


def _binarize(cell: np.ndarray) -> np.ndarray:
    gray = _gray(cell)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _threshold, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def _remove_cell_border_artifacts(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Discard thin line remnants along the edges of the inner cell crop.

    The crop is inset from the grid lines. A long, thin component that still
    reaches one of its edges is likely a residual grid segment. The edge test
    is important: a centred, narrow printed digit such as ``1`` must not be
    discarded merely because it is vertically elongated.
    """
    if binary.size == 0:
        return binary.copy(), 0
    h, w = binary.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros_like(binary)
    removed = 0
    for component_id in range(1, count):
        x, y, width, height, area = stats[component_id]
        edge_band = max(3, int(round(0.12 * min(h, w))))
        at_horizontal_edge = y <= edge_band or y + height >= h - edge_band
        at_vertical_edge = x <= edge_band or x + width >= w - edge_band
        horizontal_line = (
            width >= 0.58 * w and height <= max(4, int(0.16 * h))
            and at_horizontal_edge
        )
        vertical_line = (
            height >= 0.58 * h and width <= max(4, int(0.16 * w))
            and at_vertical_edge
        )
        # At an oblique angle, interpolation can break a border into many tiny
        # dashes. Those components no longer satisfy the long-line test, but
        # they remain attached to the left/right/top/bottom edge of the inset
        # crop. Remove only narrow edge-band components; a centred digit 1 is
        # retained because it is not adjacent to the crop's left/right edge.
        vertical_edge_fragment = (
            at_vertical_edge and width <= max(6, int(0.22 * w))
        )
        horizontal_edge_fragment = (
            at_horizontal_edge and height <= max(6, int(0.22 * h))
        )
        if horizontal_line or vertical_line or vertical_edge_fragment or horizontal_edge_fragment:
            removed += int(area)
            continue
        cleaned[labels == component_id] = 255
    return cleaned, removed


def _select_digit_glyph(binary: np.ndarray, cfg: ScanConfig) -> tuple[np.ndarray, bool, dict[str, float]]:
    """Keep a compact, centred numeral-shaped component, not arbitrary cell ink.

    This is the emptiness gate. A printed digit should be a compact glyph in the
    central portion of the cell. Broad shadows, paper gradients, filled masks,
    border fragments and scattered texture are not sent to Tesseract.
    """
    h, w = binary.shape[:2]
    empty = np.zeros_like(binary)
    empty_features = {
        "ink_fraction": 0.0, "component_count": 0.0,
        "glyph_width_fraction": 0.0, "glyph_height_fraction": 0.0,
        "glyph_density": 0.0, "glyph_center_offset": 1.0,
    }
    if h < 8 or w < 8:
        return empty, False, empty_features

    # Otsu on a nearly uniform but noisy paper patch can split random sensor
    # noise into compact connected components that resemble a tiny glyph. A
    # real printed numeral normally occupies a small minority of the cell.
    # Reject heavily textured foreground before inspecting individual blobs;
    # this prevents random noise from being promoted to a Tesseract digit.
    cell_foreground_fraction = float(np.count_nonzero(binary)) / float(binary.size)
    empty_features["cell_foreground_fraction"] = cell_foreground_fraction
    if cell_foreground_fraction > cfg.max_cell_foreground_fraction:
        return empty, False, empty_features

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    min_area = max(3, int(round(binary.size * max(
        cfg.empty_min_component_fraction, 0.0035
    ))))
    candidates: list[tuple[float, int, tuple[int, int, int, int, int]]] = []
    for component_id in range(1, count):
        x, y, cw, ch, area = map(int, stats[component_id])
        if area < min_area:
            continue
        width_fraction = cw / float(w)
        height_fraction = ch / float(h)
        area_fraction = area / float(binary.size)
        aspect = cw / float(max(1, ch))
        density = area / float(max(1, cw * ch))
        cx, cy = map(float, centroids[component_id])
        center_offset = float(np.hypot((cx - (w - 1) * 0.5) / w,
                                       (cy - (h - 1) * 0.5) / h))

        # Numerals are generally tall and compact, and sit away from the grid
        # border. The bounds are intentionally permissive for a narrow '1' and
        # different printed fonts, but reject half-cell shadows/large blobs.
        if width_fraction < 0.035 or width_fraction > 0.82:
            continue
        if height_fraction < 0.18 or height_fraction > 0.88:
            continue
        if not 0.055 <= aspect <= 2.15:
            continue
        if area_fraction > cfg.empty_max_ink_fraction:
            continue
        if density < 0.025 or density > 0.88:
            continue
        if not (0.13 <= cx / w <= 0.87 and 0.08 <= cy / h <= 0.92):
            continue
        if x <= 0 or y <= 0 or x + cw > w or y + ch > h:
            continue

        # A smooth shadow patch can become one neat connected component after
        # adaptive thresholding. Dense components are only accepted when they
        # contain an interior counter (e.g. 0, 6, 8 or 9); otherwise they look
        # like a filled rectangle, not a printed numeral.
        component_mask = (labels[y:y + ch, x:x + cw] == component_id).astype(np.uint8) * 255
        _contours, hierarchy = cv2.findContours(
            component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        hole_count = (sum(int(item[3]) >= 0 for item in hierarchy[0])
                      if hierarchy is not None else 0)
        # Smooth shadow islands often become square rings under local adaptive
        # thresholding. A real 6/9/0 counter is oval, while 8 has two counters.
        if hole_count == 1 and aspect > 0.94:
            continue
        # Dense, nearly square solid blocks are paper/lighting artefacts. Bold
        # printed 5s can also be dense, but remain taller than they are wide.
        if density > 0.58 and hole_count == 0 and aspect > 0.86:
            continue

        # Prefer large, vertically developed components over broad patches.
        score = float(area) * (0.55 + 0.45 * height_fraction) * (1.0 - 0.25 * center_offset)
        candidates.append((score, component_id, (x, y, cw, ch, area)))

    if not candidates:
        return empty, False, empty_features

    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, primary_id, primary_stats = candidates[0]
    px, py, pw, ph, _pa = primary_stats
    primary_cx, primary_cy = centroids[primary_id]

    # Retain nearby small components only if they plausibly belong to the same
    # printed glyph (e.g. a broken stroke). Do not merge unrelated paper noise.
    kept = np.zeros_like(binary)
    accepted_ids = [primary_id]
    for _candidate_score, component_id, (x, y, cw, ch, _area) in candidates[1:]:
        cx, cy = centroids[component_id]
        near_primary = np.hypot((cx - primary_cx) / max(1.0, w),
                                (cy - primary_cy) / max(1.0, h)) <= 0.22
        overlaps_glyph_band = (x <= px + pw + 0.10 * w and x + cw >= px - 0.10 * w
                               and y <= py + ph + 0.10 * h and y + ch >= py - 0.10 * h)
        if near_primary and overlaps_glyph_band:
            accepted_ids.append(component_id)

    for component_id in accepted_ids:
        kept[labels == component_id] = 255

    ys, xs = np.where(kept > 0)
    if len(xs) == 0:
        return empty, False, empty_features
    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    glyph_w, glyph_h = x_max - x_min, y_max - y_min
    total_ink = int(len(xs))
    bbox_density = total_ink / float(max(1, glyph_w * glyph_h))
    center_x = (x_min + x_max) * 0.5 / w
    center_y = (y_min + y_max) * 0.5 / h
    overall_features = {
        "ink_fraction": total_ink / float(binary.size),
        "component_count": float(len(accepted_ids)),
        "glyph_width_fraction": glyph_w / float(w),
        "glyph_height_fraction": glyph_h / float(h),
        "glyph_density": bbox_density,
        "glyph_center_offset": float(np.hypot(center_x - 0.5, center_y - 0.5)),
        "cell_foreground_fraction": cell_foreground_fraction,
    }

    # Recheck the union: accidental nearby patches must not turn a valid small
    # component into a large background region.
    valid = (
        total_ink >= cfg.empty_min_ink_pixels
        and overall_features["ink_fraction"] >= cfg.empty_ink_fraction
        and overall_features["ink_fraction"] <= cfg.empty_max_ink_fraction
        and overall_features["glyph_width_fraction"] <= 0.84
        and overall_features["glyph_height_fraction"] <= 0.90
        and 0.025 <= bbox_density <= 0.90
        and abs(center_x - 0.5) <= 0.34
        and abs(center_y - 0.5) <= 0.38
    )
    return (kept if valid else empty), bool(valid), overall_features


def _normalize_glyph(binary: np.ndarray) -> np.ndarray:
    """Centre and scale a pre-filtered digit mask on a clean OCR canvas."""
    if not np.any(binary):
        return np.full((128, 128), 255, np.uint8)
    ys, xs = np.where(binary > 0)
    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    glyph = binary[y_min:y_max, x_min:x_max]
    glyph_h, glyph_w = glyph.shape[:2]
    #scale = min(46.0 / max(1, glyph_w), 46.0 / max(1, glyph_h))
    scale = min(54.0 / max(1, glyph_w), 54.0 / max(1, glyph_h))
    new_w = max(1, int(round(glyph_w * scale)))
    new_h = max(1, int(round(glyph_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(glyph, (new_w, new_h), interpolation=interpolation)
    canvas = np.full((64, 64), 255, np.uint8)
    offset_x = (64 - new_w) // 2
    offset_y = (64 - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = 255 - resized
    canvas = cv2.copyMakeBorder(canvas, 16, 16, 16, 16,
                                cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(canvas, (128, 128), interpolation=cv2.INTER_CUBIC)


def prepare_cell(cell: np.ndarray, cfg: ScanConfig) -> tuple[np.ndarray, bool, float, np.ndarray]:
    """Classify empty/filled before OCR using compact numeral-shape evidence."""
    crop = _cell_crop(cell, cfg)
    binary = _binarize(crop)
    binary, _removed_border_ink = _remove_cell_border_artifacts(binary)
    glyph_mask, is_filled, features = _select_digit_glyph(binary, cfg)
    ink_fraction = float(features["ink_fraction"])
    prepared = _normalize_glyph(glyph_mask) if is_filled else np.full((128, 128), 255, np.uint8)
    return crop, is_filled, ink_fraction, prepared


def prepare_cell_alternative(crop: np.ndarray, cfg: ScanConfig) -> np.ndarray:
    """Prepare an alternate adaptive-threshold image only when it looks like a digit."""
    gray = _gray(crop)
    if min(gray.shape[:2]) < 3:
        return np.full((128, 128), 255, np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    side = min(enhanced.shape[:2])
    block_size = min(15, side if side % 2 == 1 else side - 1)
    block_size = max(3, block_size)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 4,
    )
    binary, _removed_border_ink = _remove_cell_border_artifacts(binary)
    glyph_mask, is_filled, _features = _select_digit_glyph(binary, cfg)
    return _normalize_glyph(glyph_mask) if is_filled else np.full((128, 128), 255, np.uint8)


def _tesseract_executable() -> str | None:
    """Find Tesseract from PATH, an explicit env var, or common Windows installs."""
    try:
        import pytesseract
        configured = str(pytesseract.pytesseract.tesseract_cmd)
    except Exception:
        configured = "tesseract"

    candidates = [os.environ.get("SUDOKUSCAN_TESSERACT", ""), configured, "tesseract"]
    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        candidates.extend([
            str(Path(program_files) / "Tesseract-OCR" / "tesseract.exe"),
            str(Path(program_files_x86) / "Tesseract-OCR" / "tesseract.exe"),
            str(Path(local_app_data) / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
            str(Path(local_app_data) / "Tesseract-OCR" / "tesseract.exe"),
        ])

    seen: set[str] = set()
    for command in candidates:
        if not command:
            continue
        normalized = os.path.normcase(command)
        if normalized in seen:
            continue
        seen.add(normalized)
        found = shutil.which(command)
        if found:
            return found
        candidate_path = Path(command).expanduser()
        if candidate_path.is_file():
            return str(candidate_path)
    return None


def _recognize_multipage(pages: list[np.ndarray], psm: int,
                         deadline: float | None = None) -> dict[int, tuple[int, float]]:
    """OCR a multipage TIFF; each page is exactly one candidate cell image."""
    if not pages:
        return {}
    executable = _tesseract_executable()
    if executable is None:
        return {}
    remaining = 15.0 if deadline is None else deadline - time.perf_counter()
    if remaining <= 0.05:
        return {}

    pil_pages = [Image.fromarray(np.asarray(page, dtype=np.uint8)) for page in pages]
    try:
        with tempfile.TemporaryDirectory(prefix="sudokuscan_ocr_") as temp_dir:
            tiff_path = Path(temp_dir) / "cells.tiff"
            pil_pages[0].save(
                tiff_path, format="TIFF", save_all=True,
                append_images=pil_pages[1:], compression="tiff_lzw",
            )
            command = [
                executable, str(tiff_path), "stdout", "--oem", "3", "--psm", str(psm),
                "-c", "tessedit_char_whitelist=123456789",
                "-c", "load_system_dawg=0", "-c", "load_freq_dawg=0", "tsv",
            ]
            remaining = 15.0 if deadline is None else deadline - time.perf_counter()
            if remaining <= 0.05:
                return {}
            # Tesseract builds commonly use OpenMP. On high-core-count PCs,
            # allowing it to create a worker per logical CPU for each batch can
            # oversubscribe the machine and make repeated dataset sweeps
            # erratic. Respect an explicit user setting; otherwise use one
            # worker for predictable CPU-only per-image latency.
            tesseract_env = os.environ.copy()
            tesseract_env.setdefault("OMP_THREAD_LIMIT", "1")
            completed = subprocess.run(
                command, capture_output=True, text=True, env=tesseract_env,
                timeout=max(0.05, min(15.0, remaining)), check=False,
            )
            if completed.returncode != 0 or not completed.stdout.strip():
                return {}
            parsed: dict[int, tuple[int, float]] = {}
            reader = csv.DictReader(completed.stdout.splitlines(), delimiter="\t")
            for row in reader:
                token = str(row.get("text", "")).strip()
                digits = [int(ch) for ch in token if ch in "123456789"]
                # Every TIFF page is a single cell. A multi-character token is
                # a segmentation artefact, not a valid one-cell prediction.
                if len(digits) != 1:
                    continue
                try:
                    page_num = int(row.get("page_num", "0"))
                    confidence = float(row.get("conf", "-1"))
                except (TypeError, ValueError):
                    continue
                if page_num < 1:
                    continue
                item = (digits[0], confidence)
                current = parsed.get(page_num)
                if current is None or confidence > current[1]:
                    parsed[page_num] = item
            return parsed
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return {}


def _fuse_cell_readings(readings: list[tuple[int, float, str]]) -> int:
    """Prefer independent OCR agreement; abstain on weak, isolated readings.

    Tesseract can hallucinate a digit from a shadow or a textured blank cell.
    One low-confidence view is not sufficient evidence. A single view is only
    accepted when its confidence is very high; otherwise two views must agree.
    """
    weights = {
        "otsu": 1.0, "adaptive": 0.95, "raw": 0.75,
        # A different Tesseract segmentation mode supplies a new segmentation
        # hypothesis, not a new image. It is therefore a confirmation vote
        # with slightly lower weight than the primary Otsu result.
        "fallback": 0.85,
    }
    valid = [
        (int(digit), float(confidence), source)
        for digit, confidence, source in readings
        if digit in range(1, 10) and confidence >= 45.0 and source in weights
    ]
    if not valid:
        return 0

    scores: dict[int, float] = {}
    confidences: dict[int, list[float]] = {}
    for digit, confidence, source in valid:
        quality = max(0.0, min(100.0, confidence)) / 100.0
        scores[digit] = scores.get(digit, 0.0) + weights[source] * (0.5 + 0.5 * quality)
        confidences.setdefault(digit, []).append(confidence)

    ranked = sorted(scores, key=lambda digit: (scores[digit], len(confidences[digit]), -digit), reverse=True)
    best = ranked[0]
    vote_count = len(confidences[best])
    if vote_count >= 2:
        # If two candidates both have repeated support, prefer the one with a
        # meaningful weighted-confidence lead. A near tie remains unknown and
        # is passed to the Sudoku candidate reconciler rather than guessed.
        if len(ranked) > 1 and len(confidences[ranked[1]]) >= 2:
            runner_up = ranked[1]
            if scores[best] < scores[runner_up] * 1.05:
                return 0
        return best

    # With conflicting one-vote readings, prefer the original-crop PSM 10
    # result only when its confidence is clearly stronger. The raw crop keeps
    # the actual glyph contour; normalized masks can occasionally turn a bold
    # 3/8/9 into another character. This branch is reached only after the
    # independent empty-cell morphology gate has already accepted a glyph.
    if len(valid) >= 2:
        raw_votes = [(digit, confidence) for digit, confidence, source in valid
                     if source == "raw"]
        other_confidences = [confidence for _digit, confidence, source in valid
                             if source != "raw"]
        if raw_votes and other_confidences:
            raw_digit, raw_confidence = max(raw_votes, key=lambda item: item[1])
            strongest_other = max(other_confidences)
            margin = raw_confidence - strongest_other
            if raw_confidence >= 85.0 and margin >= 10.0:
                return raw_digit

    # A lone OCR vote is accepted only at very high confidence, and only if no
    # competing view proposed a different digit with comparable confidence.
    best_confidence = confidences[best][0]
    #if best_confidence < 88.0:
    #    return 0
    
    if best_confidence < 88.0:
        return 0
    if len(ranked) > 1 and scores[ranked[1]] >= 0.72 * scores[best]:
        return 0
    return best


def recognize_grid(rectified: np.ndarray, cfg: ScanConfig, *, compare_raw: bool = False,
                   deadline: float | None = None
                   ) -> tuple[list[list[int]], dict[str, Any], list[list[CellRecognition]]]:
    """Recognize cells using Otsu, local-adaptive and optional raw OCR views.

    All variants are submitted in one multipage Tesseract invocation. Every
    page still represents a single cell and uses the configured single-character
    PSM and 1-9 whitelist. This avoids launching one OCR process per cell.
    """
    height, width = rectified.shape[:2]
    if height != width or height % 9 != 0:
        raise ValueError("OCR rectification must be square and divisible by 9")
    size = height
    grid = [[0] * 9 for _ in range(9)]
    cells: list[list[CellRecognition | None]] = [[None] * 9 for _ in range(9)]
    pages: list[np.ndarray] = []
    page_indices: dict[tuple[int, int], dict[str, int]] = {}
    fallback_pages: list[np.ndarray] = []
    fallback_page_indices: dict[tuple[int, int], int] = {}
    filled_count = empty_count = 0

    for row in range(9):
        for col in range(9):
            x0, x1 = int(round(col * size / 9)), int(round((col + 1) * size / 9))
            y0, y1 = int(round(row * size / 9)), int(round((row + 1) * size / 9))
            crop, is_filled, ink_fraction, prepared = prepare_cell(
                rectified[y0:y1, x0:x1], cfg
            )
            if not is_filled:
                empty_count += 1
                blank = np.full((128, 128), 255, np.uint8)
                cells[row][col] = CellRecognition(
                    digit=0, confidence=100.0, is_filled=False,
                    ink_fraction=ink_fraction, raw_digit=0, raw_confidence=100.0,
                    crop=crop, prepared=blank, alternate_digit=0,
                    alternate_confidence=100.0, alternate_prepared=blank,
                    selected_digit=0,
                )
                continue

            filled_count += 1
            alternate = prepare_cell_alternative(crop, cfg)
            page_map: dict[str, int] = {}
            for name, page in (("otsu", prepared), ("adaptive", alternate)):
                pages.append(page)
                page_map[name] = len(pages)  # Tesseract page numbers are 1-based.
            raw_gray = _gray(crop)
            # Preserve the original glyph proportions. A 2x upscale plus a
            # white quiet-zone lets Tesseract segment the one-character crop
            # without changing its stroke shape.
            raw_page = cv2.resize(
                raw_gray, (raw_gray.shape[1] * 2, raw_gray.shape[0] * 2),
                interpolation=cv2.INTER_CUBIC,
            )
            padding = max(8, int(round(min(raw_page.shape[:2]) * 0.06)))
            raw_page = cv2.copyMakeBorder(
                raw_page, padding, padding, padding, padding,
                cv2.BORDER_CONSTANT, value=255,
            )
            if compare_raw:
                pages.append(raw_page)
                page_map["raw"] = len(pages)
            # PSM 7 is a line segmentation hypothesis. Run it on the same
            # original-crop view for all candidate glyphs and use it only as a
            # secondary vote; the explicit empty-cell gate above remains the
            # sole path into OCR.
            if cfg.ocr_fallback_psm is not None:
                fallback_pages.append(raw_page)
                fallback_page_indices[(row, col)] = len(fallback_pages)
            page_indices[(row, col)] = page_map
            cells[row][col] = CellRecognition(
                digit=0, confidence=-1.0, is_filled=True,
                ink_fraction=ink_fraction, raw_digit=0, raw_confidence=-1.0,
                crop=crop, prepared=prepared, alternate_digit=0,
                alternate_confidence=-1.0, alternate_prepared=alternate,
                selected_digit=0,
            )

    predictions = _recognize_multipage(pages, cfg.ocr_psm, deadline)
    fallback_predictions: dict[int, tuple[int, float]] = {}
    fallback_attempted = False
    # Secondary segmentation mode only runs when there is meaningful time left
    # in the image budget. It is a confirmation pass on the same cells, not a
    # relaxation of the pre-OCR filled/empty decision.
    fallback_remaining = 15.0 if deadline is None else deadline - time.perf_counter()
    if fallback_pages and fallback_remaining > 0.20:
        fallback_attempted = True
        fallback_predictions = _recognize_multipage(
            fallback_pages, cfg.ocr_fallback_psm, deadline
        )
    prepared_hits = adaptive_hits = raw_hits = fallback_hits = fused_hits = 0
    primary_confidences: list[float] = []
    adaptive_confidences: list[float] = []
    raw_confidences: list[float] = []
    fallback_confidences: list[float] = []
    for (row, col), indexes in page_indices.items():
        primary_digit, primary_conf = predictions.get(indexes["otsu"], (0, -1.0))
        adaptive_digit, adaptive_conf = predictions.get(indexes["adaptive"], (0, -1.0))
        if "raw" in indexes:
            raw_digit, raw_conf = predictions.get(indexes["raw"], (0, -1.0))
        else:
            raw_digit, raw_conf = 0, -1.0
        fallback_page = fallback_page_indices.get((row, col), 0)
        fallback_digit, fallback_conf = fallback_predictions.get(fallback_page, (0, -1.0))
        readings = [(primary_digit, primary_conf, "otsu"),
                    (adaptive_digit, adaptive_conf, "adaptive")]
        if "raw" in indexes:
            readings.append((raw_digit, raw_conf, "raw"))
        if fallback_page:
            readings.append((fallback_digit, fallback_conf, "fallback"))
        selected = _fuse_cell_readings(readings)

        previous = cells[row][col]
        assert previous is not None
        cells[row][col] = CellRecognition(
            digit=primary_digit, confidence=primary_conf,
            is_filled=True, ink_fraction=previous.ink_fraction,
            raw_digit=raw_digit, raw_confidence=raw_conf,
            crop=previous.crop, prepared=previous.prepared,
            alternate_digit=adaptive_digit, alternate_confidence=adaptive_conf,
            alternate_prepared=previous.alternate_prepared,
            selected_digit=selected,
            fallback_digit=fallback_digit,
            fallback_confidence=fallback_conf,
        )
        grid[row][col] = selected
        prepared_hits += int(primary_digit > 0)
        adaptive_hits += int(adaptive_digit > 0)
        raw_hits += int(raw_digit > 0)
        fallback_hits += int(fallback_digit > 0)
        fused_hits += int(selected > 0)
        if primary_conf >= 0:
            primary_confidences.append(primary_conf)
        if adaptive_conf >= 0:
            adaptive_confidences.append(adaptive_conf)
        if raw_conf >= 0:
            raw_confidences.append(raw_conf)
        if fallback_conf >= 0:
            fallback_confidences.append(fallback_conf)

    completed_cells: list[list[CellRecognition]] = [
        [cell if cell is not None else CellRecognition(
            0, -1.0, False, 0.0, 0, -1.0, np.zeros((1, 1), np.uint8),
            np.full((128, 128), 255, np.uint8), selected_digit=0,
        ) for cell in row] for row in cells
    ]
    # If the highest-confidence per-cell choices conflict or cannot be
    # completed, search the available OCR alternatives under Sudoku rules.
    # This does not synthesize clues: it only chooses between actual OCR
    # readings, while the solver fills cells that OCR classified as empty.
    from .solver import reconcile_ocr_candidates
    grid, constraint_diagnostics = reconcile_ocr_candidates(grid, completed_cells)
    for row in range(9):
        for col in range(9):
            completed_cells[row][col].selected_digit = int(grid[row][col])

    stats = {
        "filled_cells": filled_count,
        "empty_cells": empty_count,
        "prepared_recognition_rate_on_filled": float(prepared_hits / max(1, filled_count)),
        "adaptive_recognition_rate_on_filled": float(adaptive_hits / max(1, filled_count)),
        "raw_recognition_rate_on_filled": float(raw_hits / max(1, filled_count)) if compare_raw else None,
        "fallback_recognition_rate_on_filled": float(fallback_hits / max(1, filled_count)),
        "fused_recognition_rate_on_filled": float(fused_hits / max(1, filled_count)),
        "mean_prepared_ocr_confidence": float(np.mean(primary_confidences)) if primary_confidences else -1.0,
        "mean_adaptive_ocr_confidence": float(np.mean(adaptive_confidences)) if adaptive_confidences else -1.0,
        "mean_raw_ocr_confidence": float(np.mean(raw_confidences)) if raw_confidences else None,
        "mean_fallback_ocr_confidence": float(np.mean(fallback_confidences)) if fallback_confidences else None,
        "ocr_engine": "Tesseract (primary Otsu/adaptive/raw views with secondary PSM confirmation)",
        "raw_view_scale": 2,
        "raw_view_padding_px_min": 8,
        "ocr_psm": cfg.ocr_psm,
        "ocr_psm_description": "single character" if cfg.ocr_psm == 10 else "user-selected Tesseract segmentation mode",
        "ocr_fallback_psm": cfg.ocr_fallback_psm,
        "ocr_fallback_attempted": fallback_attempted,
        "ocr_fallback_cells_submitted": len(fallback_pages),
        "tesseract_available": _tesseract_executable() is not None,
        "ocr_process_calls_requested": int(bool(pages)) + int(fallback_attempted),
        "ocr_pages_submitted": len(pages),
        "ocr_fallback_pages_submitted": len(fallback_pages) if fallback_attempted else 0,
        "fusion": "confidence-weighted agreement across preprocessing views and secondary PSM confirmation",
        "sudoku_constraint_resolution": constraint_diagnostics,
    }
    return grid, stats, completed_cells


def best_orientation(rectified: np.ndarray, cfg: ScanConfig, deadline: float | None = None
                    ) -> tuple[np.ndarray, list[list[int]], dict, list[list[CellRecognition]], int]:
    """Bonus B4: choose among 0/90/180/270-degree views by OCR confidence."""
    candidates = [
        (rectified, 0),
        (cv2.rotate(rectified, cv2.ROTATE_90_CLOCKWISE), 90),
        (cv2.rotate(rectified, cv2.ROTATE_180), 180),
        (cv2.rotate(rectified, cv2.ROTATE_90_COUNTERCLOCKWISE), 270),
    ]
    best = None
    for image, angle in candidates:
        grid, stats, cells = recognize_grid(image, cfg, deadline=deadline)
        confidence = stats["mean_prepared_ocr_confidence"]
        recognition_rate = stats["fused_recognition_rate_on_filled"]
        score = (0.7 * max(0.0, confidence) + 30.0 * recognition_rate)
        item = (score, image, grid, stats, cells, angle)
        if best is None or score > best[0]:
            best = item
    assert best is not None
    return best[1], best[2], best[3], best[4], best[5]
