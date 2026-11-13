from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanConfig:
    """Central configuration. Spatial parameters are relative to image size."""

    target_size: int = 450
    cells_per_side: int = 9
    max_detection_side: int = 1600
    # OCR gets a higher-resolution perspective warp while the required saved
    # rectifications remain exactly 450x450.
    ocr_scale: int = 2

    # Harris detector: these are initial values to study, not claimed optima.
    harris_block_size: int = 2
    harris_ksize: int = 3
    harris_k: float = 0.04
    response_threshold: float = 0.001
    nms_radius_fraction: float = 0.0045
    min_candidate_distance_fraction: float = 0.0045
    max_candidates: int = 2500

    # Region proposals only locate the likely puzzle. Final corners are selected
    # from Harris candidates and optionally refined by cornerSubPix.
    proposal_min_area_fraction: float = 0.0025
    proposal_max_area_fraction: float = 0.92
    corner_match_radius_fraction: float = 0.05
    max_proposals: int = 100
    max_hough_proposals: int = 120

    cell_margin_fraction: float = 0.15
    empty_ink_fraction: float = 0.004
    empty_min_ink_pixels: int = 12
    empty_min_component_fraction: float = 0.0015
    empty_max_ink_fraction: float = 0.34
    # If a cell has foreground across too much of its area after Otsu, it is
    # likely textured/noisy paper or a shadow rather than one printed numeral.
    max_cell_foreground_fraction: float = 0.24

    # PSM 10 is Tesseract's single-character mode, matching one digit per cell.
    ocr_psm: int = 10
    # Secondary segmentation mode used as a confirmation pass for each already
    # detected glyph. PSM 10 remains the primary single-character mode.
    ocr_fallback_psm: int | None = 7
    debug: bool = True

    def __post_init__(self) -> None:
        if self.target_size != 450:
            raise ValueError("The assignment requires a 450x450 rectification target")
        if self.cells_per_side != 9:
            raise ValueError("The assignment requires a 9x9 Sudoku grid")
        if self.target_size % self.cells_per_side:
            raise ValueError("target_size must be divisible by cells_per_side")
        if self.max_detection_side < 64:
            raise ValueError("max_detection_side must be at least 64 pixels")
        if self.ocr_scale not in {1, 2, 3}:
            raise ValueError("ocr_scale must be 1, 2 or 3")
        if self.harris_block_size < 2:
            raise ValueError("Harris blockSize must be >= 2")
        if self.harris_ksize < 1 or self.harris_ksize % 2 == 0:
            raise ValueError("Harris ksize must be a positive odd integer")
        if not 0.0 < self.harris_k < 0.25:
            raise ValueError("Harris k must be between 0 and 0.25")
        if not 0.0 < self.response_threshold <= 1.0:
            raise ValueError("response_threshold must be in (0, 1]")
        if self.nms_radius_fraction <= 0 or self.min_candidate_distance_fraction <= 0:
            raise ValueError("NMS and candidate-distance fractions must be positive")
        if self.max_candidates < 4 or self.max_proposals < 1 or self.max_hough_proposals < 1:
            raise ValueError("At least four Harris candidates and one proposal are required")
        if not 0.0 < self.proposal_min_area_fraction < self.proposal_max_area_fraction <= 1.0:
            raise ValueError("Invalid proposal area fractions")
        if not 0.0 < self.corner_match_radius_fraction <= 0.25:
            raise ValueError("corner_match_radius_fraction must be in (0, 0.25]")
        if not 0.0 <= self.cell_margin_fraction < 1.0 / 3.0:
            raise ValueError("cell_margin_fraction must be in [0, 1/3)")
        if not 0.0 <= self.empty_ink_fraction <= 1.0:
            raise ValueError("empty_ink_fraction must be in [0, 1]")
        if self.empty_min_ink_pixels < 0 or not 0.0 <= self.empty_min_component_fraction <= 1.0:
            raise ValueError("Invalid empty-cell component thresholds")
        if not 0.05 <= self.empty_max_ink_fraction <= 0.8:
            raise ValueError("empty_max_ink_fraction must be between 0.05 and 0.8")
        if not 0.05 <= self.max_cell_foreground_fraction <= 0.8:
            raise ValueError("max_cell_foreground_fraction must be between 0.05 and 0.8")
        if self.ocr_psm not in {6, 7, 8, 10, 13}:
            raise ValueError("ocr_psm must be one of 6, 7, 8, 10 or 13")
        if self.ocr_fallback_psm is not None and self.ocr_fallback_psm not in {6, 7, 8, 10, 13}:
            raise ValueError("ocr_fallback_psm must be one of 6, 7, 8, 10 or 13, or None")
        if self.ocr_fallback_psm == self.ocr_psm:
            raise ValueError("ocr_fallback_psm must differ from the primary ocr_psm")

    @property
    def cell_size(self) -> int:
        return self.target_size // self.cells_per_side
