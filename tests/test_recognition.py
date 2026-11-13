import cv2
import numpy as np

from sudokuscan.config import ScanConfig
from sudokuscan.recognition import prepare_cell, prepare_cell_alternative


def test_empty_cell_is_classified_before_ocr():
    blank = np.full((50, 50), 255, dtype=np.uint8)
    _crop, is_filled, ink_fraction, prepared = prepare_cell(blank, ScanConfig())
    assert is_filled is False
    assert ink_fraction == 0.0
    assert prepared.shape == (128, 128)
    assert np.all(prepared == 255)


def test_digit_cell_is_prepared_as_a_centered_128px_glyph():
    cell = np.full((50, 50), 255, dtype=np.uint8)
    cv2.putText(cell, "5", (13, 39), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)
    _crop, is_filled, ink_fraction, prepared = prepare_cell(cell, ScanConfig())
    assert is_filled is True
    assert ink_fraction > 0.008
    assert prepared.shape == (128, 128)
    assert np.count_nonzero(prepared < 128) > 10


def test_adaptive_preparation_produces_clean_single_cell_image():
    cell = np.full((50, 50), 235, dtype=np.uint8)
    cv2.putText(cell, "8", (13, 39), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 15, 2, cv2.LINE_AA)
    _crop, is_filled, _ink_fraction, _prepared = prepare_cell(cell, ScanConfig())
    assert is_filled is True
    alternate = prepare_cell_alternative(cell, ScanConfig())
    assert alternate.shape == (128, 128)
    assert np.count_nonzero(alternate < 128) > 10


def test_long_cell_gridline_is_removed_before_empty_decision():
    # A vertical grid-border fragment spanning the cell interior is not a digit.
    cell = np.full((100, 100), 255, dtype=np.uint8)
    for y in range(10, 91, 6):
        cv2.line(cell, (15, y), (15, min(90, y + 3)), 0, 2, cv2.LINE_AA)
    _crop, is_filled, ink_fraction, prepared = prepare_cell(cell, ScanConfig())
    assert is_filled is False
    assert ink_fraction == 0.0
    assert np.all(prepared == 255)


def test_recognizer_accepts_high_resolution_internal_rectification(monkeypatch):
    from sudokuscan.recognition import recognize_grid

    # All 81 cells are blank, so no Tesseract executable/process is needed.
    image = np.full((900, 900), 255, dtype=np.uint8)
    grid, stats, cells = recognize_grid(image, ScanConfig())
    assert len(grid) == 9 and all(len(row) == 9 for row in grid)
    assert all(value == 0 for row in grid for value in row)
    assert stats["empty_cells"] == 81
    assert len(cells) == 9 and all(len(row) == 9 for row in cells)


def test_centred_digit_one_is_not_removed_as_a_border_line():
    cell = np.full((100, 100), 255, dtype=np.uint8)
    cv2.putText(cell, "1", (43, 78), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, 0, 3, cv2.LINE_AA)
    _crop, is_filled, _ink_fraction, prepared = prepare_cell(cell, ScanConfig())
    assert is_filled is True
    assert np.count_nonzero(prepared < 128) > 20


def test_all_printed_digits_one_through_nine_pass_the_empty_cell_gate():
    for digit in range(1, 10):
        cell = np.full((100, 100), 245, dtype=np.uint8)
        cv2.putText(cell, str(digit), (35, 77), cv2.FONT_HERSHEY_SIMPLEX,
                    1.7, 70, 3, cv2.LINE_AA)
        _crop, is_filled, _ink_fraction, prepared = prepare_cell(cell, ScanConfig())
        assert is_filled is True, f"printed digit {digit} was rejected as empty"
        assert np.count_nonzero(prepared < 128) > 20


def test_noisy_blank_cell_is_not_promoted_to_a_digit():
    # Otsu can classify almost half of a flat noisy patch as foreground. The
    # whole-cell foreground guard must reject it before a local blob is treated
    # as a possible numeral.
    rng = np.random.default_rng(2)
    blank_with_sensor_noise = np.clip(
        230 + rng.normal(0, 10, (100, 100)), 0, 255
    ).astype(np.uint8)
    _crop, is_filled, ink_fraction, prepared = prepare_cell(
        blank_with_sensor_noise, ScanConfig()
    )
    assert is_filled is False
    assert ink_fraction == 0.0
    assert np.all(prepared == 255)


def test_shadow_gradient_and_shadow_patch_are_empty_before_ocr():
    cfg = ScanConfig()
    gradient = np.tile(np.linspace(180, 255, 100, dtype=np.uint8), (100, 1))
    shadow_patch = np.full((100, 100), 245, dtype=np.uint8)
    cv2.ellipse(shadow_patch, (50, 50), (22, 20), 0, 0, 360, 160, -1)

    for blank in (gradient, shadow_patch):
        crop, is_filled, ink_fraction, prepared = prepare_cell(blank, cfg)
        assert is_filled is False
        assert ink_fraction == 0.0
        assert np.all(prepared == 255)
        adaptive = prepare_cell_alternative(crop, cfg)
        assert np.all(adaptive == 255)


def test_noisy_blank_does_not_pass_adaptive_ocr_preparation():
    rng = np.random.default_rng(2)
    blank_with_sensor_noise = np.clip(
        230 + rng.normal(0, 10, (100, 100)), 0, 255
    ).astype(np.uint8)
    crop, is_filled, _ink_fraction, _prepared = prepare_cell(
        blank_with_sensor_noise, ScanConfig()
    )
    assert is_filled is False
    assert np.all(prepare_cell_alternative(crop, ScanConfig()) == 255)


def test_raw_single_character_view_wins_only_with_clear_confidence_margin():
    from sudokuscan.recognition import _fuse_cell_readings

    # The raw 2x padded crop preserves the original glyph contour and can be
    # more reliable than a thresholded mask when those views disagree.
    assert _fuse_cell_readings([
        (5, 79.9, "otsu"), (3, 94.8, "raw")
    ]) == 3
    assert _fuse_cell_readings([
        (3, 60.2, "adaptive"), (8, 85.6, "raw")
    ]) == 8
    # A small confidence difference is not enough evidence to override a
    # disagreement; abstaining avoids turning uncertain cells into givens.
    assert _fuse_cell_readings([
        (5, 88.0, "otsu"), (3, 91.0, "raw")
    ]) == 0


def test_secondary_psm_confirmation_vote_is_used_for_filled_cells(monkeypatch):
    from sudokuscan import recognition

    image = np.full((900, 900), 255, dtype=np.uint8)
    cv2.putText(image, "5", (30, 72), cv2.FONT_HERSHEY_SIMPLEX,
                1.35, 0, 3, cv2.LINE_AA)
    calls = []

    def fake_multipage(pages, psm, deadline=None):
        calls.append((len(pages), psm))
        if psm == 10:
            # Otsu and adaptive disagree; the raw pass has no result.
            return {1: (5, 78.0), 2: (3, 61.0), 3: (0, -1.0)}
        if psm == 7:
            return {1: (5, 96.0)}
        raise AssertionError(f"unexpected PSM {psm}")

    monkeypatch.setattr(recognition, "_recognize_multipage", fake_multipage)
    grid, stats, cells = recognition.recognize_grid(
        image, ScanConfig(), compare_raw=True
    )
    assert calls == [(3, 10), (1, 7)]
    assert grid[0][0] == 5
    assert cells[0][0].fallback_digit == 5
    assert cells[0][0].fallback_confidence == 96.0
    assert stats["ocr_fallback_attempted"] is True
    assert stats["fallback_recognition_rate_on_filled"] == 1.0


def test_secondary_ocr_confirmation_can_be_disabled(monkeypatch):
    from sudokuscan import recognition

    image = np.full((900, 900), 255, dtype=np.uint8)
    cv2.putText(image, "8", (30, 72), cv2.FONT_HERSHEY_SIMPLEX,
                1.35, 0, 3, cv2.LINE_AA)
    calls = []

    def fake_multipage(pages, psm, deadline=None):
        calls.append(psm)
        return {1: (8, 95.0), 2: (8, 94.0), 3: (8, 93.0)}

    monkeypatch.setattr(recognition, "_recognize_multipage", fake_multipage)
    grid, stats, cells = recognition.recognize_grid(
        image, ScanConfig(ocr_fallback_psm=None), compare_raw=True
    )
    assert calls == [10]
    assert grid[0][0] == 8
    assert cells[0][0].fallback_digit == 0
    assert stats["ocr_fallback_attempted"] is False
    assert stats["fallback_recognition_rate_on_filled"] == 0.0


def test_tesseract_defaults_to_one_openmp_thread(monkeypatch):
    from subprocess import CompletedProcess
    import sudokuscan.recognition as recognition

    captured = {}
    monkeypatch.delenv("OMP_THREAD_LIMIT", raising=False)
    monkeypatch.setattr(recognition, "_tesseract_executable", lambda: "tesseract")

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        output = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t95\t7\n"
        )
        return CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(recognition.subprocess, "run", fake_run)
    result = recognition._recognize_multipage([np.full((32, 32), 255, np.uint8)], 10)
    assert result[1] == (7, 95.0)
    assert captured["env"]["OMP_THREAD_LIMIT"] == "1"


def test_tesseract_respects_explicit_openmp_thread_limit(monkeypatch):
    from subprocess import CompletedProcess
    import sudokuscan.recognition as recognition

    captured = {}
    monkeypatch.setenv("OMP_THREAD_LIMIT", "2")
    monkeypatch.setattr(recognition, "_tesseract_executable", lambda: "tesseract")

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        output = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t95\t7\n"
        )
        return CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(recognition.subprocess, "run", fake_run)
    recognition._recognize_multipage([np.full((32, 32), 255, np.uint8)], 10)
    assert captured["env"]["OMP_THREAD_LIMIT"] == "2"
