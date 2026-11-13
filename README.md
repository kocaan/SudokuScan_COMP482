# SudokuScan — COMP 482 Project 1

A classical computer-vision pipeline that turns a photograph of a printed Sudoku into a 9x9 grid. The implementation follows the supplied MEF University COMP 482 Project 1 brief and keeps the required stages separate.

## 1. Implemented features

| Requirement | Implementation |
|---|---|
| Preprocessing | Grayscale, Gaussian denoising, illumination normalization, CLAHE and adaptive thresholding. Spatial kernel sizes scale with the image. |
| Harris detection | `cv2.cornerHarris`, relative response threshold, dilation-based non-maximum suppression and spatial thinning. Response map and all candidates are saved. |
| Automatic corner selection | Contours / minimum-area rectangles are used only as region proposals. Each final corner is selected from a Harris candidate and refined with `cv2.cornerSubPix`. The heuristic scores candidate quads using Harris support and 10x10 grid evidence. |
| A3 | Three-point affine transform from TL, TR and BL; reports where the unused BR lands and its distance from (450, 450). |
| A4 | Four-point affine least-squares fit; reports four-corner RMS residual. |
| H4 | Four-point perspective homography; reports mapping residual. |
| Stage 4 measurements | Opposite-side obliqueness indicator, grid-line deviations and the same OCR stage run on A3/A4/H4. |
| Cell recognition | 81 equal crops; 15% inner margin by default; explicit empty-versus-filled decision before OCR; Otsu and CLAHE/adaptive-threshold preparation plus a raw crop view; confidence-weighted OCR fusion with primary Tesseract PSM 10 and a secondary PSM 7 confirmation pass on the raw crop; a 1-9 whitelist is applied in both passes. The raw-crop view is enlarged 2x with a white quiet-zone to preserve the original glyph contour. |
| Validity | Checks duplicate givens in every row, column and 3x3 box; debug image marks all conflicting cells. |
| Required output | `*_grid.txt`, `*_result.json`, `*_harris.png`, all three rectifications and `*_debug.png`. |
| Batch mode | Recursively scans the chosen folder and its subfolders, writes per-image outputs without name collisions between subfolders, records a `batch_summary.json`, and continues after a bad file. |
| Solver (B1) | Enabled by default: bit-mask/MRV backtracking fills the grid, writes a completed solution file, and counts solutions up to 2 to detect uniqueness. |
| Bonus B2 | Projects recognized digits or the solved grid back onto the original photograph. |
| Bonus B3 | Maps Harris candidates near the 10x10 grid nodes and estimates a RANSAC homography. |
| Bonus B4 | Tests 0/90/180/270 degree rectified orientations using OCR confidence. |
| Evaluation | Per-image and per-condition metrics from Appendix A ground-truth annotations. |
| Parameter study | Sweep script for blockSize, ksize, k and response threshold. |

## 2. Requirements and installation

- Python 3.10 or newer
- OpenCV 4.8–5.x, NumPy, pytesseract and Matplotlib
- Tesseract OCR executable (installed separately from the Python package)
- CPU only; no GPU or deep-learning model is used

### Windows (one-click)

1. Double-click `INSTALL.bat`. It creates `.venv`, installs the Python dependencies and pytest, attempts to install Tesseract OCR with `winget` if it is missing, and runs the test suite.
2. Double-click `RUN.bat`. Choose one image or a folder in the file picker. You can also drag a photo or folder onto `RUN.bat`.
3. Results are written under `results\single` or `results\batch`. Option 2 includes images in nested subfolders and writes a batch summary.

The installer needs a working Python 3.10+ installation and an internet connection for package installation. If `winget` is unavailable, install the Tesseract OCR Windows application manually; the Python code also checks common Windows install locations, not only `PATH`.

For manual setup:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the Tesseract OCR application for Windows and ensure `tesseract.exe` is on `PATH`. The code also checks the common UB-Mannheim install paths and the `SUDOKUSCAN_TESSERACT` environment variable. If it is elsewhere, pass its full path using `--tesseract-cmd`, for example:

```powershell
python sudoku_scan.py --input photo.jpg --outdir out --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the OCR executable:

```bash
tesseract --version
python -c "import cv2, numpy, pytesseract; print(cv2.__version__)"
```

## 3. Run the scanner

One image:

```bash
python sudoku_scan.py --input photo.jpg --outdir out/
```

Batch/grader mode (recursive, including nested folders):

```bash
python sudoku_scan.py --batch images_dir/ --outdir results/
```

Batch results preserve the source subfolder structure under the output folder. A `batch_summary.json` lists the result and solver status for every image. A single unreadable image does not stop the remaining batch; a non-zero exit code signals that at least one file failed to process.

The Windows menu's Option 2 calls `launcher.py`, which passes the selected directory as an argument to the scanner (rather than trying to parse file-picker output through nested batch/PowerShell quoting). It recursively scans images below the selected folder, always enables solving, and reports the number of images that received a solution. The batch command also handles an output directory above the selected input directory and rejects the unsafe case where input and output are the same directory.

The solver is enabled by default in both single-image and batch mode. It preserves the recognized givens in `*_grid.txt` and writes a separate completed board to `*_solution.txt` when a solution exists. It refuses to create a random completion when OCR found no givens (for example, when Tesseract is missing). `*_result.json` reports `solved_unique`, `solved_multiple_solutions`, `invalid_givens`, `unsatisfiable` or `no_recognized_givens`; a multiple-solution puzzle still gets a valid first completion, but is not mislabeled as unique. Use `--no-solve` to skip the solver.

Optional bonus flags:

```bash
python sudoku_scan.py --input photo.jpg --outdir out/ --solve --ar-overlay --many-point --orientation-search
```

Parameter overrides:

```bash
python sudoku_scan.py --input photo.jpg --outdir out/ \
  --harris-block-size 2 --harris-ksize 3 --harris-k 0.04 \
  --response-threshold 0.001 --max-detection-side 1600 --ocr-psm 10 \
  --cell-margin-fraction 0.15
```

The program resizes only for corner detection when the long side exceeds 1600 px. Output corner coordinates are converted back to original-image coordinates. The four output corners still come from Harris candidates; contour/minimum-area geometry supplies region proposals only, and `cornerSubPix` refines the selected Harris points. When a Harris point is substantially displaced from the proposed outer-grid junction, the subpixel-refinement window grows relative to the adjacent side length (with a cap to avoid jumping to an internal grid intersection). OCR subprocesses share an 18-second per-image deadline to leave time for the remaining stages under the assignment's 20-second target; if the time budget is exhausted, remaining OCR cells are left blank rather than allowing unbounded runtime.

### Classical OCR accuracy improvements and Sudoku-guided reading selection

Each cell is checked for emptiness before OCR. Thin residual grid-border components are removed only when they touch the edge of the inset crop, preserving a centered narrow digit such as `1`. A whole-cell foreground-coverage limit (default `0.24`) rejects highly textured/noisy blank cells before connected-component analysis, while compact, centered glyph checks reject smooth shadows and broad patches. Filled cells are represented three ways: Otsu thresholding, CLAHE plus adaptive thresholding, and the original grayscale crop enlarged proportionally by 2x with a white quiet-zone. The three pages are submitted together to Tesseract using the configured PSM (default 10, single character) and the `123456789` whitelist. A confidence-weighted agreement rule prefers independent agreement; when the original crop clearly outranks a conflicting thresholded view by at least 10 confidence points and scores at least 85, it can be selected. Weak or near-tied disagreements remain blank instead of being guessed. If the preferred grid is contradictory or cannot be completed, the program searches actual alternative OCR readings under row, column and 3x3-box constraints, with a fixed search-node cap. It does not use a neural network, a pre-trained Sudoku recognizer, an online solver or any dependency beyond `requirements.txt` and the separately installed Tesseract executable.

For small or oblique print, the assignment output images remain 450x450, but OCR is performed on an additional 2x internal warp generated directly from the original full-resolution photograph using the same A3/A4/H4 transform. This avoids enlarging an already-downsampled 450px image and preserves more digit-stroke detail. The OCR code derives the cell size from the input dimensions, so the internal 900x900 warp is split into 100px cells without changing the required output format.

Synthetic angle/lighting regression measurements are documented in Section 3.1 below. They are software tests using generated transformations of one puzzle, not evidence of performance on your required personal phone-photo dataset. Re-measure with your own original photographs before reporting project results.

## 3.1 Additional alignment and false-digit fixes (11 November 2026)

A follow-up investigation found a mismatch between the command-line default and `ScanConfig`: the CLI previously overrode the configured Harris response threshold with `0.01`, even though the config default was `0.001`. The default now comes directly from `ScanConfig`, so the weak outer-corner candidates are not discarded by an accidental tenfold increase.

Further changes:

- `sudokuscan/corners.py`: the normal detector now merges spatially deduplicated Harris maxima from the configured scale and two supplemental `cornerHarris` parameter sets. Contours and Hough lines still only propose the puzzle region; the selected corners remain Harris-derived and are subpixel-refined.
- `sudokuscan/config.py` and `sudokuscan/recognition.py`: added a whole-cell foreground-coverage guard. A cell with excessive thresholded foreground is treated as noisy/ambiguous, not as a numeral. This is specifically intended to stop flat sensor noise from becoming a compact connected component and a false OCR digit.
- `sudokuscan/recognition.py`: the original grayscale cell view is enlarged 2x with a white border for PSM 10, instead of being stretched to a fixed 128x128 image with no quiet-zone. In disagreement cases, the raw view can win only when its confidence is at least 85 and exceeds the strongest processed view by at least 10 points. The empty-cell morphology gate still runs before any OCR view is submitted.
- `tests/test_detection.py`, `tests/test_recognition.py` and `tests/test_pipeline.py`: added regressions for strong keystone alignment, grid-line straightness, the CLI Harris default, all digits 1-9, noisy blank cells, gradient/shadow blanks and confidence-based raw-view selection.

**New controlled-angle regression run (synthetic only):** four generated views of the same puzzle board (three perspective strengths and one shadowed perspective) produced 100% H4 cell accuracy, zero false-positive digits in empty cells, normalized corner error from 0.249% to 0.456%, grid-line RMS from 0.707 to 0.742 px and runtime from 2.31 to 2.49 seconds per image. These are repeatable software tests, not a claim of 100% accuracy on arbitrary phone photographs or the assignment's hidden set.

## 4. Output files

For an input called `photo.jpg`, the output directory contains:

- `photo_grid.txt`: 9 rows of 9 digits, `0` for empty cells (H4 recognized givens; unchanged assignment output contract).
- `photo_solution.txt`: completed 9x9 board written by the default solver when at least one solution exists.
- `photo_debug_solved.png`: rectified Sudoku with the completed board overlaid when solving succeeds.
- `photo_result.json`: corners in TL, TR, BR, BL order, A3/A4/H4 matrices, residuals, OCR diagnostics and the solver's solution, solution count (capped at two), uniqueness status and runtime.
- `photo_harris.png`: normalized Harris response map, candidate maxima in red and selected corners in green.
- `photo_rect_A3.png`, `photo_rect_A4.png`, `photo_rect_H4.png`: 450x450 rectifications.
- `photo_debug.png`: H4 image with cell boundaries, recognized digits and invalid cells marked.
- `photo_cells/`: raw inner-margin crops, Otsu-prepared cells, adaptive-prepared cells and per-cell metadata for primary/fallback OCR readings and confidence values.
- Optional `photo_ar.png` and `photo_rect_Hmany.png` when their bonus flags are enabled.

If no puzzle is found, a result JSON with `verdict: "not_found"` and a zero grid is written. In batch mode, an exception for one image is recorded in its result file and does not terminate the rest of the batch.

## 5 Interactive personal-dataset annotation tool

After running `INSTALL.bat` once, use `ANNOTATE_DATASET.bat` (Windows) or run:

```bash
python annotate_dataset.py --images data/personal/images --annotations data/personal/annotations
```

The tool searches the image folder recursively and saves one Appendix A JSON file per image, preserving nested subfolders. Click the four *outer Sudoku-grid intersections* in this exact order: **TL → TR → BR → BL**. The right panel shows which corner is expected next. A live magnifier displays the cursor-centred source pixels, the pixel coordinates are shown while hovering, the mouse wheel zooms, and right-drag pans. `Backspace` undoes the last point; `R` resets all corners; `Ctrl+S` saves.

See [`ANNOTATOR_GUIDE.md`](ANNOTATOR_GUIDE.md) for the full controls and workflow. The same window includes a 9×9 givens editor and C1–C5 condition selectors. Enter the digits printed on the paper and put `0` in every empty cell. Saving requires all four valid corners, at least one capture condition, and the explicit “I checked all 81 cells” confirmation. Existing JSON annotations are loaded for editing. The application reads images with OpenCV's normal image-loading behavior so the coordinates match the image coordinate system used by the scanner. Original photographs are never modified.

**Accuracy note:** the magnifier, zoom and ordering checks reduce annotation mistakes, but no tool can promise perfect corners automatically for every photograph. Click the outer grid border intersections (not the paper corners or the inner edge of a thick border), and visually inspect the quadrilateral before saving. The tool validates bounds, clockwise ordering, convexity and minimum geometry; it cannot know with certainty which pixel you intended.

## 6. Parameter study

After collecting and annotating at least six personal images, run:

```bash
python tools/parameter_sweep.py --images data/personal --annotations data/personal --out report/parameter_sweep.csv --limit 6
python tools/plot_parameter_sweep.py --csv report/parameter_sweep.csv --out report/figures/harris_parameter_study.png
```

The sweep varies blockSize (2/3/4), ksize (3/5), k (0.04/0.06) and response threshold (0.005/0.01/0.02). Discuss candidate counts and corner error, then choose and justify your final settings. The supplied default values are starting values, not claimed optimal settings. To compare Tesseract segmentation modes, use `diagnose_dataset.py --psm-sweep`; it scans the same annotated set with primary PSM 6/7/8/10/13 and disables the fallback pass during each sweep so the comparison isolates the primary mode. The default is PSM 10 because each non-empty cell is one character. Normal operation then uses a PSM 7 secondary confirmation pass on the same already-classified filled cells. The default inner crop margin is 15% per edge; it reduces grid-border remnants on the included synthetic check, but you must verify both choices on your own photos. PSM comparisons should be measured against annotations rather than OCR confidence.

## 7. Evaluation

Run the scanner over the personal image folder, then evaluate its results:

```bash
python sudoku_scan.py --batch data/personal --outdir results/personal
python evaluate.py --annotations data/personal --results results/personal --out report/metrics
python tools/plot_metrics.py --metrics report/metrics/per_image_metrics.csv --outdir report/figures
python tools/make_comparison_figure.py --results results/personal --out report/figures/most_oblique_rectifications.png
```

The evaluator reports corner error as a percentage of mean ground-truth side length, A3/A4/H4 cell accuracy, H4 Otsu-prepared, adaptive-prepared, raw-crop, fallback-only and fused accuracy, affine residuals, per-line position and adjacent-line spacing deviations, empty-cell F1, exact-grid match, runtime, frequent digit confusions, sample crop pairs and breakdown by C1-C5. Annotated images with missing result JSON files are retained as failed attempts in the robustness denominator instead of being silently skipped. The plotting scripts create the Stage 4 scatter plots for corner residual, grid straightness, grid spacing and OCR accuracy, plus the required side-by-side A3/A4/H4 figure for the most oblique input. It relies on correctly named annotation files and their original ground-truth `grid` and `corners` fields.

For the report's OCR comparison, the result JSON includes per-model OCR grids (`grid_A3`, `grid_A4`, `grid`) and the unprepared H4 grid (`grid_H4_raw`). The evaluator calculates actual cell accuracy against your annotations. A recognized-character rate is not the same as accuracy.


## 8. Personal-dataset diagnosis and report measurements

The one-command diagnosis script runs the scanner on your original photographs, evaluates every cell against the manually annotated grids, and creates report-ready tables, plots and error examples. Use a separate output folder outside the source image/annotation folders.

If images and annotations are in separate folders (their relative subfolder structure should match):

```bash
python diagnose_dataset.py --images data/personal/images --annotations data/personal/annotations --out report/personal_diagnosis
```

If images and their same-basename JSON annotations are together:

```bash
python diagnose_dataset.py --images data/personal --out report/personal_diagnosis
```

On Windows, double-click `DIAGNOSE_DATASET.bat` and select the images, annotation and output folders in the dialogs. It also asks whether to run the extended five-mode OCR comparison and the six-image Harris parameter study.

The script validates the annotation structure and corner bounds, runs recursive batch scanning with the primary PSM 10 plus secondary PSM 7 confirmation pass, then creates:

- `REPORT_SUMMARY.md`: actual macro-averaged results and a paste-ready report paragraph;
- `dataset_quality.json` and `dataset_manifest.csv`: dataset size, image dimensions, C1-C5 coverage, optional puzzle IDs and missing annotation warnings;
- `evaluation/per_image_metrics.csv` and `condition_metrics.csv`: per-image and per-condition metrics;
- `cell_diagnostics.csv`: one row per cell, with ground truth, prediction, error type, precheck decision and each OCR view's digit/confidence;
- `confusion_matrix.csv` and `figures/cell_confusion_matrix.png`: digit/empty-cell confusion matrix;
- `error_examples/`: raw, Otsu and adaptive crops for representative missed digits, substitutions and false positives in empty cells;
- `figures/`: accuracy by image, accuracy/corner error versus obliqueness, grid geometry plots and the most-oblique A3/A4/H4 comparison.

For a controlled segmentation-mode comparison (five extra dataset scans), run:

```bash
python diagnose_dataset.py --images data/personal/images --annotations data/personal/annotations --out report/psm_study --psm-sweep
```

The PSM comparison disables the fallback pass, so it compares primary modes 6/7/8/10/13 fairly. The diagnosis prints progress for each mode, applies a per-batch timeout (default: 24 seconds per annotated image plus 30 seconds, minimum 90 seconds), and records timed-out/failed modes rather than silently treating them as successful. You can override the limit with `--command-timeout 600`. To also run the required 36-setting Harris study on at least six annotated photographs:

```bash
python diagnose_dataset.py --images data/personal/images --annotations data/personal/annotations --out report/full_study --psm-sweep --parameter-study --parameter-limit 6
```

The diagnostic script does not invent annotations. Results are only meaningful when each JSON grid and corner set has been manually checked against its original photograph. It cannot automatically verify the handwritten card, original EXIF capture, or that the puzzles came from three different sources; those still require manual confirmation.


### OpenCV 4/5 HoughLinesP compatibility

The detector accepts both common `cv2.HoughLinesP` result layouts: `(N, 1, 4)` and `(N, 4)`. OpenCV 4 and 5 builds can expose different array layouts; the line coordinates are normalized to an `N x 4` array before processing. The supported dependency range is `opencv-python>=4.8,<6`.
