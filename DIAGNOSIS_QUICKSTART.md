# Personal dataset diagnosis quick start

## 1. Prepare the dataset

For every original phone photograph, create one manually checked Appendix A JSON annotation. The annotation must contain the original image filename, capture condition labels (`C1`–`C5`), four outer grid corners in original-image coordinates ordered TL/TR/BR/BL, and the 9x9 grid with `0` for empty cells. Do not annotate using the scanner's predicted corners or OCR grid; those fields are the ground truth used to calculate error.

You may keep images and annotations together:

```text
data/personal/
  photo_01.jpg
  photo_01.json
  photo_02.jpg
  photo_02.json
```

Or in matching folders:

```text
data/personal/
  images/photo_01.jpg
  images/photo_02.jpg
  annotations/photo_01.json
  annotations/photo_02.json
```

See `data/personal/README.md` for the required annotation schema and the course capture requirements.

## 2. Run the diagnosis

Basic diagnosis (one scanner pass using primary PSM 10 plus secondary PSM 7 confirmation):

```bash
python diagnose_dataset.py --images data/personal/images --annotations data/personal/annotations --out report/personal_diagnosis
```

If images and JSON files are together, omit `--annotations`:

```bash
python diagnose_dataset.py --images data/personal --out report/personal_diagnosis
```

Windows users can double-click `DIAGNOSE_DATASET.bat`, select the three folders, and choose whether to run the optional studies.

## 3. Run the extended studies

Compare Tesseract page-segmentation modes against the same ground truth:

```bash
python diagnose_dataset.py --images data/personal/images --annotations data/personal/annotations --out report/psm_study --psm-sweep
```

Run the required Harris parameter study on six annotated images:

```bash
python diagnose_dataset.py --images data/personal/images --annotations data/personal/annotations --out report/full_study --parameter-study --parameter-limit 6
```

Run both extended studies:

```bash
python diagnose_dataset.py --images data/personal/images --annotations data/personal/annotations --out report/full_study --psm-sweep --parameter-study --parameter-limit 6
```

The PSM sweep intentionally disables the secondary OCR fallback so each primary PSM is compared fairly. The script reports progress and records any timed-out mode rather than silently presenting it as successful. The per-batch timeout defaults to 24 seconds per annotated photo plus 30 seconds, with a 90-second minimum; override it with `--command-timeout 600` if needed.

## 4. Use the generated results in the report

Open `REPORT_SUMMARY.md` first. Then inspect:

- `evaluation/per_image_metrics.csv`: per-photo corner error, A3/A4/H4 cell accuracy, empty-cell F1, exact-grid result, runtime and robustness;
- `condition_metrics.csv`: C1–C5 breakdown;
- `cell_diagnostics.csv`: all 81 cells per photo, ground truth, prediction, error category, filled/empty decision, and OCR variant readings/confidences;
- `confusion_matrix.csv` and `figures/cell_confusion_matrix.png`: digit confusions;
- `error_examples/`: actual crops for missed digits, substitutions and false positives in empty cells;
- `psm_sweep_summary.csv`: dataset-measured PSM comparison, when enabled;
- `harris_parameter_sweep.csv` and `figures/harris_parameter_study.png`: parameter-study measurements, when enabled.

The script measures accuracy only by comparing against your annotations. Review every ground-truth grid and corner set and inspect the failure crops before copying the results into the report. Synthetic examples are for software tests only and must not be described as your personal-dataset results.
