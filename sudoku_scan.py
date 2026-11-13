from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import cv2

from sudokuscan.config import ScanConfig
from sudokuscan.pipeline import process_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect, rectify and read a photographed 9x9 Sudoku puzzle."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Path to one image")
    source.add_argument("--batch", type=Path, help="Directory of images (recursive batch/grader mode)")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    solve_group = parser.add_mutually_exclusive_group()
    solve_group.add_argument("--solve", dest="solve", action="store_true",
                             help="Solve the recognized Sudoku and test uniqueness (default)")
    solve_group.add_argument("--no-solve", dest="solve", action="store_false",
                             help="Only recognize the givens; skip the solver")
    parser.set_defaults(solve=True)
    parser.add_argument("--ar-overlay", action="store_true", help="Bonus B2: project digits onto the source photo")
    parser.add_argument("--many-point", action="store_true", help="Bonus B3: Harris grid-point RANSAC homography")
    parser.add_argument("--orientation-search", action="store_true", help="Bonus B4: test 0/90/180/270 degree orientations")
    parser.add_argument("--harris-block-size", type=int, default=2)
    parser.add_argument("--harris-ksize", type=int, default=3)
    parser.add_argument("--harris-k", type=float, default=0.04)
    parser.add_argument("--response-threshold", type=float, default=ScanConfig().response_threshold,
                        help="Harris response threshold as a fraction of the maximum (default 0.001)")
    parser.add_argument("--ocr-psm", type=int, choices=[6, 7, 8, 10, 13], default=10,
                        help="Primary Tesseract mode (default 10: single character)")
    parser.add_argument("--ocr-fallback-psm", type=int, choices=[6, 7, 8, 10, 13], default=7,
                        help="Secondary Tesseract confirmation mode (default 7: single text line)")
    parser.add_argument("--no-ocr-fallback", action="store_true",
                        help="Disable the secondary OCR confirmation pass (useful for controlled PSM experiments)")
    parser.add_argument("--cell-margin-fraction", type=float, default=0.15,
                        help="Fraction cropped from each cell edge before emptiness/OCR (default: 0.15)")
    parser.add_argument("--max-detection-side", type=int, default=1600)
    parser.add_argument("--tesseract-cmd", type=str, default=None,
                        help="Optional path to the Tesseract executable")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tesseract_cmd:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
        except Exception as exc:
            print(f"Warning: could not configure Tesseract: {exc}", file=sys.stderr)
    from sudokuscan.recognition import _tesseract_executable
    if _tesseract_executable() is None:
        print("WARNING: Tesseract executable was not found; OCR cells will be blank. "
              "Install Tesseract or pass --tesseract-cmd.", file=sys.stderr)

    try:
        cfg = ScanConfig(
            harris_block_size=args.harris_block_size,
            harris_ksize=args.harris_ksize,
            harris_k=args.harris_k,
            response_threshold=args.response_threshold,
            ocr_psm=args.ocr_psm,
            ocr_fallback_psm=None if args.no_ocr_fallback else args.ocr_fallback_psm,
            cell_margin_fraction=args.cell_margin_fraction,
            max_detection_side=args.max_detection_side,
        )
    except ValueError as exc:
        # `parser` is local to parse_args(); report validation errors here
        # without referencing an out-of-scope variable.
        print(f"sudoku_scan.py: error: {exc}", file=sys.stderr)
        return 2
    batch_root: Path | None = None
    if args.input:
        images = [args.input]
    else:
        if not args.batch.is_dir():
            print(f"Batch path is not a directory: {args.batch}", file=sys.stderr)
            return 2
        batch_root = args.batch.resolve()
        output_root = args.outdir.resolve()
        if output_root == batch_root:
            print("The output directory cannot be the same as the batch input directory.",
                  file=sys.stderr)
            return 2
        # Search recursively so selecting a parent folder also processes images
        # stored in per-puzzle subfolders. Only exclude a result tree when it is
        # actually nested inside the selected input tree; an output parent of the
        # input directory must not cause every source image to be skipped.
        output_is_inside_input = batch_root in output_root.parents
        images = []
        for candidate in args.batch.rglob("*"):
            if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved = candidate.resolve()
            if output_is_inside_input and (resolved == output_root or output_root in resolved.parents):
                continue
            images.append(candidate)
        images.sort(key=lambda item: str(item.relative_to(args.batch)))

    if not images:
        print("No supported images found.", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)
    failures = 0
    batch_records: list[dict[str, object]] = []
    for path in images:
        image_started = time.perf_counter()
        if batch_root is not None:
            relative = path.resolve().relative_to(batch_root)
            image_outdir = args.outdir / relative.parent
        else:
            relative = path.name
            image_outdir = args.outdir
        image_outdir.mkdir(parents=True, exist_ok=True)
        try:
            result = process_image(
                path, image_outdir, cfg, solve=args.solve,
                ar_overlay=args.ar_overlay, many_point=args.many_point,
                orientation_search=args.orientation_search,
            )
            solver_info = result.get("solver")
            solver_text = (f", solver={solver_info['status']}" if solver_info else "")
            if not args.quiet:
                print(f"{path.name}: verdict={result['verdict']}{solver_text}, "
                      f"runtime={result['runtime_s']:.3f}s")
            batch_records.append({
                "image": str(relative), "result": str((image_outdir / f"{path.stem}_result.json")),
                "verdict": result.get("verdict"),
                "solver_status": solver_info.get("status") if solver_info else None,
                "runtime_s": result.get("runtime_s"), "processing_error": None,
            })
        except Exception as exc:
            failures += 1
            # One bad image must not prevent other files from being scanned/solved.
            error_result = {
                "image": path.name, "corners": [], "A3": [], "A4": [], "H4": [],
                "a3_corner_err_px": None, "a4_rms_px": None,
                "grid": [[0] * 9 for _ in range(9)], "verdict": "not_found",
                "runtime_s": float(time.perf_counter() - image_started),
                "processing_error": str(exc),
                "solver": {"status": "not_run_due_to_processing_error"} if args.solve else None,
            }
            (image_outdir / f"{path.stem}_result.json").write_text(
                json.dumps(error_result, indent=2), encoding="utf-8")
            (image_outdir / f"{path.stem}_grid.txt").write_text(
                ("000000000\n" * 9), encoding="utf-8")
            batch_records.append({
                "image": str(relative), "result": str(image_outdir / f"{path.stem}_result.json"),
                "verdict": "not_found", "solver_status": "not_run_due_to_processing_error" if args.solve else None,
                "runtime_s": error_result["runtime_s"], "processing_error": str(exc),
            })
            if not args.quiet:
                print(f"{path.name}: processing failed: {exc}", file=sys.stderr)
                traceback.print_exc()

    if batch_root is not None:
        summary = {
            "source_directory": str(batch_root), "recursive": True,
            "images_found": len(images), "processing_failures": failures,
            "successful_scans": len(images) - failures,
            "solver_enabled": bool(args.solve), "images": batch_records,
        }
        (args.outdir / "batch_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
    if not args.quiet:
        print(f"Processed {len(images)} image(s); processing failures: {failures}.")
        if batch_root is not None:
            solved_count = sum(str(record.get("solver_status", "")).startswith("solved_")
                               for record in batch_records)
            print(f"Solver completed a solution for {solved_count} image(s).")
    # Return non-zero for a partial batch failure, but all other images and
    # their result files have already been processed and preserved.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
