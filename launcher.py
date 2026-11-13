from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_FILTERS = [("Sudoku photographs", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]


def choose_target() -> tuple[str, Path] | None:
    """Show a simple numbered menu, then use native Windows file/folder dialogs."""
    print("============================================================")
    print(" SudokuScan - choose what to scan")
    print("============================================================")
    print(" 1. Scan one Sudoku photograph")
    print(" 2. Batch scan and solve a folder (including subfolders)")
    print(" 3. Exit")
    while True:
        try:
            choice = input("Enter 1, 2 or 3: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice == "3":
            return None
        if choice in {"1", "2"}:
            break
        print("Please enter 1, 2 or 3.")

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"Could not open the Windows picker: {exc}")
        raw = input("Enter the full image/folder path manually: ").strip().strip('"')
        path = Path(raw).expanduser()
        if not path.exists():
            print("The selected path does not exist.")
            return None
        return ("batch" if choice == "2" else "single", path)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if choice == "1":
            selected = filedialog.askopenfilename(
                title="Choose one Sudoku photograph", filetypes=IMAGE_FILTERS,
            )
            mode = "single"
        else:
            selected = filedialog.askdirectory(
                title="Choose a folder of Sudoku photographs (subfolders included)",
                mustexist=True,
            )
            mode = "batch"
    finally:
        root.destroy()
    if not selected:
        print("Nothing selected; returning to the menu.")
        return choose_target()
    return mode, Path(selected)


def main() -> int:
    if len(sys.argv) > 1:
        supplied = Path(sys.argv[1].strip('"')).expanduser()
        if not supplied.exists():
            print(f"Path does not exist: {supplied}")
            return 2
        mode, target = ("batch", supplied) if supplied.is_dir() else ("single", supplied)
    else:
        selection = choose_target()
        if selection is None:
            return 0
        mode, target = selection

    if mode == "batch" and not target.is_dir():
        print("Batch mode requires a directory.")
        return 2
    if mode == "single" and (not target.is_file() or target.suffix.lower() not in {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    }):
        print("Please choose a supported image file.")
        return 2

    from sudokuscan.recognition import _tesseract_executable
    if _tesseract_executable() is None:
        print("WARNING: Tesseract OCR is not installed or was not found.")
        print("The scanner can locate/rectify the grid, but it cannot read digits or solve it.")
        answer = input("Continue anyway? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            return 1

    outdir = ROOT / "results" / ("batch" if mode == "batch" else "single")
    command = [sys.executable, str(ROOT / "sudoku_scan.py"),
               "--outdir", str(outdir), "--solve"]
    if mode == "batch":
        command += ["--batch", str(target)]
    else:
        command += ["--input", str(target)]

    print()
    print(f"Mode: {'RECURSIVE BATCH + SOLVE' if mode == 'batch' else 'SINGLE IMAGE + SOLVE'}")
    print(f"Input: {target}")
    print(f"Output: {outdir}")
    print("=" * 60)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    print("=" * 60)
    print(f"Scanner exit code: {completed.returncode}")
    print(f"Results: {outdir}")
    if os.name == "nt" and outdir.exists():
        try:
            os.startfile(str(outdir))  # type: ignore[attr-defined]
        except OSError:
            pass
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
