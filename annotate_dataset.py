#!/usr/bin/env python3
"""High-precision, human-in-the-loop Appendix A annotation GUI.

Click the four outer Sudoku intersections in the displayed order TL, TR, BR, BL.
Use the magnifier, zoom and pan for precise placement. Fill the 9x9 printed givens
and choose the capture conditions before saving.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

from sudokuscan.annotation import (
    CONDITIONS,
    CORNER_LABELS,
    annotation_path,
    validate_corners,
    validate_grid,
    write_annotation,
)

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _tk_modules():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:  # pragma: no cover - depends on desktop Python install
        raise RuntimeError(
            "Tkinter is unavailable. On Windows, reinstall Python with 'tcl/tk and IDLE' enabled."
        ) from exc
    return tk, filedialog, messagebox, ttk


class DatasetAnnotator:
    def __init__(self, image_root: Path, annotation_root: Path):
        tk, filedialog, messagebox, ttk = _tk_modules()
        self.tk, self.filedialog, self.messagebox, self.ttk = tk, filedialog, messagebox, ttk
        self.image_root = image_root.resolve()
        self.annotation_root = annotation_root.resolve()
        self.image_paths = sorted(
            (p for p in self.image_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED),
            key=lambda p: str(p.relative_to(self.image_root)).casefold(),
        )
        if not self.image_paths:
            raise RuntimeError(f"No supported images found in: {self.image_root}")

        self.root = tk.Tk()
        self.root.title("SudokuScan — Personal Dataset Annotator")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        initial_width = max(900, min(1510, screen_width - 32))
        initial_height = max(640, min(940, screen_height - 80))
        self.root.geometry(f"{initial_width}x{initial_height}")
        self.root.minsize(min(900, screen_width - 24), min(640, screen_height - 60))

        self.index = 0
        self.image_bgr: np.ndarray | None = None
        self.image_rgb: np.ndarray | None = None
        self.image_pil: Image.Image | None = None
        self.image_width = 0
        self.image_height = 0
        self.photo = None
        self.points: list[tuple[float, float]] = []
        self.zoom = 1.0
        self.fit_scale = 1.0
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.fit_pending = True
        self.pan_anchor: tuple[int, int] | None = None
        self.last_mouse_image: tuple[float, float] | None = None
        self.magnifier_photo = None
        self._saved_state = None
        self.grid_vars: list[list[tk.StringVar]] = []
        self.grid_entries = []
        self._entry_positions = {}
        self.condition_vars = {c: tk.BooleanVar(value=False) for c in CONDITIONS}
        self.corner_status_var = tk.StringVar(value="Click the TOP-LEFT corner (TL).")
        self.file_status_var = tk.StringVar(value="")
        self.puzzle_id_var = tk.StringVar(value="")
        self.grid_verified_var = tk.BooleanVar(value=False)
        self.coordinate_var = tk.StringVar(value="Move over the image to inspect pixel coordinates.")
        self._build_ui()
        self.load_current_image()
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        body = ttk.Frame(self.root, padding=8)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        header = ttk.Frame(body)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 7))
        ttk.Label(header, textvariable=self.file_status_var, font=("Segoe UI", 11, "bold")).pack(side="left")
        ttk.Label(header, text="  |  Click order: TL → TR → BR → BL", foreground="#176b37").pack(side="left")

        self.canvas = tk.Canvas(body, background="#202329", highlightthickness=1, highlightbackground="#555")
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 9))
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.2))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / 1.2))
        self.canvas.bind("<ButtonPress-3>", self._pan_start)
        self.canvas.bind("<B3-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._pan_end)

        side_outer = ttk.Frame(body, width=365)
        side_outer.grid(row=1, column=1, sticky="ns")
        side_outer.grid_propagate(False)
        side_canvas = tk.Canvas(side_outer, highlightthickness=0, width=342)
        side_scrollbar = ttk.Scrollbar(side_outer, orient="vertical", command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scrollbar.set)
        side_scrollbar.pack(side="right", fill="y")
        side_canvas.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(side_canvas, width=342)
        side_window = side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", lambda _e: side_canvas.configure(scrollregion=side_canvas.bbox("all")))
        side_canvas.bind("<Configure>", lambda e: side_canvas.itemconfigure(side_window, width=e.width))
        side_canvas.bind("<MouseWheel>", lambda e: side_canvas.yview_scroll(-1 * int(e.delta / 120), "units"))
        side.columnconfigure(0, weight=1)

        ttk.Label(side, text="Corner annotation", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(side, textvariable=self.corner_status_var, wraplength=330, foreground="#145caa").grid(row=1, column=0, sticky="w", pady=(4, 5))
        ttk.Label(side, textvariable=self.coordinate_var, wraplength=330).grid(row=2, column=0, sticky="w")

        magnifier_box = ttk.LabelFrame(side, text="Precision magnifier (cursor center)", padding=5)
        magnifier_box.grid(row=3, column=0, sticky="ew", pady=7)
        self.magnifier_label = ttk.Label(magnifier_box, text="Move cursor over a corner", anchor="center")
        self.magnifier_label.pack(fill="both")

        corner_buttons = ttk.Frame(side)
        corner_buttons.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(corner_buttons, text="Undo last corner", command=self.undo_corner).pack(side="left", fill="x", expand=True)
        ttk.Button(corner_buttons, text="Reset corners", command=self.reset_corners).pack(side="left", fill="x", expand=True, padx=(5, 0))
        ttk.Label(side, text="Zoom: mouse wheel   |   Pan: right-drag   |   Undo: Backspace", wraplength=330).grid(row=5, column=0, sticky="w", pady=(0, 8))

        ttk.Separator(side).grid(row=6, column=0, sticky="ew", pady=5)
        ttk.Label(side, text="Printed Sudoku givens", font=("Segoe UI", 12, "bold")).grid(row=7, column=0, sticky="w")
        ttk.Label(side, text="Enter the digits printed on paper. Use 0 for empty cells.", wraplength=330).grid(row=8, column=0, sticky="w", pady=(2, 5))
        grid_frame = ttk.Frame(side)
        grid_frame.grid(row=9, column=0, sticky="w")
        self._create_grid_editor(grid_frame)
        ttk.Checkbutton(
            side, text="I checked all 81 cells against the photograph",
            variable=self.grid_verified_var,
        ).grid(row=10, column=0, sticky="w", pady=(5, 3))

        ttk.Separator(side).grid(row=11, column=0, sticky="ew", pady=5)
        ttk.Label(side, text="Capture conditions", font=("Segoe UI", 11, "bold")).grid(row=12, column=0, sticky="w")
        descriptions = {
            "C1": "Near-frontal", "C2": "Oblique / keystone", "C3": "Rotated 15–30°",
            "C4": "Shadow / uneven lighting", "C5": "Cluttered background",
        }
        cond_frame = ttk.Frame(side)
        cond_frame.grid(row=13, column=0, sticky="w")
        for i, code in enumerate(CONDITIONS):
            ttk.Checkbutton(cond_frame, text=f"{code} — {descriptions[code]}", variable=self.condition_vars[code]).grid(row=i, column=0, sticky="w")

        pid_frame = ttk.Frame(side)
        pid_frame.grid(row=14, column=0, sticky="ew", pady=(5, 0))
        ttk.Label(pid_frame, text="Puzzle ID (optional):").pack(side="left")
        ttk.Entry(pid_frame, textvariable=self.puzzle_id_var, width=16).pack(side="left", padx=5)

        controls = ttk.Frame(body)
        controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(controls, text="◀ Previous", command=self.previous_image).pack(side="left")
        ttk.Button(controls, text="Next ▶", command=self.next_image).pack(side="left", padx=5)
        ttk.Button(controls, text="Save", command=self.save_current).pack(side="right")
        ttk.Button(controls, text="Save & Next", command=lambda: self.save_current(go_next=True)).pack(side="right", padx=5)
        ttk.Button(controls, text="Skip", command=self.next_image).pack(side="right", padx=5)

        self.root.bind("<BackSpace>", self._global_undo)
        self.root.bind("<Control-s>", lambda _e: self.save_current())
        self.root.bind("<KeyPress-r>", self._global_reset)

    def _create_grid_editor(self, parent) -> None:
        tk = self.tk
        validate_cmd = (self.root.register(self._validate_digit), "%P")
        self.grid_vars.clear()
        self.grid_entries.clear()
        for r in range(9):
            row_vars = []
            row_entries = []
            for c in range(9):
                var = tk.StringVar(value="0")
                entry = tk.Entry(
                    parent, textvariable=var, width=2, justify="center", font=("Consolas", 11, "bold"),
                    validate="key", validatecommand=validate_cmd, relief="solid", borderwidth=1,
                )
                padx = (1, 3 if c in (2, 5) else 1)
                pady = (1, 3 if r in (2, 5) else 1)
                entry.grid(row=r, column=c, padx=padx, pady=pady, ipady=3)
                entry.bind("<FocusIn>", lambda e: e.widget.select_range(0, "end"))
                entry.bind("<KeyRelease>", self._on_grid_edit)
                entry.bind("<Up>", lambda e, rr=r, cc=c: self._move_grid_focus(rr - 1, cc))
                entry.bind("<Down>", lambda e, rr=r, cc=c: self._move_grid_focus(rr + 1, cc))
                entry.bind("<Left>", lambda e, rr=r, cc=c: self._move_grid_focus(rr, cc - 1))
                entry.bind("<Right>", lambda e, rr=r, cc=c: self._move_grid_focus(rr, cc + 1))
                row_vars.append(var)
                row_entries.append(entry)
                self._entry_positions[str(entry)] = (r, c)
            self.grid_vars.append(row_vars)
            self.grid_entries.append(row_entries)

    @staticmethod
    def _validate_digit(proposed: str) -> bool:
        return proposed == "" or (len(proposed) == 1 and proposed in "0123456789")

    def _on_grid_edit(self, event=None) -> None:
        self.grid_verified_var.set(False)
        # Rapid row-major entry: after typing one digit, move to the next cell.
        # Arrow keys remain available for skips and corrections.
        if event is not None and getattr(event, "keysym", "") in tuple("0123456789"):
            position = self._entry_positions.get(str(event.widget))
            if position is not None:
                row, col = position
                if col < 8:
                    self.root.after_idle(lambda: self._move_grid_focus(row, col + 1))
                elif row < 8:
                    self.root.after_idle(lambda: self._move_grid_focus(row + 1, 0))

    def _global_undo(self, _event=None):
        # Backspace edits a digit when an Entry has focus; otherwise it undoes a corner.
        if isinstance(self.root.focus_get(), self.tk.Entry):
            return None
        self.undo_corner()
        return "break"

    def _global_reset(self, _event=None):
        # Do not steal the letter 'r' while the user is editing the puzzle ID.
        if isinstance(self.root.focus_get(), self.tk.Entry):
            return None
        self.reset_corners()
        return "break"

    def _state_fingerprint(self) -> str:
        state = {
            "points": [[round(x, 3), round(y, 3)] for x, y in self.points],
            "grid": self._current_grid(),
            "conditions": [c for c in CONDITIONS if self.condition_vars[c].get()],
            "puzzle_id": self.puzzle_id_var.get(),
            "grid_verified": bool(self.grid_verified_var.get()),
        }
        return json.dumps(state, sort_keys=True)

    def _is_dirty(self) -> bool:
        return self._saved_state is not None and self._state_fingerprint() != self._saved_state

    def _move_grid_focus(self, row: int, col: int):
        row = max(0, min(8, row))
        col = max(0, min(8, col))
        self.grid_entries[row][col].focus_set()
        self.grid_entries[row][col].select_range(0, "end")
        return "break"

    def _annotation_file(self, image_path: Path) -> Path:
        return annotation_path(image_path, self.image_root, self.annotation_root)

    def load_current_image(self) -> None:
        path = self.image_paths[self.index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            self.messagebox.showerror("Cannot open image", f"OpenCV could not read:\n{path}\n\nThis file will be skipped.")
            if self.index + 1 < len(self.image_paths):
                self.index += 1
                self.load_current_image()
            return
        self.image_bgr = image
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_pil = Image.fromarray(self.image_rgb)
        self.image_height, self.image_width = image.shape[:2]
        self.points = []
        self.zoom = 1.0
        self.fit_pending = True
        self.last_mouse_image = None
        self.corner_status_var.set("Click the TOP-LEFT corner (TL).")
        rel = path.relative_to(self.image_root)
        self.file_status_var.set(f"Image {self.index + 1}/{len(self.image_paths)}: {rel}  ({self.image_width}×{self.image_height})")
        for row in range(9):
            for col in range(9):
                self.grid_vars[row][col].set("0")
        for var in self.condition_vars.values():
            var.set(False)
        self.puzzle_id_var.set("")
        self.grid_verified_var.set(False)

        ann_path = self._annotation_file(path)
        if ann_path.is_file():
            try:
                data = json.loads(ann_path.read_text(encoding="utf-8"))
                corners = data.get("corners", [])
                if isinstance(corners, list) and len(corners) <= 4:
                    self.points = [(float(p[0]), float(p[1])) for p in corners]
                grid = data.get("grid", [])
                if len(grid) == 9 and all(len(row) == 9 for row in grid):
                    for r in range(9):
                        for c in range(9):
                            self.grid_vars[r][c].set(str(int(grid[r][c])))
                    self.grid_verified_var.set(True)
                for code in CONDITIONS:
                    self.condition_vars[code].set(code in data.get("conditions", []))
                self.puzzle_id_var.set(str(data.get("puzzle_id", "")))
            except Exception as exc:
                self.messagebox.showwarning("Annotation could not be loaded", f"{ann_path}\n\n{exc}")
        self._update_corner_status()
        self._saved_state = self._state_fingerprint()
        self.root.after(50, self._fit_image)

    def _fit_image(self) -> None:
        if self.image_pil is None:
            return
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        self.fit_scale = min((cw - 24) / self.image_width, (ch - 24) / self.image_height)
        self.zoom = 1.0
        self.scale = self.fit_scale * self.zoom
        dw = self.image_width * self.scale
        dh = self.image_height * self.scale
        self.offset_x = (cw - dw) / 2
        self.offset_y = (ch - dh) / 2
        self._redraw()
        self.fit_pending = False

    def _on_canvas_resize(self, _event=None) -> None:
        if self.image_pil is not None and self.fit_pending:
            self.root.after(80, self._fit_image)

    def _redraw(self) -> None:
        if self.image_pil is None:
            return
        dw = max(1, int(round(self.image_width * self.scale)))
        dh = max(1, int(round(self.image_height * self.scale)))
        interpolation = Image.Resampling.LANCZOS if self.scale < 1 else Image.Resampling.BICUBIC
        display = self.image_pil.resize((dw, dh), interpolation)
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor="nw", tags="photo")

        display_points = [self._image_to_canvas(x, y) for x, y in self.points]
        if len(display_points) >= 2:
            for i in range(len(display_points) - 1):
                self.canvas.create_line(*display_points[i], *display_points[i + 1], fill="#00e676", width=2)
        if len(display_points) == 4:
            self.canvas.create_line(*display_points[-1], *display_points[0], fill="#00e676", width=2)
        colors = ("#00e5ff", "#ffeb3b", "#ff9800", "#e040fb")
        for i, (x, y) in enumerate(display_points):
            color = colors[i]
            self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, outline="black", width=3)
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline=color, width=3)
            self.canvas.create_text(x + 12, y - 14, text=f"{CORNER_LABELS[i]} ({i+1})", anchor="sw", fill="black", font=("Segoe UI", 10, "bold"))
            self.canvas.create_text(x + 12, y - 14, text=f"{CORNER_LABELS[i]} ({i+1})", anchor="sw", fill=color, font=("Segoe UI", 10, "bold"))

    def _image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.offset_x + x * self.scale, self.offset_y + y * self.scale

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        ix = (x - self.offset_x) / self.scale
        iy = (y - self.offset_y) / self.scale
        if 0 <= ix < self.image_width and 0 <= iy < self.image_height:
            return ix, iy
        return None

    def _on_left_click(self, event) -> None:
        if len(self.points) >= 4:
            self.corner_status_var.set("Four corners already selected. Undo or reset to change them.")
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            return
        self.points.append(point)
        self.canvas.focus_set()
        self._update_corner_status()
        self._redraw()

    def _update_corner_status(self) -> None:
        if len(self.points) < 4:
            label = CORNER_LABELS[len(self.points)]
            self.corner_status_var.set(f"Next: click {label} — {self._corner_name(label)} ({len(self.points) + 1}/4).")
        else:
            valid, reason = validate_corners(self.points, self.image_width, self.image_height)
            self.corner_status_var.set("Four corners selected. Geometry looks valid." if valid else f"Check corner order/selection: {reason}")

    @staticmethod
    def _corner_name(label: str) -> str:
        return {"TL": "top-left", "TR": "top-right", "BR": "bottom-right", "BL": "bottom-left"}[label]

    def undo_corner(self) -> None:
        if self.points:
            self.points.pop()
            self._update_corner_status()
            self._redraw()

    def reset_corners(self) -> None:
        self.points.clear()
        self._update_corner_status()
        self._redraw()

    def _on_mouse_move(self, event) -> None:
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            return
        self.last_mouse_image = point
        self.coordinate_var.set(f"Original-image pixel: x={point[0]:.1f}, y={point[1]:.1f}")
        self._update_magnifier(point[0], point[1])

    def _update_magnifier(self, x: float, y: float) -> None:
        if self.image_rgb is None:
            return
        h, w = self.image_rgb.shape[:2]
        side = max(70, min(180, int(min(w, h) * 0.06)))
        cx, cy = int(round(x)), int(round(y))
        x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
        x1, y1 = min(w, cx + side // 2), min(h, cy + side // 2)
        crop = self.image_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return
        magnified = Image.fromarray(crop).resize((270, 270), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(magnified)
        mx = (cx - x0) * 270 / max(1, x1 - x0)
        my = (cy - y0) * 270 / max(1, y1 - y0)
        draw.line((mx, 0, mx, 270), fill="#ff1744", width=1)
        draw.line((0, my, 270, my), fill="#ff1744", width=1)
        draw.ellipse((mx - 3, my - 3, mx + 3, my + 3), outline="white", width=2)
        self.magnifier_photo = ImageTk.PhotoImage(magnified)
        self.magnifier_label.configure(image=self.magnifier_photo, text="")

    def _on_mouse_wheel(self, event) -> None:
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, x: int, y: int, factor: float) -> None:
        if self.image_pil is None:
            return
        image_point = self._canvas_to_image(x, y)
        if image_point is None:
            image_point = ((x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale)
        new_zoom = max(0.5, min(12.0, self.zoom * factor))
        self.zoom = new_zoom
        self.scale = self.fit_scale * self.zoom
        self.offset_x = x - image_point[0] * self.scale
        self.offset_y = y - image_point[1] * self.scale
        self._redraw()

    def _pan_start(self, event) -> None:
        self.pan_anchor = (event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def _pan_move(self, event) -> None:
        if self.pan_anchor is None:
            return
        dx, dy = event.x - self.pan_anchor[0], event.y - self.pan_anchor[1]
        self.offset_x += dx
        self.offset_y += dy
        self.pan_anchor = (event.x, event.y)
        self._redraw()

    def _pan_end(self, _event=None) -> None:
        self.pan_anchor = None
        self.canvas.configure(cursor="")

    def _current_grid(self) -> list[list[str]]:
        return [[var.get() for var in row] for row in self.grid_vars]

    def save_current(self, go_next: bool = False, show_dialog: bool = True) -> bool:
        if len(self.points) != 4:
            self.messagebox.showwarning("Four corners required", f"You selected {len(self.points)}/4 corners.\n\nClick TL, TR, BR and BL in that order.")
            return False
        valid, reason = validate_corners(self.points, self.image_width, self.image_height)
        if not valid:
            self.messagebox.showwarning("Check corner selection", reason)
            return False
        ok, reason, grid = validate_grid(self._current_grid())
        if not ok:
            self.messagebox.showwarning("Invalid Sudoku grid", reason)
            return False
        if not self.grid_verified_var.get():
            self.messagebox.showwarning("Confirm ground truth", "Please check all 81 cells against the original photograph and tick the verification box. Empty cells must be 0.")
            return False
        conditions = [code for code in CONDITIONS if self.condition_vars[code].get()]
        if not conditions:
            self.messagebox.showwarning("Capture conditions missing", "Select at least one applicable condition (C1–C5).")
            return False

        image_path = self.image_paths[self.index]
        relative = image_path.relative_to(self.image_root)
        payload = {
            "image": image_path.name,
            "conditions": conditions,
            "corners": [[round(float(x), 1), round(float(y), 1)] for x, y in self.points],
            "grid": grid,
        }
        puzzle_id = self.puzzle_id_var.get().strip()
        if puzzle_id:
            payload["puzzle_id"] = puzzle_id
        output_path = self._annotation_file(image_path)
        try:
            write_annotation(output_path, payload)
        except OSError as exc:
            self.messagebox.showerror("Save failed", str(exc))
            return False
        self.corner_status_var.set(f"Saved: {output_path.relative_to(self.annotation_root)}")
        self._saved_state = self._state_fingerprint()
        if go_next:
            self._navigate(1, confirm=False)
        elif show_dialog:
            self.messagebox.showinfo("Annotation saved", f"Saved annotation for:\n{relative}\n\n{output_path}")
        return True

    def _navigate(self, direction: int, confirm: bool = True) -> None:
        if confirm and self._is_dirty():
            answer = self.messagebox.askyesnocancel(
                "Unsaved annotation",
                "This image has unsaved changes.\n\nYes = Save before moving\nNo = Discard changes\nCancel = Stay on this image",
            )
            if answer is None:
                return
            if answer is True and not self.save_current(show_dialog=False):
                return
        new_index = self.index + direction
        if 0 <= new_index < len(self.image_paths):
            self.index = new_index
            self.load_current_image()
        elif direction > 0:
            self.messagebox.showinfo("Dataset complete", "You have reached the last image. All saved annotations are in the selected annotation folder.")

    def previous_image(self) -> None:
        self._navigate(-1)

    def next_image(self) -> None:
        self._navigate(1)

    def run(self) -> None:
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive SudokuScan Appendix A dataset annotator")
    parser.add_argument("--images", type=Path, help="Root folder containing original photographs (searched recursively)")
    parser.add_argument("--annotations", type=Path, help="Output root folder for same-relative-path JSON annotations")
    args = parser.parse_args(argv)

    tk, filedialog, messagebox, _ttk = _tk_modules()
    chooser_root = tk.Tk()
    chooser_root.withdraw()
    image_root = args.images
    if image_root is None:
        selected = filedialog.askdirectory(title="Select folder containing original Sudoku photographs")
        if not selected:
            chooser_root.destroy()
            return 0
        image_root = Path(selected)
    annotation_root = args.annotations
    if annotation_root is None:
        selected = filedialog.askdirectory(title="Select folder to save JSON annotations", mustexist=False)
        if not selected:
            chooser_root.destroy()
            return 0
        annotation_root = Path(selected)
    chooser_root.destroy()

    if not image_root.is_dir():
        print(f"Image folder does not exist: {image_root}", file=sys.stderr)
        return 2
    try:
        app = DatasetAnnotator(image_root, annotation_root)
        app.run()
    except Exception as exc:
        print(f"Annotation tool error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
