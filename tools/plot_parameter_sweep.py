from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Harris parameter-sweep CSV output.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("report/figures/harris_parameter_study.png"))
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("Parameter sweep file is empty.")
        return 1

    settings = [
        ("blockSize", "Harris blockSize"),
        ("ksize", "Harris ksize"),
        ("k", "Harris k"),
        ("response_threshold", "Relative response threshold"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (key, label) in zip(axes.flat, settings):
        grouped_errors = defaultdict(list)
        grouped_candidates = defaultdict(list)
        for row in rows:
            try:
                value = float(row[key])
                candidate_count = float(row["candidate_count"])
                grouped_candidates[value].append(candidate_count)
                error = float(row["corner_error_percent"])
                if np.isfinite(error):
                    grouped_errors[value].append(error)
            except (ValueError, TypeError, KeyError):
                continue
        x = sorted(grouped_candidates)
        error_means = [float(np.mean(grouped_errors[value])) if grouped_errors[value] else np.nan for value in x]
        candidate_means = [float(np.mean(grouped_candidates[value])) for value in x]
        if x:
            ax.plot(x, error_means, marker="o", label="Mean corner error")
            ax.set_ylabel("Mean corner error (% of side)")
            ax2 = ax.twinx()
            ax2.plot(x, candidate_means, marker="s", linestyle="--", label="Mean candidate count")
            ax2.set_ylabel("Mean Harris candidates")
            lines = ax.get_lines() + ax2.get_lines()
            ax.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=7)
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Harris parameter study (means across other settings and tested images)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
