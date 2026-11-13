# Report workflow

1. Collect and annotate your own photographs. The final report cannot be completed from the supplied assignment PDF alone.
2. Run `tools/parameter_sweep.py` on at least six annotated images and plot both Harris candidate counts and corner errors with `tools/plot_parameter_sweep.py`.
3. Run the scanner over the personal set and `evaluate.py` against your manual annotations.
4. Generate the three Stage 4 plots with `tools/plot_metrics.py` and the most-oblique A3/A4/H4 comparison with `tools/make_comparison_figure.py`.
5. Copy the generated figures into `report/figures/` (the scripts use those defaults in the example commands), complete every placeholder in `report_template.tex`, and compile it with the IEEE conference template available in your LaTeX installation or Overleaf.
6. Keep the IEEE report to at most six pages plus appendices. Remove unused placeholder figures and replace all example wording with your measured evidence.

The report template includes an AI-use disclosure row for this project's ChatGPT-assisted implementation. Edit it to reflect what you actually used and retained.
