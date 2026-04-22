"""Visualize interpretability vs. performance from overall_results.csv.

Usage:
    uv run visualize.py
    uv run visualize.py path/to/overall_results.csv
    uv run visualize.py path/to/overall_results.csv --output path/to/output.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_GROUPS = {
    "black-box": {"RF", "GBM", "MLP", "TabPFN"},
    "imodels":   {"FIGS_mini", "FIGS_large", "RuleFit"},
    "additive":    {"OLS", "LASSO", "LassoCV", "RidgeCV", "EBM", "PyGAM"},
    "tree":      {"DT_mini", "DT_large", "HSTree_mini", "HSTree_large"},
}
GROUP_COLORS = {
    "black-box": "black",
    "imodels":   "tab:orange",
    "additive":  "tab:green",
    "tree":      "tab:red",
}


_blues_cmap = plt.cm.Blues


def _is_known(name: str) -> bool:
    """Return True if the model name belongs to any MODEL_GROUPS entry."""
    return any(name in members for members in MODEL_GROUPS.values())


def _model_colors_and_markers(names: list[str]) -> tuple[list, list[str]]:
    """Return a color and marker for each name.

    Known models get group colors and a circle ("o") marker.
    Unknown models get Blues-cmap colors and a diamond ("D") marker.
    """
    # Collect unknown names in order of first appearance
    seen = set()
    unknown_names = []
    for n in names:
        if n not in seen and not _is_known(n):
            unknown_names.append(n)
            seen.add(n)
    # Build color map for unknowns based on how many there are
    n_unknown = len(unknown_names)
    unknown_color_map = {}
    for i, uname in enumerate(unknown_names):
        t = 0.3 + 0.6 * i / max(1, n_unknown - 1)
        unknown_color_map[uname] = _blues_cmap(t)
    # Assign colors and markers
    colors = []
    markers = []
    for name in names:
        if _is_known(name):
            for group, members in MODEL_GROUPS.items():
                if name in members:
                    colors.append(GROUP_COLORS[group])
                    break
            markers.append("X")
        else:
            colors.append(unknown_color_map[name])
            markers.append("o")
    return colors, markers


def plot_interp_vs_performance(csv_path: str | Path, out_path: str | Path | None = None) -> None:
    """Scatter: x=frac_interpretability_tests_passed, y=mean_rank, labeled by description."""
    csv_path = Path(csv_path)
    if out_path is None:
        out_path = csv_path.parent / "interpretability_vs_performance.png"
    out_path = Path(out_path)

    df = pd.read_csv(csv_path)
    df = df.iloc[::-1].reset_index(drop=True)

    required = {"mean_rank", "frac_interpretability_tests_passed", "model_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.replace("", np.nan).replace("nan", np.nan).dropna(subset=list(required)).copy()
    df["mean_rank"] = df["mean_rank"].astype(float)
    df["frac_interpretability_tests_passed"] = df["frac_interpretability_tests_passed"].astype(float)

    names  = df["model_name"].tolist()
    x      = df["frac_interpretability_tests_passed"].to_numpy()
    y      = df["mean_rank"].to_numpy()
    colors, markers = _model_colors_and_markers(names)

    try:
        from adjustText import adjust_text
        use_adjust = True
    except ImportError:
        use_adjust = False

    fig, ax = plt.subplots(figsize=(10, 10))
    for xi, yi, color, marker in zip(x, y, colors, markers):
        ax.scatter(xi, yi, color=color, marker=marker, s=60, zorder=5,
                   edgecolors="white", linewidths=0.6)


    texts = [ax.text(xi, yi, name, fontsize=8.5, color=color)
             for xi, yi, name, color in zip(x, y, names, colors)]

    # change ylim to ignore largest outlier
    # second_largest = np.partition(y, -2)[-2]
    # ax.set_ylim(top=second_largest * 1.1, bottom=y.min() - 0.05) #, bottom=0)  # start y-axis at 0 for better interpretability

    if use_adjust:
        adjust_text(texts, x=x, y=y, ax=ax,
                    # expand=(1.1, 1.4),
                    # force_text=(400.0, 400.0),
                    # force_points=(0.4, 0.8),
                    # force_explode=(0.3, 0.5),
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))

    ax.set_xlabel("Interpretability (Frac Interpretability Tests Passed)", fontsize=10)
    ax.set_ylabel("Prediction performance (Rank, lower is better)", fontsize=10)
    ax.set_title("Interpretability vs. Performance", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="X", color="w",
               markerfacecolor=GROUP_COLORS[g], markersize=9,
               label=g.replace("-", " ").title())
        for g in GROUP_COLORS if any(n in MODEL_GROUPS[g] for n in names)
    ]
    if any(not _is_known(n) for n in names):
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="steelblue", markersize=8,
                   label="Evolved")
        )
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "overall_results.csv",
        help="Path to overall_results.csv",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output PNG path (default: <input_dir>/interpretability_vs_performance.png)",
    )
    args = parser.parse_args()
    plot_interp_vs_performance(args.input, args.output)
