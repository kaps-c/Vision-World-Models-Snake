"""Shared figure style: one palette and rcParams for runs and notebooks.

Categorical slots are validated for colour-vision deficiency (adjacent
CVD dE >= 9). Slots 3-4 sit below 3:1 contrast on the surface, so figures
always carry direct labels and ship with a companion table.
"""

from __future__ import annotations

import matplotlib as mpl

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
BASELINE = "#8a8984"  # reference lines (identity, ceilings) - not a series
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # fixed order, never cycled


def apply() -> None:
    """Apply the project style to matplotlib."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_MUTED,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 2,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.size": 9,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def log_axis(ax) -> None:
    """Log-scale y with readable 1-2-5 tick labels (not a lone ``10^0``)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
