"""Per-run evaluation artefacts: a markdown probe table and training curves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from snekwm.eval import style  # noqa: E402


def metrics_table(rows: Mapping[str, Mapping[str, float]], label: str = "Model") -> str:
    """Render ``{row: {column: value}}`` as markdown (3 d.p., ``—`` if absent)."""
    cols: list[str] = []
    for values in rows.values():
        cols += [c for c in values if c not in cols]
    lines = [f"| {label} | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, values in rows.items():
        cells = [f"{values[c]:.3f}" if c in values else "—" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def probe_table(rows: Mapping[str, Mapping[str, Mapping[str, float]]]) -> str:
    """Render ``{model: {probe: result}}`` as a markdown table of R^2_joint."""
    return metrics_table(
        {
            m: {p: r["r2_joint"] for p, r in res.items() if "r2_joint" in r}
            for m, res in rows.items()
        }
    )


@dataclass(frozen=True)
class Panel:
    """One small-multiple panel of a training-curve figure.

    Attributes:
        key: History column to plot.
        title: Panel title (say which direction is better).
        baseline: Optional history column drawn as a dashed reference line.
        baseline_label: Text for the reference line.
        log: Log-scale y axis.
    """

    key: str
    title: str
    baseline: str | None = None
    baseline_label: str = "baseline"
    log: bool = False


def plot_curves(
    history: Sequence[Mapping[str, float]],
    path: Path,
    title: str,
    panels: Sequence[Panel],
) -> None:
    """Plot one panel per metric (small multiples, never a dual axis)."""
    style.apply()
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(4 * len(panels), 3), squeeze=False
    )
    for ax, panel in zip(axes[0], panels, strict=True):
        values = [h[panel.key] for h in history]
        ax.plot(epochs, values, color=style.SERIES[0], marker="o")
        ax.annotate(
            f"{values[-1]:.3f}",
            (epochs[-1], values[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            color=style.TEXT,
            fontsize=8,
        )
        if panel.baseline and panel.baseline in history[0]:
            ref = history[0][panel.baseline]
            ax.axhline(ref, color=style.BASELINE, linestyle="--", linewidth=1.2)
            # Label at the left end: curves usually start far from the baseline.
            ax.annotate(
                f"{panel.baseline_label} {ref:.3f}",
                (epochs[0], ref),
                textcoords="offset points",
                xytext=(0, 4),
                ha="left",
                color=style.TEXT_MUTED,
                fontsize=8,
            )
        ax.set(title=panel.title, xlabel="epoch")
        if panel.log:
            style.log_axis(ax)
        ax.set_xticks(epochs)
    fig.suptitle(title, color=style.TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
