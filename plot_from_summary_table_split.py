from dataclasses import dataclass
import json
from pathlib import Path
import re

from typing import Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RIGID_COLOR = "#D68156"
DEFORM_COLOR = "#4292C0"

RIG_DEF_SPACING = 0.2

SIG_DIST = 2.0

F_SIZE_TITLES = 20
F_SIZE_LABELS = 14
F_SIZE_TICKS = 14
F_SIZE_LEGEND = 14
F_SIZE_ASTERISKS = 18

SIZE_WHISKERS = 4

XErr = Union[float, Tuple[float, float]]

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


@dataclass(frozen=True)
class CellValue:
    mean: float
    sd: Optional[float]
    is_percent: bool


def parse_cell_value(cell: str) -> CellValue:
    """Parse a table cell like '6.80 ± 8.13' or '83.3% ± 37.5%' or '67.2%'."""
    s = str(cell).strip()
    is_percent = "%" in s
    s = s.replace("%", "").replace("±", "+/-")
    s = re.sub(r"\s+", " ", s)

    pm = re.search(r"([+-]?\d+(?:\.\d+)?)\s*\+/-\s*([+-]?\d+(?:\.\d+)?)", s)
    if pm:
        return CellValue(mean=float(pm.group(1)), sd=float(pm.group(2)), is_percent=is_percent)

    single = re.search(r"([+-]?\d+(?:\.\d+)?)", s)
    if single:
        return CellValue(mean=float(single.group(1)), sd=None, is_percent=is_percent)

    raise ValueError(f"Could not parse cell: {cell!r}")


def metric_direction(metric: str) -> str:
    """Return 'down' if lower is better, 'up' if higher is better."""
    m = metric.lower()
    if "matching error" in m or "duration" in m or "workflow load" in m:
        return "down"
    if "localization rate" in m or "detection rate" in m:
        return "up"
    raise ValueError(f"Unknown metric direction for: {metric!r}")


def propagated_sd(sd_a: Optional[float], sd_b: Optional[float]) -> Optional[float]:
    """Compute sqrt(sd_a^2 + sd_b^2) if both SDs exist."""
    if sd_a is None or sd_b is None:
        return None
    return float(np.sqrt(sd_a**2 + sd_b**2))


def add_extra_left_column(
    ax: plt.Axes,
    y: np.ndarray,
    labels: Sequence[str],
    x: float = -0.15,
) -> None:
    """Add a text column left of the y-axis, aligned with existing y-ticks."""
    if len(labels) != len(y):
        raise ValueError("labels must match y")

    def task_color(label: str) -> str:

        return "#000000"

        if "Lymph" in label:
            return "#c2910b"
        if "Recurrence" in label:
            return "#be4f23"
        return "#09817b"

    for yi, lab in zip(y, labels):
        lab_s = str(lab)
        ax.text(
            x,
            float(yi),
            lab_s,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            clip_on=False,
            fontweight="bold",
            color=task_color(lab_s) if lab_s else "black",
            fontsize=F_SIZE_LABELS,
        )


def grouped_y(tasks: Sequence[str], gap: float = 0.8) -> np.ndarray:
    """Create y positions with gaps between task groups (works for any subset)."""
    y: list[float] = []
    offset = 0.0
    prev: Optional[str] = None
    for i, t in enumerate(tasks):
        if i > 0 and t != prev:
            offset += gap
        y.append(i + offset)
        prev = t
    return np.array(y, dtype=float)


def plot_significance_at_end_of_errorbars(
    rigid_imp: Sequence[float],
    deform_imp: Sequence[float],
    rigid_imp_sd: Sequence[Optional[float]],
    deform_imp_sd: Sequence[Optional[float]],
    y: np.ndarray,
    ax: plt.Axes,
    rigid_sig: Sequence[int],
    deform_sig: Sequence[int],
    dx_points: float = 6.0,
) -> None:
    """Place significance asterisks with constant screen-space x-offset (points)."""
    for i in range(len(y)):
        if rigid_sig[i] == 1:
            x_end = float(rigid_imp[i]) + float(rigid_imp_sd[i] or 0.0)
            ax.annotate(
                "*",
                xy=(x_end, float(y[i] - RIG_DEF_SPACING)),
                xytext=(dx_points, 0.0),
                textcoords="offset points",
                ha="left",
                va="center",
                color="black",
                fontsize=F_SIZE_ASTERISKS,
                clip_on=False,
            )

        if deform_sig[i] == 1:
            x_end = float(deform_imp[i]) + float(deform_imp_sd[i] or 0.0)
            ax.annotate(
                "*",
                xy=(x_end, float(y[i] + RIG_DEF_SPACING)),
                xytext=(dx_points, 0.0),
                textcoords="offset points",
                ha="left",
                va="center",
                color="black",
                fontsize=F_SIZE_ASTERISKS,
                clip_on=False,
            )


def add_gaps_to_y() -> np.ndarray:
    # import numpy as np

    n = 18
    group_size = 3
    gap = 0.8

    y = []
    offset = 0.0
    for i in range(n):
        if i > 0 and i % group_size == 0:
            offset += gap
        y.append(i + offset)

    y = np.array(y)

    return y


def plot_errorbars_with_colour_significance(rigid_imp, deform_imp, rigid_imp_sd, deform_imp_sd, y, ax, rigid_sig, deform_sig) -> None:
    rigid_sig_mask = rigid_sig == 1
    rigid_nsig_mask = ~rigid_sig_mask
    deform_sig_mask = deform_sig == 1
    deform_nsig_mask = ~deform_sig_mask

    # ---- RIGID (blue): filled = significant ----
    ax.errorbar(
        np.array(rigid_imp, dtype=float)[rigid_sig_mask],
        (y - RIG_DEF_SPACING)[rigid_sig_mask],
        xerr=xerr_array(rigid_imp_sd)[rigid_sig_mask] if xerr_array(
            rigid_imp_sd) is not None else None,
        fmt="o",
        capsize=2,
        markerfacecolor=RIGID_COLOR,
        markeredgecolor=RIGID_COLOR,
        ecolor=RIGID_COLOR,
        label="Rigid vs None",
    )
    ax.errorbar(
        np.array(rigid_imp, dtype=float)[rigid_nsig_mask],
        (y - RIG_DEF_SPACING)[rigid_nsig_mask],
        xerr=xerr_array(rigid_imp_sd)[rigid_nsig_mask] if xerr_array(
            rigid_imp_sd) is not None else None,
        fmt="o",
        capsize=2,
        markerfacecolor="white",
        markeredgecolor=RIGID_COLOR,
        ecolor=RIGID_COLOR,
    )

    # ---- DEFORMABLE (orange): filled = significant ----
    ax.errorbar(
        np.array(deform_imp, dtype=float)[deform_sig_mask],
        (y + RIG_DEF_SPACING)[deform_sig_mask],
        xerr=xerr_array(deform_imp_sd)[deform_sig_mask] if xerr_array(
            deform_imp_sd) is not None else None,
        fmt="o",
        capsize=2,
        markerfacecolor=DEFORM_COLOR,
        markeredgecolor=DEFORM_COLOR,
        ecolor=DEFORM_COLOR,
        label="Deformable vs None",
    )
    ax.errorbar(
        np.array(deform_imp, dtype=float)[deform_nsig_mask],
        (y + RIG_DEF_SPACING)[deform_nsig_mask],
        xerr=xerr_array(deform_imp_sd)[deform_nsig_mask] if xerr_array(
            deform_imp_sd) is not None else None,
        fmt="o",
        capsize=2,
        markerfacecolor="white",
        markeredgecolor=DEFORM_COLOR,
        ecolor=DEFORM_COLOR,
    )


def xerr_array(xerr: Sequence[Optional[XErr]]) -> Optional[np.ndarray]:
    """Convert per-point symmetric/asymmetric xerr into matplotlib-compatible array."""
    if all(v is None for v in xerr):
        return None

    left: list[float] = []
    right: list[float] = []

    for v in xerr:
        if v is None:
            left.append(np.nan)
            right.append(np.nan)
        elif isinstance(v, tuple):
            l, r = v
            left.append(float(l)*100)
            right.append(float(r)*100)
        else:
            f = float(v)
            left.append(f)
            right.append(f)

    return np.array([left, right], dtype=float)


def plot_normal_errorbars(rigid_imp, deform_imp, rigid_imp_sd, deform_imp_sd, y, ax) -> None:
    """Dont differentiate based on colour"""
    # ---- RIGID ----

    ax.errorbar(
        np.array(rigid_imp, dtype=float),
        (y - RIG_DEF_SPACING),
        xerr=xerr_array(rigid_imp_sd) if xerr_array(
            rigid_imp_sd) is not None else None,
        fmt="o",
        markersize=8,
        markeredgewidth=1.8,
        elinewidth=1.8,
        capsize=SIZE_WHISKERS,
        capthick=1.8,
        markerfacecolor="white",
        markeredgecolor=RIGID_COLOR,
        ecolor=RIGID_COLOR,
        label="Rigid vs None",
    )

    # ---- DEFORMABLE ----
    ax.errorbar(
        np.array(deform_imp, dtype=float),
        (y + RIG_DEF_SPACING),
        xerr=xerr_array(deform_imp_sd) if xerr_array(
            deform_imp_sd) is not None else None,
        fmt="o",
        markersize=8,
        markeredgewidth=1.8,
        elinewidth=1.8,
        capsize=SIZE_WHISKERS,
        capthick=1.8,
        markerfacecolor="white",
        markeredgecolor=DEFORM_COLOR,
        ecolor=DEFORM_COLOR,
        label="Deformable vs None",
    )


def make_option1_delta_plot(csv_path: Path, out_path: Path) -> None:
    """Create 3-panel Δ-from-None dot plot: Accuracy | Duration | Workflow load."""
    df = pd.read_csv(csv_path)

    tasks_all = df["Task"].astype(str).tolist()
    metrics_all = df["Metric"].astype(str).tolist()

    none_vals = [parse_cell_value(v) for v in df["None"].tolist()]
    rigid_vals = [parse_cell_value(v) for v in df["Rigid"].tolist()]
    deform_vals = [parse_cell_value(v) for v in df["Deformable"].tolist()]

    # CI for localization of lymph node and detection of recurrence
    with open('outputs/new/diff_plot_data.json', 'r') as f:
        ci_data = json.load(f)

    dirs = [metric_direction(m) for m in metrics_all]

    def improvement(n: float, x: float, d: str) -> float:
        return (n - x) if d == "down" else (x - n)

    rigid_imp_all = [improvement(n.mean, r.mean, d)
                     for n, r, d in zip(none_vals, rigid_vals, dirs)]
    deform_imp_all = [improvement(n.mean, z.mean, d)
                      for n, z, d in zip(none_vals, deform_vals, dirs)]

    rigid_imp_sd_all = [propagated_sd(n.sd, r.sd)
                        for n, r in zip(none_vals, rigid_vals)]
    deform_imp_sd_all = [propagated_sd(n.sd, z.sd)
                         for n, z in zip(none_vals, deform_vals)]

    # replace with CI values (12,15)
    rigid_imp_sd_all[12] = (ci_data['lymph_node']['linear']
                            ['error'][1], ci_data['lymph_node']['linear']['error'][0])
    deform_imp_sd_all[12] = (ci_data['lymph_node']['nonlinear']
                             ['error'][1],  ci_data['lymph_node']['nonlinear']['error'][0])
    rigid_imp_sd_all[15] = (ci_data['recurrence']['linear']
                            ['error'][1],  ci_data['recurrence']['linear']['error'][0])
    deform_imp_sd_all[15] = (ci_data['recurrence']['nonlinear']
                             ['error'][1],  ci_data['recurrence']['nonlinear']['error'][0])

    rigid_sig_all = np.array(
        [
            0, 0, 0,
            0, 0, 1,
            0, 0, 0,
            0, 0, 0,
            0, 1, 0,
            0, 0, 0
        ],
        dtype=int,
    )
    deform_sig_all = np.array(
        [
            0, 1, 1,
            1, 1, 0,
            0, 1, 0,
            1, 1, 0,
            0, 1, 0,
            0, 0, 0,
        ],
        dtype=int,
    )

    def panel_kind(metric: str) -> str:
        m = metric.lower()
        if "duration" in m:
            return "duration"
        if "workflow load" in m:
            return "workflow"
        return "accuracy"

    panel_defs: list[tuple[str, str]] = [
        ("accuracy", "Accuracy"),
        ("duration", "Duration"),
        ("workflow", "Workflow load"),
    ]

    max_rows = max(sum(panel_kind(m) == key for m in metrics_all)
                   for key, _ in panel_defs)
    fig_h = max(5.0, 0.35 * max_rows)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, fig_h))

    for ax, (key, title) in zip(axes, panel_defs):
        idx = [i for i, m in enumerate(metrics_all) if panel_kind(m) == key]

        tasks = [tasks_all[i] for i in idx]
        metrics = [metrics_all[i] for i in idx]

        rigid_imp = [rigid_imp_all[i] for i in idx]
        deform_imp = [deform_imp_all[i] for i in idx]
        rigid_imp_sd = [rigid_imp_sd_all[i] for i in idx]
        deform_imp_sd = [deform_imp_sd_all[i] for i in idx]

        rigid_sig = rigid_sig_all[idx]
        deform_sig = deform_sig_all[idx]

        y = grouped_y(tasks)

        ax.axvline(0.0, linewidth=1.0, color="gray", linestyle="--")
        plot_normal_errorbars(rigid_imp, deform_imp,
                              rigid_imp_sd, deform_imp_sd, y, ax)
        plot_significance_at_end_of_errorbars(
            rigid_imp, deform_imp, rigid_imp_sd, deform_imp_sd, y, ax, rigid_sig, deform_sig
        )

        ax.set_title(title, fontsize=F_SIZE_TITLES)
        ax.set_yticks(y)
        ax.set_yticklabels(["" for _ in y])  # hide y-tick labels
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=F_SIZE_TICKS)
        ax.invert_yaxis()

    labels_y = []
    prev: Optional[str] = None
    tasks_acc = [tasks_all[i] for i, m in enumerate(
        metrics_all) if panel_kind(m) == "accuracy"]
    y_acc = grouped_y(tasks_acc)
    for t in tasks_acc:
        labels_y.append(t if t != prev else "")
        prev = t

    add_extra_left_column(axes[0], y_acc, labels_y, x=-0.05)

    for ax in axes:
        ax.legend_.remove() if ax.get_legend() is not None else None

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.55, -0.08),
        fontsize=F_SIZE_LEGEND,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    # fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    make_option1_delta_plot(
        Path("overview_table_from_latex_to_csv.csv"), Path("option1_delta_dotplot_split.png"))
    print("Wrote: option1_delta_dotplot.png")
