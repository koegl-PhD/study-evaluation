import matplotlib as mpl
from typing import Sequence
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RIGID_COLOR = "#D68156"
DEFORM_COLOR = "#4292C0"

RIG_DEF_SPACING = 0.2

SIG_DIST = 2.0

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

    count = 0

    for yi, lab in zip(y, labels):

        if count < 12:
            color = "#09817b"
        elif count < 15:
            color = "#c2910b"
        else:
            color = "#be4f23"

        ax.text(
            x,
            float(yi),
            str(lab),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            clip_on=False,
            fontweight="bold",
            color=color,
        )

        count += 1


def plot_significance_at_end_of_errorbars(rigid_imp, deform_imp, rigid_imp_sd, deform_imp_sd, y, ax, rigid_sig, deform_sig) -> None:
    """at the end of each SD bar on the right plot an asterisk with .text() if significant"""
    for i in range(len(y)):
        # once for rigid and once for deformable
        if rigid_sig[i] == 1:
            ax.text(
                rigid_imp[i] + (rigid_imp_sd[i] if rigid_imp_sd[i]
                                is not None else 0) + SIG_DIST,
                y[i] - RIG_DEF_SPACING,
                "*",
                color="black",
                fontsize=12,
                va="center",
            )
        if deform_sig[i] == 1:
            ax.text(
                deform_imp[i] + (deform_imp_sd[i]
                                 if deform_imp_sd[i] is not None else 0) + SIG_DIST,
                y[i] + RIG_DEF_SPACING,
                "*",
                color="black",
                fontsize=12,
                va="center",
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


def xerr_array(xerr: list[Optional[float]]) -> Optional[np.ndarray]:
    if all(v is None for v in xerr):
        return None
    return np.array([np.nan if v is None else float(v) for v in xerr], dtype=float)


def make_option1_delta_plot(csv_path: Path, out_path: Path) -> None:
    """Create Option 1 Δ-from-None dot plot (right = better) from the CSV."""
    df = pd.read_csv(csv_path)

    tasks = df["Task"].astype(str).tolist()
    metrics = df["Metric"].astype(str).tolist()

    none_vals = [parse_cell_value(v) for v in df["None"].tolist()]
    rigid_vals = [parse_cell_value(v) for v in df["Rigid"].tolist()]
    deform_vals = [parse_cell_value(v) for v in df["Deformable"].tolist()]

    dirs = [metric_direction(m) for m in metrics]

    def improvement(n: float, x: float, d: str) -> float:
        return (n - x) if d == "down" else (x - n)

    rigid_imp = [improvement(n.mean, r.mean, d)
                 for n, r, d in zip(none_vals, rigid_vals, dirs)]
    deform_imp = [improvement(n.mean, z.mean, d)
                  for n, z, d in zip(none_vals, deform_vals, dirs)]

    rigid_imp_sd = [propagated_sd(n.sd, r.sd)
                    for n, r in zip(none_vals, rigid_vals)]
    deform_imp_sd = [propagated_sd(n.sd, z.sd)
                     for n, z in zip(none_vals, deform_vals)]

    labels = [f"{t} — {m}" for t, m in zip(tasks, metrics)]
    y = np.arange(len(labels))
    y = add_gaps_to_y()

    fig_h = max(5.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(10.8, fig_h))

    ax.axvline(0.0, linewidth=1.0)

    # ---- USER-PROVIDED SIGNIFICANCE (0 = not significant, 1 = significant) ----
    rigid_sig = np.array(
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
    deform_sig = np.array(
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

    if len(rigid_sig) != len(y) or len(deform_sig) != len(y):
        raise ValueError("Significance vectors must match number of rows")

    # plot_errorbars_with_colour_significance(
    #     rigid_imp, deform_imp, rigid_imp_sd, deform_imp_sd, y, ax, rigid_sig, deform_sig)
    plot_normal_errorbars(
        rigid_imp, deform_imp, rigid_imp_sd, deform_imp_sd, y, ax)
    plot_significance_at_end_of_errorbars(
        rigid_imp, deform_imp, rigid_imp_sd, deform_imp_sd, y, ax, rigid_sig, deform_sig)

    ax.set_yticks(y)
    ax.set_yticklabels([m.replace('%', '\%') for m in metrics])
    ax.invert_yaxis()
    # ax.set_xlabel("Improvement vs None (signed so right = better)")
    # ax.set_title("Δ-from-None by task and metric")
    ax.legend(loc="upper right")

    # y = np.arange(18)

    labels_y = []
    for i in range(len(metrics)):
        if (i + 2) % 3 == 0:
            labels_y.append(tasks[i])
        else:
            labels_y.append("")

    add_extra_left_column(ax, y, labels_y, x=-0.20)
    fig.subplots_adjust(left=0.45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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


def plot_normal_errorbars(rigid_imp, deform_imp, rigid_imp_sd, deform_imp_sd, y, ax) -> None:
    """Dont differentiate based on colour"""
    # ---- RIGID ----
    ax.errorbar(
        np.array(rigid_imp, dtype=float),
        (y - RIG_DEF_SPACING),
        xerr=xerr_array(rigid_imp_sd) if xerr_array(
            rigid_imp_sd) is not None else None,
        fmt="o",
        capsize=2,
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
        capsize=2,
        markerfacecolor="white",
        markeredgecolor=DEFORM_COLOR,
        ecolor=DEFORM_COLOR,
        label="Deformable vs None",
    )


if __name__ == "__main__":
    make_option1_delta_plot(
        Path("overview_table_from_latex_to_csv.csv"), Path("option1_delta_dotplot.png"))
    print("Wrote: option1_delta_dotplot.png")
