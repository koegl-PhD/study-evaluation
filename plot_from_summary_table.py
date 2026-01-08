from typing import Sequence
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    x: float = -0.25,
) -> None:
    """Add a text column left of the y-axis, aligned with existing y-ticks."""
    if len(labels) != len(y):
        raise ValueError("labels must match y")

    for yi, lab in zip(y, labels):
        ax.text(
            x,
            float(yi),
            str(lab),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            clip_on=False,
        )


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

    fig_h = max(5.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(10.8, fig_h))

    ax.axvline(0.0, linewidth=1.0)

    def xerr_array(xerr: list[Optional[float]]) -> Optional[np.ndarray]:
        if all(v is None for v in xerr):
            return None
        return np.array([np.nan if v is None else float(v) for v in xerr], dtype=float)

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

    rigid_sig_mask = rigid_sig == 1
    rigid_nsig_mask = ~rigid_sig_mask
    deform_sig_mask = deform_sig == 1
    deform_nsig_mask = ~deform_sig_mask

    # ---- RIGID (blue): filled = significant ----
    ax.errorbar(
        np.array(rigid_imp, dtype=float)[rigid_sig_mask],
        (y - 0.12)[rigid_sig_mask],
        xerr=xerr_array(rigid_imp_sd)[rigid_sig_mask] if xerr_array(
            rigid_imp_sd) is not None else None,
        fmt="o",
        capsize=2,
        markerfacecolor="tab:blue",
        markeredgecolor="tab:blue",
        ecolor="tab:blue",
        label="Rigid vs None",
    )
    ax.errorbar(
        np.array(rigid_imp, dtype=float)[rigid_nsig_mask],
        (y - 0.12)[rigid_nsig_mask],
        xerr=xerr_array(rigid_imp_sd)[rigid_nsig_mask] if xerr_array(
            rigid_imp_sd) is not None else None,
        fmt="o",
        capsize=2,
        markerfacecolor="white",
        markeredgecolor="tab:blue",
        ecolor="tab:blue",
    )

    # ---- DEFORMABLE (orange): filled = significant ----
    ax.errorbar(
        np.array(deform_imp, dtype=float)[deform_sig_mask],
        (y + 0.12)[deform_sig_mask],
        xerr=xerr_array(deform_imp_sd)[deform_sig_mask] if xerr_array(
            deform_imp_sd) is not None else None,
        fmt="o",
        capsize=2,
        markerfacecolor="tab:orange",
        markeredgecolor="tab:orange",
        ecolor="tab:orange",
        label="Deformable vs None",
    )
    ax.errorbar(
        np.array(deform_imp, dtype=float)[deform_nsig_mask],
        (y + 0.12)[deform_nsig_mask],
        xerr=xerr_array(deform_imp_sd)[deform_nsig_mask] if xerr_array(
            deform_imp_sd) is not None else None,
        fmt="o",
        capsize=2,
        markerfacecolor="white",
        markeredgecolor="tab:orange",
        ecolor="tab:orange",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_xlabel("Improvement vs None (signed so right = better)")
    ax.set_title("Δ-from-None by task and metric")
    ax.legend(loc="lower right")

    y = np.arange(18)

    labels_y = []
    for i in range(len(metrics)):
        if (i + 2) % 3 == 0:
            labels_y.append(tasks[i])
        else:
            labels_y.append("")

    add_extra_left_column(ax, y, labels_y, x=-0.30)
    fig.subplots_adjust(left=0.45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_option1_delta_plot(
        Path("table.csv"), Path("option1_delta_dotplot.png"))
    print("Wrote: option1_delta_dotplot.png")
