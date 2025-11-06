from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches
import matplotlib as mpl

import utils


def add_experience(df: pd.DataFrame, participants: Dict[str, Dict[str, int | bool | str]]) -> pd.DataFrame:
    """Attach 'experience' (Experienced/Inexperienced) from participants.json."""
    exp_map = {uid: ("Experienced" if info["experienced"] else "Inexperienced")
               for uid, info in participants.items()}
    out = df.copy()
    out["experience"] = out["user_id"].map(exp_map)
    return out.dropna(subset=["experience"])


def pairwise_mwu_holm(y: pd.Series, groups: pd.Series) -> List[Tuple[str, str, float]]:
    """MWU with Holm across all group pairs; returns (g1,g2,p_adj)."""
    levels = [str(l) for l in groups.unique().tolist()]
    pairs = list(combinations(levels, 2))
    p_raw = []
    for a, b in pairs:
        va = y[groups == a].values
        vb = y[groups == b].values
        if len(va) == 0 or len(vb) == 0:
            p_raw.append(np.nan)
        else:
            p_raw.append(mannwhitneyu(va, vb, alternative="two-sided").pvalue)
    _, p_adj, _, _ = multipletests(
        p_raw, method="holm", is_sorted=False, returnsorted=False)
    return [(a, b, float(p)) for (a, b), p in zip(pairs, p_adj)]


def annotate_pairs(ax: plt.Axes, x_positions: Dict[str, float], y_top: float, pairs: List[Tuple[str, str, float]], experience: bool) -> None:
    """Draw brackets and significance stars for given pairs."""
    h = (y_top * 0.04) if y_top > 0 else 0.2
    cur = y_top + h
    h = 2.7534800754194753
    if experience:
        m, n, o = pairs
        pairs = [o, n, m]
        # h += 10
    else:
        pairs.reverse()

    for a, b, p in pairs:
        x1, x2 = x_positions[a], x_positions[b]
        color = "#be6058" if p < 0.05 else "#b4b4b4"
        thickness = 1.5 if p < 0.05 else 1.0
        ax.plot([x1, x1, x2, x2], [cur, cur + h, cur + h, cur],
                linewidth=thickness, color=mpl.colors.to_rgba(color, 1.0))
        label = "ns" if p >= 0.05 else ("*" if p < 0.05 else "")
        if p < 0.01:
            label = "**"
        if p < 0.001:
            label = "***"

        if p < 0.05:
            s = f"{p:.2e}"
            mant, exp = s.split("e")
            mant = mant.rstrip("0").rstrip(".")
            exp = int(exp)
            label = f"$p={mant}\\times10^{{{exp}}}$"

        ax.text((x1 + x2) / 2, cur + h * 1.0, label, ha="center", va="bottom")
        cur += h * 1.7


def plot_tre_violins(df: pd.DataFrame, out_path: str) -> None:
    """Create seaborn violins of matching error by TRE bin, faceted by experience, with medians and MWU-Holm stars."""
    bins = [0, 5, 10, np.inf]
    labels = ["good ($<5mm$)", "moderate ($5-10mm$)", "poor ($>10mm$)"]
    work = df.copy()
    work = work[work["transform_type"] !=
                "TransformType.NONE"].reset_index(drop=True)
    work["TRE_bin"] = pd.cut(work["tre"], bins=bins, labels=labels, right=True)
    work = work.dropna(subset=["TRE_bin", "bifurcation_error", "experience"])

    work["bifurcation_error_viz"] = (
        work.groupby(["experience", "TRE_bin"])["bifurcation_error"]
        .transform(lambda s: s.clip(s.quantile(0.03), s.quantile(0.97)))
    )

    g = sns.catplot(
        kind="violin",
        data=work,
        x="TRE_bin",
        y="bifurcation_error",
        col="experience",
        inner=None,
        # cut=0,
        scale="width",
        sharey=True,
        linewidth=0.5,
    )

    for ax, (exp, df_exp) in zip(g.axes.flat, work.groupby("experience", sort=False)):
        med = df_exp.groupby("TRE_bin")["bifurcation_error"].median()
        x_ticks = [t.get_text() for t in ax.get_xticklabels()]
        x_pos = {lab: i for i, lab in enumerate(x_ticks)}
        # ax.scatter([x_pos[str(k)] for k in med.index.astype(str)],
        #            med.values, marker="x", s=50, zorder=3)

        p_table = pairwise_mwu_holm(
            df_exp["bifurcation_error"], df_exp["TRE_bin"].astype(str))
        y_top = df_exp["bifurcation_error"].max()
        annotate_pairs(ax, x_pos, float(y_top), p_table,
                       experience=exp == "Experienced")
        ax.set_xlabel("TRE bin")
        ax.set_ylabel("Matching error ($mm$)")

        color = "#8f4926" if exp == "Inexperienced" else "#1d4e6d"
        sns.stripplot(
            data=df_exp,
            x="TRE_bin",
            y="bifurcation_error",
            ax=ax,
            color=mpl.colors.to_rgba(color, 1.0),        # black dots
            size=4,           # point size
            jitter=True,      # random horizontal offset
            alpha=0.5,        # transparency
            zorder=2,         # draw above violins
        )

    titles = {
        "Experienced": r"Experienced",
        "Inexperienced": r"Inexperienced"
    }

    for ax in g.axes.flat:
        # find which experience this axis corresponds to
        exp = ax.get_title().split(" = ")[-1]
        if exp in titles:
            ax.set_title(titles[exp], fontsize=13)

    exp_colors = {
        "Experienced": "#86b2cb",
        "Inexperienced": "#f1cbb8"
    }
    for ax in g.axes.flat:
        exp = ax.get_title().split(" = ")[-1]
        for pc in [c for c in ax.collections if isinstance(c, mpl.collections.PolyCollection)]:
            pc.set_facecolor(mpl.colors.to_rgba(exp_colors[exp], 1.0))
            pc.set_edgecolor("black")
            # pc.set_linewidth(0.5)

    for ax, (exp, df_exp) in zip(g.axes.flat, work.groupby("experience", sort=False)):
        grouped = df_exp.groupby("TRE_bin")["bifurcation_error"]
        stats = grouped.agg(['count', 'mean', 'std'])
        for i, (label, row) in enumerate(stats.iterrows()):
            ax.text(
                i, ax.get_ylim()[0] - (ax.get_ylim()[1] *
                                       0.085),  # slightly below axis
                f"count=${int(row['count'])}$\n${row['mean']:.2f}\\pm{row['std']:.2f}mm$",
                ha="center", va="top", fontsize=12
            )

    for ax in g.axes.flat:
        ax.set_xlabel("TRE bin", labelpad=35)

    g.fig.tight_layout()
    g.fig.savefig(out_path, dpi=300)

    x = 0


def main() -> None:
    """Load data, add experience, and save violin plots."""

    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.unicode_minus": False,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12
    })

    df = pd.read_csv('outputs/results.csv')
    participants: Dict[str, Dict[str, int | bool | str]
                       ] = json.load(open('resources/participants.json'))
    df = utils.remove_calibration(df)
    df = add_experience(df, participants)

    rel_cols = [c for c in df.columns if c.endswith("_rel")]
    dfs = []
    for task_id, df_task in df.groupby("task_id"):
        if not task_id.startswith('a_'):
            continue
        tmp = df_task[["task_index", "duration_seconds", "tre",
                       'bifurcation_error', "transform_type", "user_id", "experience"]].copy()
        tmp["task_id"] = task_id
        dfs.append(tmp)
    df_all2 = pd.concat(dfs).dropna()

    plot_tre_violins(df_all2, "outputs/fig_tre_violin_by_experience.png")

    x = 0


if __name__ == "__main__":
    main()
