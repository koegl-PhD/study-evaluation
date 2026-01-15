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
from scipy.stats import kruskal
import utils
from statsmodels.nonparametric.smoothers_lowess import lowess
import itertools


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size between two samples."""
    nx, ny = len(x), len(y)
    rank_sum = sum(int(xi > yi) - int(xi < yi) for xi in x for yi in y)
    return rank_sum / (nx * ny)


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


def remove_y_ticks_smaller_than(g: sns.FacetGrid, threshold: float) -> None:
    for ax in g.axes.flat:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(bottom=y_min, top=71)
        ax.set_yticks([t for t in ax.get_yticks() if t >= 0 and t < 71])


def remove_y_ticks_bigger_than(g: sns.FacetGrid, threshold: float) -> None:
    for ax in g.axes.flat:
        # get current yticks and keep only those < threshold
        yticks = [t for t in ax.get_yticks() if t < threshold]
        ax.set_yticks(yticks)


def annotate_pairs(
    ax: plt.Axes,
    x_positions: Dict[str, float],
    y_top: float,
    pairs: List[Tuple[str, str, float]],
    experience: str,
    effect_sizes: Dict[Tuple[str, str], float],
) -> None:
    """Draw brackets and significance stars for given pairs, with optional effect sizes."""
    h = (y_top * 0.04) if y_top > 0 else 0.2
    cur = y_top + h
    h = 2.7534800754194753+1.7
    if experience == "Experienced":
        m, n, o = pairs
        pairs = [o, n, m]
        cur += 9.6
    else:
        cur -= 20
        pairs.reverse()

    cur -= 7.0

    effect_size = effect_sizes[experience]

    for a, b, p in pairs:
        x1, x2 = x_positions[a], x_positions[b]
        color = "#be6058" if p < 0.05 else "#b4b4b4"
        thickness = 1.5 if p < 0.05 else 1.0
        ax.plot([x1, x1, x2, x2], [cur, cur + h, cur + h, cur],
                linewidth=thickness, color=mpl.colors.to_rgba(color, 1.0))

        # build p-value label
        if p < 0.05:
            s = f"{p:.2e}"
            mant, exp = s.split("e")
            mant = mant.rstrip("0").rstrip(".")
            exp = int(exp)
            label = f"$p={mant}\\times10^{{{exp}}}$"
        else:
            label = f"$p={p:.3f}$"

        # add effect size
        key = (a, b)
        if key not in effect_size and (b, a) in effect_size:
            key = (b, a)
        if key in effect_size:
            δ = effect_size[key]
            label += f"\n$\\delta={δ:.3f}$"

        ax.text((x1 + x2) / 2, cur + h * 1.0, label,
                ha="center", va="bottom", fontsize=10)
        cur += h * 1.7


def plot_tre_robustness(df: pd.DataFrame, out_path: str, frac: float = 0.25, n_boot: int = 200, seed: int = 42) -> None:
    """LOESS robustness curves of matching error vs. TRE with bootstrap 95% CI, faceted by experience."""
    rng = np.random.default_rng(seed)
    work = df.copy()
    work = work[work["transform_type"] != "TransformType.NONE"].dropna(
        subset=["tre", "bifurcation_error", "experience"])
    work = work[(work["tre"] > 0) & np.isfinite(work["tre"])
                & np.isfinite(work["bifurcation_error"])]

    tre_grid = np.linspace(np.nanpercentile(
        work["tre"], 1), np.nanpercentile(work["tre"], 99), 200)

    exp_colors = {
        "Experienced": "#4895c2",
        "Inexperienced": "#d17d53"
    }

    range_vals = {
        "Experienced": [],
        "Inexperienced": []
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (exp, dfe) in zip(axes, work.groupby("experience", sort=False)):
        base = lowess(endog=dfe["bifurcation_error"],
                      exog=dfe["tre"], frac=frac, it=0, return_sorted=True)
        base_x, base_y = base[:, 0], base[:, 1]
        base_interp = np.interp(tre_grid, base_x, base_y,
                                left=np.nan, right=np.nan)

        boots = np.empty((n_boot, tre_grid.size), dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, len(dfe), len(dfe))
            b = dfe.iloc[idx]
            sm = lowess(endog=b["bifurcation_error"],
                        exog=b["tre"], frac=frac, it=0, return_sorted=True)
            bx, by = sm[:, 0], sm[:, 1]
            boots[i] = np.interp(tre_grid, bx, by, left=np.nan, right=np.nan)

        lo = np.nanpercentile(boots, 2.5, axis=0)
        hi = np.nanpercentile(boots, 97.5, axis=0)

        ax.plot(tre_grid, base_interp, linewidth=2,
                color=mpl.colors.to_rgba(exp_colors[exp], 1.0))
        ax.fill_between(tre_grid, lo, hi, alpha=0.3, linewidth=0,
                        color=mpl.colors.to_rgba(exp_colors[exp], 1.0))
        ax.axvline(5.0, linestyle="--", linewidth=1, color="red")
        ax.axvline(10.0, linestyle="--", linewidth=1, color="red")

        ax.set_title("Final-year medical students" if exp ==
                     "Inexperienced" else "Radiology residents")
        ax.set_xlabel("TRE (mm)")
        ax.set_xlim(tre_grid.min(), tre_grid.max())

        last_below_10 = np.searchsorted(tre_grid, 10, side='right') - 1
        minimum_below_10 = np.min(base_interp[:last_below_10 + 1])
        maximum_below_10 = np.max(base_interp[:last_below_10 + 1])

        range_vals[exp] = [minimum_below_10, maximum_below_10]

        ax.hlines(y=minimum_below_10, xmin=0, xmax=10,
                  linestyle=":", color="black", linewidth=1)
        ax.hlines(y=maximum_below_10, xmin=0, xmax=10,
                  linestyle=":", color="black", linewidth=1)

    axes[0].set_ylabel("Matching error (mm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)

    

    print(range_vals)

    x = 0


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
        kind="box",
        data=work,
        x="TRE_bin",
        y="bifurcation_error",
        col="experience",
        # inner=None,
        # cut=0,
        # scale="width",
        sharey=False,
        linewidth=1.0,
    )

    exp_colors = {
        "Experienced": "#86b2cb",
        "Inexperienced": "#f1cbb8"
    }

    effect_sizes = {
        "Experienced": {},
        "Inexperienced": {}
    }

    for ax, (exp, df_exp) in zip(g.axes.flat, work.groupby("experience", sort=False)):
        med = df_exp.groupby("TRE_bin")["bifurcation_error"].median()
        x_ticks = [t.get_text() for t in ax.get_xticklabels()]
        x_pos = {lab: i for i, lab in enumerate(x_ticks)}
        # ax.scatter([x_pos[str(k)] for k in med.index.astype(str)],
        #            med.values, marker="x", s=50, zorder=3)

        for line in ax.lines:
            line.set_color("black")       # whiskers, caps, medians
            line.set_linewidth(1.0)
        for patch in ax.patches:
            patch.set_edgecolor("black")  # main box border

        exp = ax.get_title().split(" = ")[-1]
        for patch in [c for c in ax.patches if isinstance(c, mpl.patches.PathPatch)]:
            patch.set_facecolor(exp_colors[exp])
            patch.set_edgecolor("black")
            patch.set_alpha(1.0)

        groups_error = [*[df_exp.loc[df_exp["TRE_bin"] == b, "bifurcation_error"]
                          for b in df_exp["TRE_bin"].unique()]]
        kw_stat, kw_p = kruskal(
            *[df_exp.loc[df_exp["TRE_bin"] == b, "bifurcation_error"] for b in df_exp["TRE_bin"].unique()]
        )
        N_err, k_err = sum(len(g) for g in groups_error), len(groups_error)
        eps2_err = utils.epsilon_squared_kw(
            float(kw_stat), N_err, k_err)
        print(
            f"Kruskal–Wallis H={kw_stat:.2f}, p={kw_p:.4f}, ε²={eps2_err:.4f} for bifurcation_error across TRE bins")

        p_table = pairwise_mwu_holm(
            df_exp["bifurcation_error"], df_exp["TRE_bin"].astype(str))
        y_top = df_exp["bifurcation_error"].max()

        for (a, b) in itertools.combinations(df_exp["TRE_bin"].unique(), 2):
            x = df_exp.loc[df_exp["TRE_bin"] ==
                           a, "bifurcation_error"].to_numpy()
            y = df_exp.loc[df_exp["TRE_bin"] ==
                           b, "bifurcation_error"].to_numpy()
            U, _ = mannwhitneyu(x, y, alternative="two-sided")

            # rank-biserial correlation (r) from Mann–Whitney U
            n1, n2 = len(x), len(y)
            r_rb = 1 - (2 * U) / (n1 * n2)

            # Cliff's delta
            d = cliffs_delta(x, y)

            print(f"{exp}: {a} vs {b} → r_rb={r_rb:.3f}, δ={d:.3f}")

            # reverse order, because we want to compare fromleft to right
            if a == "poor ($>10mm$)" and b == "moderate ($5-10mm$)":
                d *= -1.0  # reverse direction for this pair

            effect_sizes[exp][(str(a), str(b))] = d

        annotate_pairs(ax, x_pos, float(y_top), p_table,
                       experience=exp, effect_sizes=effect_sizes)
        ax.set_xlabel(" ", labelpad=15)
        ax.set_ylabel("Matching error ($mm$)")

    titles = {
        "Experienced": r"Experienced",
        "Inexperienced": r"Inexperienced"
    }

    for ax in g.axes.flat:
        # find which experience this axis corresponds to
        exp = ax.get_title().split(" = ")[-1]
        if exp in titles:
            ax.set_title(titles[exp], fontsize=14)

    # for violin
    """
    for ax in g.axes.flat:
        exp = ax.get_title().split(" = ")[-1]
        for pc in [c for c in ax.collections if isinstance(c, mpl.collections.PolyCollection)]:
            pc.set_facecolor(mpl.colors.to_rgba(exp_colors[exp], 1.0))
            pc.set_edgecolor("black")
            # pc.set_linewidth(0.5)
    """
    # for boxplot
    for ax in g.axes.flat:
        exp = ax.get_title().split(" = ")[-1]
        for patch in [c for c in ax.patches if isinstance(c, mpl.patches.PathPatch)]:
            patch.set_facecolor(exp_colors[exp])
            patch.set_edgecolor("black")
            patch.set_alpha(1.0)

    for ax, (exp, df_exp) in zip(g.axes.flat, work.groupby("experience", sort=False)):
        grouped = df_exp.groupby("TRE_bin")["bifurcation_error"]
        stats = grouped.agg(['count', 'mean', 'std'])

        offset_adjustment = 7.0
        down_room = 0.13

        y_min, y_max = ax.get_ylim()
        y_offset = y_min + (y_max - y_min) * 0.00 - offset_adjustment

        for i, (label, row) in enumerate(stats.iterrows()):
            txt = (
                f"count$={int(row['count'])}$\n"
                f"${row['mean']:.2f}\\pm{row['std']:.2f}\\,\\mathrm{{mm}}$"
            )
            ax.text(
                i-0.37,
                y_offset,
                txt,
                ha="left", va="bottom", fontsize=12,
                bbox=dict(facecolor="white", edgecolor="gray",
                          boxstyle="round,pad=0.3")
            )

        # extend y-axis slightly downward to make room
        ax.set_ylim(y_min - (y_max - y_min) * down_room, y_max)

    remove_y_ticks_smaller_than(g, 0.0)

    ymin, ymax = g.axes.flat[0].get_ylim()

    for ax in g.axes.flat:
        ax.set_ylim(ymin, ymax+7.0)

    ax = g.axes.flat[1]
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel("")  # label on right side
    ax.set_yticklabels([])  # hide tick labels
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # ax.set_yticks([])       # hide ticks

    ax_left = g.axes.flat[0]
    ax_right = g.axes.flat[1]

    # get y-position of x-axis
    ymin = ax_left.get_ylim()[0]
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["bottom"].set_visible(False)

    # get rightmost x of left subplot and leftmost x of right subplot
    xmax_left = ax_left.get_xlim()[1]
    xmin_left = ax_left.get_xlim()[0]
    xmin_right = ax_right.get_xlim()[0]
    xmax_right = ax_right.get_xlim()[1]

    ymin -= 0.05

    # draw a connecting line between subplots
    g.fig.lines.append(
        plt.Line2D(
            [xmin_left, 2 * xmax_left + 0.64], [ymin, ymin],
            color="black", linewidth=0.8, transform=ax_left.transData, clip_on=False
        )
    )
    ymin += 91.15
    g.fig.lines.append(
        plt.Line2D(
            [xmin_left, 2 * xmax_left + 0.64], [ymin, ymin],
            color="black", linewidth=0.8, transform=ax_left.transData, clip_on=False
        )
    )

    # replace your title-setting loop with this to place titles inside each subplot
    c = 0
    for ax in g.axes.flat:
        exp = ax.get_title().split(" = ")[-1]
        ax.set_title("")  # remove default title

        if c == 0:
            title = "Final-year medical students"
        else:
            title = "Radiology residents"
        ax.text(
            0.5, 0.97, title,  # y < 1.0 puts it inside
            transform=ax.transAxes, ha="center", va="top",
            fontsize=14, fontweight="bold"
        )

        c += 1

    for ax in g.axes.flat:
        ax.xaxis.set_ticks_position("both")
        ax.tick_params(axis="x", top=True, labeltop=False)

    g.fig.tight_layout()

    g.fig.text(
        0.5, -0.03, "TRE bins",  # (x, y) in figure coordinates (0–1)
        ha="center", va="center",
        fontsize=ax_left.yaxis.get_label().get_size()-2,
    )

    g.fig.savefig(out_path, dpi=300, bbox_inches="tight")

    x = 0


def main() -> None:
    """Load data, add experience, and save violin plots."""

    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"""
            \usepackage{helvet}
            \usepackage{sansmath}
            \sansmath
            \renewcommand{\familydefault}{\sfdefault}
        """,
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

    plot_tre_robustness(df_all2, "outputs/fig_tre_robustness_loess.png")

    x = 0


if __name__ == "__main__":
    main()
