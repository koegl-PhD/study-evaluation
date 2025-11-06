import json

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import kruskal
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro, normaltest, levene, bartlett

import utils


def main():
    # Load the results
    df = pd.read_csv('outputs/results.csv')

    participants: Dict[str, Dict[str, int | bool | str]
                       ] = json.load(open('resources/participants.json'))

    df = utils.remove_calibration(df)

    df = df[df['user_id'].isin([uid for uid, info in participants.items() if info['experienced'] == False])].reset_index(drop=True)  # nopep8
    # keep only where transform_type is not NONE
    df = df[df["transform_type"] !=
            "TransformType.NONE"].reset_index(drop=True)
    # df = df[~((df['transform_type'] != 'TransformType.NONE') & (df['synchronised_count'] == 0))].reset_index(drop=True)  # nopep8

    rel_cols = [c for c in df.columns if c.endswith("_rel")]
    task_to_errorcol = {c.replace("_rel", ""): c for c in rel_cols}

    dfs = []
    for task_id, df_task in df.groupby("task_id"):

        if not task_id.startswith('a_'):
            continue

        tmp = df_task[["task_index", "duration_seconds",
                       "tre", 'bifurcation_error', "transform_type"]].copy()
        tmp["task_id"] = task_id
        dfs.append(tmp)

    df_all2 = pd.concat(dfs).dropna()

    # Define TRE bins again
    bins = [0, 5, 10, np.inf]
    labels = ["good (<5)", "moderate (5-10)", "poor (>10)"]
    df_all2["TRE_bin"] = pd.cut(
        df_all2["tre"], bins=bins, labels=labels, right=True)

    # for col in ["duration_seconds", "bifurcation_error"]:
    #     print(f"\nNormality check for {col}:")
    #     for name, group in df_all2.groupby("TRE_bin"):
    #         if len(group) >= 3:  # Shapiro requires at least 3 values
    #             stat, p = shapiro(group[col])
    #             print(f"  {name}: p={p:.3f}")

    groups_dur = [g["duration_seconds"].values for _,
                  g in df_all2.groupby("TRE_bin") if len(g) > 0]
    groups_err = [g["bifurcation_error"].values for _,
                  g in df_all2.groupby("TRE_bin") if len(g) > 0]

    stat_lev_dur, p_lev_dur = levene(*groups_dur)
    stat_lev_err, p_lev_err = levene(*groups_err)
    # print(f"\nLevene duration p={p_lev_dur:.3f}")
    # print(f"Levene error p={p_lev_err:.3f}")

    # Compute mean duration and error per bin for experienced only
    summary_exp = df_all2.groupby(
        "TRE_bin")[["duration_seconds", "bifurcation_error"]].agg(["mean", "std", "count"])

    # Kruskal-Wallis test across TRE bins for experienced radiologists
    groups_duration = [g["duration_seconds"].values for _,
                       g in df_all2.groupby("TRE_bin")]
    groups_error = [g["bifurcation_error"].values for _,
                    g in df_all2.groupby("TRE_bin")]

    # drop empty groups if any
    groups_duration = [g for g in groups_duration if len(g) > 0]
    groups_error = [g for g in groups_error if len(g) > 0]

    kw_duration = kruskal(*groups_duration)
    kw_error = kruskal(*groups_error)

    N_dur, k_dur = sum(len(g) for g in groups_duration), len(groups_duration)
    N_err, k_err = sum(len(g) for g in groups_error), len(groups_error)

    eps2_dur = utils.epsilon_squared_kw(
        float(kw_duration.statistic), N_dur, k_dur)
    eps2_err = utils.epsilon_squared_kw(
        float(kw_error.statistic), N_err, k_err)

    print(summary_exp)
    print(
        f"Kruskal-Wallis test for duration: H={kw_duration.statistic}, p={kw_duration.pvalue}, ε²={eps2_dur:.3f}")
    print(
        f"Kruskal-Wallis test for error: H={kw_error.statistic}, p={kw_error.pvalue}, ε²={eps2_err:.3f}")

    dunn_dur = sp.posthoc_dunn(df_all2, val_col="duration_seconds",
                               group_col="TRE_bin", p_adjust="holm").reindex(index=labels, columns=labels)
    dunn_err = sp.posthoc_dunn(df_all2, val_col="bifurcation_error", group_col="TRE_bin",
                               p_adjust="holm").reindex(index=labels, columns=labels)

    print()
    print()
    print()
    print()
    print()
    print("Dunn–Holm adjusted p-values (Duration):")
    print(dunn_dur)
    print()
    print()
    print("Dunn–Holm adjusted p-values (Error):")
    print(dunn_err)

    print()
    print()
    print()
    print()
    print()
    print()
    print()
    # ----- Piecewise fixed-effects models (robust alternative to MixedLM) -----
    candidates = np.linspace(2, 40, 39)

    res_dur = utils.fit_piecewise_fe_grid(
        df=df_all2, y_col="duration_seconds", tre_col="tre",
        group_col="task_id", candidates_mm=candidates,
    )
    res_err = utils.fit_piecewise_fe_grid(
        df=df_all2, y_col="bifurcation_error", tre_col="tre",
        group_col="task_id", candidates_mm=candidates,
    )

    print("\nPiecewise FE model — Duration ~ TRE:")
    if res_dur["ok"]:
        print(
            f"breakpoint ≈ {res_dur['breakpoint_mm']:.1f} mm | "
            f"pre-slope = {res_dur['pre_slope_per_mm']:.3f} s/mm | "
            f"post-slope = {res_dur['post_slope_per_mm']:.3f} s/mm | "
            f"AIC_pw = {res_dur['aic_piecewise']:.1f} | AIC_lin = {res_dur['aic_linear']:.1f} | "
            f"ΔAIC = {res_dur['delta_aic']:.1f}"
        )
    else:
        print("fit failed")

    print("Piecewise FE model — Error ~ TRE:")
    if res_err["ok"]:
        print(
            f"breakpoint ≈ {res_err['breakpoint_mm']:.1f} mm | "
            f"pre-slope = {res_err['pre_slope_per_mm']:.3f} mm/mm | "
            f"post-slope = {res_err['post_slope_per_mm']:.3f} mm/mm | "
            f"AIC_pw = {res_err['aic_piecewise']:.1f} | AIC_lin = {res_err['aic_linear']:.1f} | "
            f"ΔAIC = {res_err['delta_aic']:.1f}"
        )
    else:
        print("fit failed")

    x = 0
    """

    Because all groups deviate significantly from normality, Kruskal–Wallis followed by Dunn–Holm post-hoc comparisons is the correct and statistically justified choice.

                    duration_seconds                  bifurcation_error                
                                mean        std count       mean       std count
    TRE_bin                                                                     
    good (<5)              21.123900  13.710998   150   2.977284  2.999141   150
    moderate (5-10)        27.410413  23.003325    46   3.932401  4.922826    46
    poor (>10)             41.255522  32.946191    92   6.827359  8.014651    92
    Kruskal-Wallis test for duration: H=40.42498295262341, p=1.6665812303044107e-09, ε²=0.135
    Kruskal-Wallis test for error: H=23.37397925964933, p=8.402429818600772e-06, ε²=0.075

    Dunn–Holm adjusted p-values (Duration):
                        good (<5)  moderate (5-10)    poor (>10)
    good (<5)        1.000000e+00         0.145264  6.826286e-10
    moderate (5-10)  1.452642e-01         1.000000  1.996182e-03
    poor (>10)       6.826286e-10         0.001996  1.000000e+00

    Dunn–Holm adjusted p-values (Error):
                    good (<5)  moderate (5-10)  poor (>10)
    good (<5)         1.000000         0.222688    0.000004
    moderate (5-10)   0.222688         1.000000    0.032508
    poor (>10)        0.000004         0.032508    1.000000

    Piecewise FE model — Duration ~ TRE:
    breakpoint ≈ 18.0 mm | pre-slope = 1.502 s/mm | post-slope = 1.244 s/mm | AIC_pw = 2618.3 | AIC_lin = 2638.9 | ΔAIC = 20.6
    Piecewise FE model — Error ~ TRE:
    breakpoint ≈ 26.0 mm | pre-slope = 0.195 mm/mm | post-slope = 0.156 mm/mm | AIC_pw = 1784.8 | AIC_lin = 1790.4 | ΔAIC = 5.6


    RESULTS
    To assess whether registration quality (as measured by target registration error, TRE) influenced radiologist performance, we categorized tasks into three TRE bins: good (<5 mm), moderate (5–10 mm), and poor (>10 mm). This binning reflects clinically interpretable accuracy thresholds commonly used in registration evaluation.
    Task duration and landmark error were compared across bins using the Kruskal–Wallis test with Dunn’s post-hoc tests (Holm correction), and effect sizes were quantified with ε².
    Radiologists were significantly faster in the <5 mm bin (mean = 21 s) compared to the >10 mm bin (mean = 41 s, p < 1×10⁻⁹, ε² = 0.135, large effect). Similarly, landmark error was significantly lower in the <5 mm bin (mean = 3.0 mm) compared to the >10 mm bin (mean = 6.8 mm, p = 0.000004, ε² = 0.075, medium effect). By contrast, differences between <5 mm and 5–10 mm were not significant for either duration or error, suggesting that radiologists tolerated TRE up to ~10 mm without measurable degradation.
    To confirm that this effect was nonlinear rather than gradual, we fitted piecewise fixed-effects regression models with task as a covariate. These revealed thresholds at ~18 mm for duration (ΔAIC = 20.6 vs linear) and ~26 mm for error (ΔAIC = 5.6 vs linear). Both models fit better than linear alternatives, supporting the interpretation that radiologist performance remained stable up to a threshold and then declined.
    Spearman correlations between TRE and performance metrics showed moderate positive associations with task duration for some tasks (ρ = 0.43–0.45, p < 0.001) but generally weaker or absent associations with error (Table X). These weaker correlations reflect the nonlinear threshold effect identified in our binning and piecewise analyses: radiologist performance remains stable for TRE <10 mm, diluting monotonic associations across the full TRE range.
    
    📌 Discussion
    Our analyses consistently indicate that radiologist performance does not degrade linearly with TRE but instead follows a threshold pattern. Performance remained stable up to ~10 mm TRE and deteriorated markedly above this level. The categorical binning analysis provides strong evidence for this ~10 mm threshold, with large effect sizes for duration and moderate effects for error.
    Piecewise regression models suggested somewhat higher thresholds (~18 mm for duration and ~26 mm for error). However, these estimates were driven by very few datapoints in the extreme tail of the TRE distribution (≤5 cases above 26 mm, ≤13 above 18 mm), making them less robust. We therefore consider the ~10 mm threshold identified in the binning analysis to be the more clinically meaningful cut-off, while the piecewise models strengthen the conclusion that performance declines nonlinearly rather than gradually.
    Notably, when TRE was included as a continuous predictor in linear mixed-effects models, no significant association with performance was found. This apparent discrepancy is explained by the threshold nature of the effect: linear models fail to capture the plateau–drop-off pattern, whereas categorical and piecewise approaches make it visible.
    While correlations suggested some link between TRE and duration, the weaker associations with error and across tasks highlight that radiologist performance does not degrade gradually with increasing TRE. Instead, our categorical and piecewise analyses demonstrate that performance remains stable until a usability threshold is exceeded.


    “New registration algorithms should prioritize robustness and failure prevention over incremental accuracy improvements below ~10 mm. The critical clinical need is to ensure that registrations remain within a usability threshold (~10 mm TRE), since radiologists’ performance only degrades once this threshold is exceeded. Thus, robustness against difficult cases and prevention of extreme misalignments may yield greater clinical impact than optimizing mean TRE values by a few millimeters.”
    
    # TODO
    make this for lymphnode - problem: we dont have float values for bifurcation_error, only bool for correct/incorrect
    """


if __name__ == "__main__":
    main()
