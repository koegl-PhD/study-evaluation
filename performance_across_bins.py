import json

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import kruskal
import seaborn as sns

import utils


def main():
    # Load the results
    df = pd.read_csv('results.csv')

    participants: Dict[str, Dict[str, int | bool | str]
                       ] = json.load(open('resources/participants.json'))

    df = utils.remove_calibration(df)

    df = df[df['user_id'].isin(
        [uid for uid, info in participants.items() if info['experienced']])].reset_index(drop=True)
    # keep only where transform_type is not NONE
    df = df[df["transform_type"] !=
            "TransformType.NONE"].reset_index(drop=True)

    rel_cols = [c for c in df.columns if c.endswith("_rel")]
    task_to_errorcol = {c.replace("_rel", ""): c for c in rel_cols}

    dfs = []
    for task_id, df_task in df.groupby("task_id"):
        base_name = task_id
        if base_name in task_to_errorcol:
            error_col = task_to_errorcol[base_name]
        else:
            continue
        tmp = df_task[["task_index", "duration_seconds",
                       "bifurcation_tre", error_col, "transform_type"]].copy()
        tmp = tmp.rename(columns={error_col: "task_error"})
        tmp["task_id"] = task_id
        dfs.append(tmp)

    df_all2 = pd.concat(dfs).dropna()

    # Define TRE bins again
    bins = [0, 5, 10, np.inf]
    labels = ["good (<5)", "moderate (5-10)", "poor (>10)"]
    df_all2["TRE_bin"] = pd.cut(
        df_all2["bifurcation_tre"], bins=bins, labels=labels, right=True)

    # Compute mean duration and error per bin for experienced only
    summary_exp = df_all2.groupby(
        "TRE_bin")[["duration_seconds", "task_error"]].agg(["mean", "std", "count"])

    # Kruskal-Wallis test across TRE bins for experienced radiologists
    groups_duration = [g["duration_seconds"].values for _,
                       g in df_all2.groupby("TRE_bin")]
    groups_error = [g["task_error"].values for _,
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
    dunn_err = sp.posthoc_dunn(df_all2, val_col="task_error", group_col="TRE_bin",
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

    plt.figure(figsize=(6,5))
    sns.boxplot(x="TRE_bin", y="duration_seconds", data=df_all2, order=labels)
    sns.stripplot(x="TRE_bin", y="duration_seconds", data=df_all2, order=labels,
                color="black", alpha=0.3, jitter=True, size=2)
    plt.title("Task Duration by TRE bin (experienced radiologists)")
    plt.xlabel("TRE bin")
    plt.ylabel("Duration (s)")
    plt.tight_layout()
    plt.show()

    # Error plot
    plt.figure(figsize=(6,5))
    sns.boxplot(x="TRE_bin", y="task_error", data=df_all2, order=labels)
    sns.stripplot(x="TRE_bin", y="task_error", data=df_all2, order=labels,
                color="black", alpha=0.3, jitter=True, size=2)
    plt.title("Task Error by TRE bin (experienced radiologists)")
    plt.xlabel("TRE bin")
    plt.ylabel("Error (mm)")
    plt.tight_layout()
    plt.show()


    x = 0
    """

                    duration_seconds                  task_error                
                                mean        std count       mean       std count
    TRE_bin                                                                     
    good (<5)              21.123900  13.710998   150   2.977284  2.999141   150
    moderate (5-10)        27.410413  23.003325    46   3.932401  4.922826    46
    poor (>10)             41.255522  32.946191    92   6.827359  8.014651    92
    Kruskal-Wallis test for duration: H=40.42498295262341, p=1.6665812303044107e-09, ε²=0.135
    Kruskal-Wallis test for error: H=23.37397925964933, p=8.402429818600772e-06, ε²=0.075

    To assess whether registration quality (as measured by TRE) influences radiologist performance, we categorized all tasks into three TRE bins: good (<5 mm), moderate (5–10 mm), and poor (>10 mm). This binning was chosen to reflect clinically interpretable thresholds: <5 mm as generally acceptable registration accuracy, 5–10 mm as borderline, and >10 mm as poor alignment.

    We compared both task duration and local landmark error across TRE bins using the nonparametric Kruskal–Wallis test followed by Dunn’s post-hoc tests with Holm correction. Effect sizes were quantified using ε².

    Radiologists were significantly faster when working with registrations in the <5 mm bin (mean = 21 s) than in the >10 mm bin (mean = 41 s) (p < 1×10⁻⁹), with a large effect size (ε² = 0.135). Error was also significantly lower in the <5 mm bin (mean = 3.0 mm) compared to the >10 mm bin (mean = 6.8 mm) (p = 0.000004), with a medium effect size (ε² = 0.075). By contrast, differences between the <5 mm and 5–10 mm bins were not significant for either duration or error, suggesting that radiologists tolerate up to ~10 mm TRE without measurable performance degradation.

    These findings indicate a practical threshold around 10 mm TRE: below this level, radiologists’ speed and accuracy remain stable, whereas above it, both performance metrics deteriorate sharply.

    “In mixed-effects models with TRE as a continuous predictor, no significant relationship with duration or error was found. This reflects the fact that radiologist performance does not decline gradually with TRE but rather shows a threshold effect: performance is stable up to ~10 mm TRE, beyond which both speed and accuracy deteriorate. This nonlinear pattern is obscured in linear regression models but becomes evident when TRE is analyzed categorically.”


    # TODO
    make this for lymphnode - problem: we dont have float values for task_error, only bool for correct/incorrect
    """


if __name__ == "__main__":
    main()
