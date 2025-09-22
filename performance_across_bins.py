import json

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kruskal

import utils


def main():
    # Load the results
    df = pd.read_csv('results.csv')

    participants: Dict[str, Dict[str, int | bool | str]
                       ] = json.load(open('resources/participants.json'))
    task_metric_map: Dict[str, List[str]] = json.load(
        open('resources/task_metric_map.json'))
    tasks = json.load(open('resources/tasks.json'))

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
    groups_duration = [g["duration_seconds"].values for _, g in df_all2.groupby("TRE_bin")]
    groups_error = [g["task_error"].values for _, g in df_all2.groupby("TRE_bin")]

    kw_duration = kruskal(*groups_duration)
    kw_error = kruskal(*groups_error)

    print(summary_exp)
    print(f"Kruskal-Wallis test for duration: H={kw_duration.statistic}, p={kw_duration.pvalue}")
    print(f"Kruskal-Wallis test for error: H={kw_error.statistic}, p={kw_error.pvalue}")

    """
    “There is a significant association between TRE category and both task duration and error: radiologists perform faster and more accurately when TRE is lower.”

                        duration_seconds                  task_error                
                                mean        std count       mean       std count
    TRE_bin                                                                     
    good (<5)              21.123900  13.710998   150   2.977284  2.999141   150
    moderate (5-10)        27.410413  23.003325    46   3.932401  4.922826    46
    poor (>10)             41.255522  32.946191    92   6.827359  8.014651    92
    Kruskal-Wallis test for duration: H=40.42498295262341, p=1.6665812303044107e-09
    Kruskal-Wallis test for error: H=23.37397925964933, p=8.402429818600772e-06
    """


    x = 0


if __name__ == "__main__":
    main()
