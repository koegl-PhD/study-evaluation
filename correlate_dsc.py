import json

from typing import Dict, List, Literal, Tuple, Callable

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from dcor import distance_correlation
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pointbiserialr, mannwhitneyu

import all_evaluations
import analysis_functions as af
import utils


def cor_continuous(df: pd.DataFrame, cor_from: str, cor_to: str) -> Dict[str, float]:
    pearson_corr, pearson_p = pearsonr(df[cor_from], df[cor_to])
    spearman_corr, spearman_p = spearmanr(df[cor_from], df[cor_to])

    dist_corr = distance_correlation(df[cor_from], df[cor_to])

    return {
        "pearson_corr": float(pearson_corr),
        "pearson_p": float(pearson_p),
        "spearman_corr": float(spearman_corr),
        "spearman_p": float(spearman_p),
        "distance_corr": float(dist_corr)
    }


def cor_categorical(df_ori: pd.DataFrame, cor_from: str, cor_to: str, n_bins: int = 4) -> Dict[str, float]:

    df = df_ori.copy()

    # ensure boolean -> int
    df[cor_to] = df[cor_to].astype(int)

    # Point-biserial correlation
    r_pb, p_pb = pointbiserialr(df[cor_to], df[cor_from])

    # Mann-Whitney U test
    group0 = df.loc[df[cor_to] == 0, cor_from]
    group1 = df.loc[df[cor_to] == 1, cor_from]
    u_stat, p_u = mannwhitneyu(group0, group1, alternative="two-sided")

    mean_diff = group1.mean() - group0.mean()
    pooled_sd = np.sqrt(((group0.std()**2 + group1.std()**2) / 2))
    cohens_d = mean_diff / pooled_sd

    return {
        "pointbiserial_r": float(r_pb),
        "pointbiserial_p": float(p_pb),
        "mannwhitneyu_u": float(u_stat / (len(group0)*len(group1))),
        "mannwhitneyu_p": float(p_u),
        "cohens_d": float(cohens_d)
    }


def main():
    # Load the results
    df = pd.read_csv('outputs/results.csv')

    df = utils.remove_calibration(df)

    df = df[df["transform_type"] !=
            "TransformType.NONE"].reset_index(drop=True)

    # remove rows where synchronised_duration_seconds is 0.0
    df = df[df["synchronised_duration_seconds"] != 0.0].reset_index(drop=True)

    combinations: Dict[str, Dict[str, Callable[[pd.DataFrame, str, str], Dict[str, float]] | List[str]] | None] = {
        "tre+duration_seconds": {"func": cor_continuous, "exclude_tasks": ["recurrence"]},
        "tre+bifurcation_error": {"func": cor_continuous, "exclude_tasks": ["lymph_node", "recurrence"]},
        "tre+abs": {"func": cor_categorical, "exclude_tasks": ["recurrence"]},
        "tre+recurrence_abs": None,

        "dsc+duration_seconds": {"func": cor_continuous, "exclude_tasks": []},
        "dsc+bifurcation_error": {"func": cor_continuous, "exclude_tasks": ["lymph_node", "recurrence"]},
        "dsc+abs": {"func": cor_categorical, "exclude_tasks": ["recurrence"]},
        "dsc+recurrence_abs": {"func": cor_categorical, "exclude_tasks": ["a_vertebralis_r", "a_vertebralis_l", "a_carotisexterna_r", "a_carotisexterna_l", "lymph_node"]},
    }

    task_subset = [
        # "a_vertebralis_r",
        # "a_vertebralis_l",
        # "a_carotisexterna_r",
        # "a_carotisexterna_l",
        # "lymph_node",
        "recurrence"
    ]
    df = df[df["task_id"].isin(task_subset)].reset_index(drop=True)

    print()
    print()
    for comb_k, comb_v in combinations.items():

        if comb_v is None:
            continue

        cor_from, cor_to = comb_k.split('+')
        func = comb_v["func"]
        exclude = comb_v["exclude_tasks"]

        if func != cor_categorical:
            # pass
            continue

        df_current = df[~df["task_id"].isin(exclude)].reset_index(drop=True)
        if len(df_current) == 0:
            print(
                f"Skipping correlation between {cor_from} and {cor_to} as there is no data after excluding tasks {exclude}.\n\n")
            continue

        cor_results = func(df_current, cor_from, cor_to)

        print(
            f"Correlation between {cor_from} and {cor_to} for tasks:{task_subset}\n {json.dumps(cor_results, indent=2)}\n\n")

    x = 0


if __name__ == "__main__":
    main()
