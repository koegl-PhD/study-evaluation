import json

from typing import Dict, List, Union

import numpy as np
import pandas as pd

import utils


NestedDict = Dict[str, Union[float, "NestedDict"]]


def average_nested_dicts(d1: NestedDict, d2: NestedDict) -> NestedDict:
    """Recursively average values of two nested dicts with same structure."""
    result: NestedDict = {}
    for k in d1:
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            result[k] = average_nested_dicts(v1, v2)
        else:
            result[k] = (v1 + v2) / 2
    return result


def main():
    # Load the results
    df = pd.read_csv('results.csv')

    participants: Dict[str, Dict[str, int | bool | str]
                       ] = json.load(open('resources/participants.json'))
    task_metric_map: Dict[str, List[str]] = json.load(
        open('resources/task_metric_map.json'))
    tasks = json.load(open('resources/tasks.json'))

    df = utils.remove_calibration(df)

    means_all = []
    stds_all = []

    for experienced in [True, False]:

        # keep only (in)experienced radiologists
        df_new = df[df['user_id'].isin(
            [uid for uid, info in participants.items() if info['experienced'] == experienced])].reset_index(drop=True)
        # Clean transform_type labels
        df_new["transform_type"] = df_new["transform_type"].str.replace(
            "TransformType.", "")
        # Extract numeric task index for sorting
        df_new["task_idx_num"] = df_new["task_index"].str.replace(
            "task_idx_", "").astype(int)

        means = {}
        stds = {}

        for task in tasks:

            df_filtered = df_new[df_new['task_id']
                                 == task].reset_index(drop=True)

            # accumulators for combined workflow load z-score
            wl_vals = {"NONE": [], "LINEAR": [], "NONLINEAR": []}
            wl_std = None
            wl_count = 0

            for res in task_metric_map[task] + task_metric_map["common"] + task_metric_map["workflow"]:

                std = None
                if task == "recurrence" and res == "recurrence":
                    temp1 = df_filtered.groupby("transform_type")[res].sum()
                    temp1 = utils.count_confusion_values(temp1.to_dict())

                    temp = {
                        "NONE": temp1["NONE"]["Correct"] / (temp1["NONE"]["Correct"] + temp1["NONE"]["Incorrect"]),
                        "LINEAR": temp1["LINEAR"]["Correct"] / (temp1["LINEAR"]["Correct"] + temp1["LINEAR"]["Incorrect"]),
                        "NONLINEAR": temp1["NONLINEAR"]["Correct"] / (temp1["NONLINEAR"]["Correct"] + temp1["NONLINEAR"]["Incorrect"]),
                    }

                    # zero std
                    std = {"NONE": 0.0, "LINEAR": 0.0, "NONLINEAR": 0.0}
                elif res.endswith("_abs_5"):
                    temp = df_filtered.groupby("transform_type")[res].mean()
                    temp = utils.convert_vals_to_percent(temp.to_dict())
                    std = {}
                    std["NONE"] = df_filtered[df_filtered["transform_type"] ==
                                              "NONE"][res].std()
                    std["LINEAR"] = df_filtered[df_filtered["transform_type"] ==
                                                "LINEAR"][res].std()
                    std["NONLINEAR"] = df_filtered[df_filtered["transform_type"] ==
                                                   "NONLINEAR"][res].std()
                elif res in task_metric_map["workflow"]:
                    temp = df_filtered.groupby("transform_type")[res].mean().reindex([
                        "NONE", "LINEAR", "NONLINEAR"])
                    vals = temp.values.astype(float)

                    # z-scores across transform types (handle zero variance)
                    std_across_types = np.nanstd(vals, ddof=0)
                    zscores = np.zeros_like(vals) if not np.isfinite(
                        std_across_types) or std_across_types == 0 else (vals - np.nanmean(vals)) / std_across_types
                    zdict = dict(
                        zip(["NONE", "LINEAR", "NONLINEAR"], map(float, zscores)))

                    for k in wl_vals:
                        wl_vals[k].append(zdict[k])
                    wl_count += 1

                    continue
                else:
                    temp = df_filtered.groupby("transform_type")[res].mean()

                    # change order to "NONE", "LINEAR", "NONLINEAR
                    temp = temp.reindex(["NONE", "LINEAR", "NONLINEAR"])
                    temp = temp.to_dict()
                    std = {}
                    std["NONE"] = df_filtered[df_filtered["transform_type"] ==
                                              "NONE"][res].std()
                    std["LINEAR"] = df_filtered[df_filtered["transform_type"] ==
                                                "LINEAR"][res].std()
                    std["NONLINEAR"] = df_filtered[df_filtered["transform_type"] ==
                                                   "NONLINEAR"][res].std()

                # dict of dicts
                if task not in means:
                    means[task] = {}
                if task not in stds:
                    stds[task] = {}

                if "duration" in res:
                    res = "Duration (s)"
                elif "rel" in res:
                    res = "Distance (mm)"
                elif "abs" in res or "recurrence" in res.lower():
                    res = "Correctness (\\%)"

                means[task][res] = temp
                stds[task][res] = std

            # add combined workflow load (average of per-metric z-scores)
            if any(len(v) for v in wl_vals.values()):
                means[task]["Workflow load (z-score)"] = {k: float(
                    np.mean(wl_vals[k])) if wl_vals[k] else float("nan") for k in wl_vals}
                stds[task]["Workflow load (z-score)"] = {k: float(
                    np.std(wl_vals[k], ddof=0)) if wl_vals[k] else 0.0 for k in wl_vals}

        x = 0

        means = {
            "A. Vertebralis R.": means["a_vertebralis_r"],
            "A. Vertebralis L.": means["a_vertebralis_l"],
            "A. Carotis Externa R.": means["a_carotisexterna_r"],
            "A. Carotis Externa L.": means["a_carotisexterna_l"],
            "Lymph Node": means["lymph_node"],
            "Recurrence": means["recurrence"],
        }
        stds = {
            "A. Vertebralis R.": stds["a_vertebralis_r"],
            "A. Vertebralis L.": stds["a_vertebralis_l"],
            "A. Carotis Externa R.": stds["a_carotisexterna_r"],
            "A. Carotis Externa L.": stds["a_carotisexterna_l"],
            "Lymph Node": stds["lymph_node"],
            "Recurrence": stds["recurrence"],
        }

        means_all.append(means)
        stds_all.append(stds)
    
    means = average_nested_dicts(means_all[0], means_all[1])
    stds = average_nested_dicts(stds_all[0], stds_all[1])

    metrics_tex = utils.json_to_latex_tables(means, stds)
    with open("outputs/metrics_table.tex", "w") as f:
        f.write(metrics_tex)
    x = 0


if __name__ == "__main__":
    main()
