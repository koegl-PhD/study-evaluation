import json

from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

import all_evaluations
import analysis_functions as af
import utils


def main():
    # Load the results
    df = pd.read_csv('results.csv')

    participants: Dict[str, Dict[str, int | bool | str]
                       ] = json.load(open('resources/participants.json'))
    task_metric_map: Dict[str, List[str]] = json.load(
        open('resources/task_metric_map.json'))
    tasks = json.load(open('resources/tasks.json'))

    df_new = utils.remove_calibration(df)
    df_new.to_csv("temp.csv")

    # keep only experienced radiologists
    df_new = df_new[df_new['user_id'].isin(
        [uid for uid, info in participants.items() if info['experienced']])].reset_index(drop=True)
    # Clean transform_type labels
    df_new["transform_type"] = df_new["transform_type"].str.replace(
        "TransformType.", "")
    # Extract numeric task index for sorting
    df_new["task_idx_num"] = df_new["task_index"].str.replace(
        "task_idx_", "").astype(int)

    means = {}

    for task in tasks:

        df_filtered = df_new[df_new['task_id'] == task].reset_index(drop=True)

        for res in task_metric_map[task] + task_metric_map["common"]:

            if task == "recurrence" and res == "recurrence":
                temp = df_filtered.groupby("transform_type")[res].sum()
                temp = utils.count_confusion_values(temp.to_dict())
            elif res.endswith("_abs_5"):
                temp = df_filtered.groupby("transform_type")[res].mean()
                temp = utils.convert_vals_to_percent(temp.to_dict())
            else:
                temp = df_filtered.groupby("transform_type")[res].mean()

                # change order to "NONE", "LINEAR", "NONLINEAR
                temp = temp.reindex(["NONE", "LINEAR", "NONLINEAR"])
                temp = temp.to_dict()

            # dict of dicts
            if task not in means:
                means[task] = {}

            if "duration" in res:
                res = "Duration (s)"
            elif "rel" in res:
                res = "Distance (mm)"
            elif "abs" in res:
                res = "Correctness (\\%)"

            means[task][res] = temp

    x = 0

    for task, results in means.items():
        print(f"Task: {task}")

        for result, values in results.items():
            print(f"  Result: {result}")
            for transform_type, mean_value in values.items():

                val = f"{mean_value:.2f}" if isinstance(
                    mean_value, float) else mean_value

                print(f"    {transform_type}: {val}")

        x = 0
        for _ in range(5):
            print()

    metrics_tex, recurrence_tex = utils.json_to_latex_tables(means)
    with open("outputs/metrics_table.tex", "w") as f:
        f.write(metrics_tex)
    with open("outputs/recurrence_table.tex", "w") as f:
        f.write(recurrence_tex)
    x = 0


if __name__ == "__main__":
    main()
