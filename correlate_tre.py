import json

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

import all_evaluations
import analysis_functions as af
import utils


def main():
    # Load the results
    df = pd.read_csv('results.csv')

    participants: Dict[str, Dict[str, int | bool | str]
                       ] = json.load(open('resources/participants.json'))

    df = utils.remove_calibration(df)

    # df = df[df['user_id'].isin(
    #     [uid for uid, info in participants.items() if info['experienced']])].reset_index(drop=True)

    rads = ['rad_2', 'rad_4', 'rad_5']

    tre_and_duration = {}
    tre_and_duration_sorted_by_tre = {}
    tre_and_user_error = {}
    tre_and_user_error_sorted_by_tre = {}

    for rad in rads:
        df_rad = df[df['user_id'] == rad].reset_index(drop=True)

        # keep only where transform_type is not NONE
        df_rad = df_rad[df_rad["transform_type"] !=
                        "TransformType.NONE"].reset_index(drop=True)

        rel_cols = [c for c in df.columns if c.endswith("_rel")]
        # task_to_errorcol = {c.replace("_rel", ""): c for c in rel_cols}
        task_to_errorcol = {"lymph_node": "lymph_node_tre"}

        dfs = []
        for task_id, df_task in df_rad.groupby("task_id"):
            base_name = task_id
            if base_name in task_to_errorcol:
                error_col = task_to_errorcol[base_name]
            else:
                continue
            tmp = df_task[["task_index", "duration_seconds", error_col, "transform_type"]].copy()
            # tmp = tmp.rename(columns={error_col: "user_error"})
            tmp["task_id"] = task_id
            dfs.append(tmp)

        df_all2 = pd.concat(dfs).dropna()

        duration = df_all2["duration_seconds"].values
        # user_error = df_all2["user_error"].values

        tre = df_all2["lymph_node_tre"].values

        tre_and_duration[rad] = list(zip(tre, duration))
        tre_and_duration_sorted_by_tre[rad] = sorted(
            tre_and_duration[rad], key=lambda x: x[0])

        # tre_and_user_error[rad] = list(zip(tre, user_error))
        # tre_and_user_error_sorted_by_tre[rad] = sorted(
        #     tre_and_user_error[rad], key=lambda x: x[0])

    # make figure with 4 subplots - each having two lines, one for tre and one for duration or user_error
    fig, axs = plt.subplots(len(rads), 2, figsize=(10, len(rads) * 5))

    for i, rad in enumerate(rads):
        axs[i, 0].set_title(f"Rad {rad} - TRE vs Duration (sorted by TRE)")
        axs[i, 0].set_xlabel("Sample index")
        axs[i, 0].set_ylabel("Value")
        axs[i, 0].plot(
            [x[0] for x in tre_and_duration_sorted_by_tre[rad]], label="TRE")
        axs[i, 0].plot([x[1]
                        for x in tre_and_duration_sorted_by_tre[rad]], label="Duration")
        axs[i, 0].legend()

        # axs[i, 1].set_title(f"Rad {rad} - TRE vs User Error (sorted by TRE)")
        # axs[i, 1].set_xlabel("Sample index")
        # axs[i, 1].set_ylabel("Value")
        # axs[i, 1].plot([x[0]
        #                 for x in tre_and_user_error_sorted_by_tre[rad]], label="TRE")
        # axs[i, 1].plot(
        #     [x[1] for x in tre_and_user_error_sorted_by_tre[rad]], label="User Error")
        # axs[i, 1].legend()

    plt.tight_layout()
    plt.show()
    x = 0


if __name__ == "__main__":
    main()
