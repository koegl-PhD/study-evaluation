from typing import List
import json

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

import all_evaluations
import analysis_functions as af
import utils


def get_common_patients(df: pd.DataFrame, rads: List[str]) -> pd.DataFrame:

    patient_sets = []

    for rad in rads:
        patient_sets.append(set(df[df['user_id'] == rad]['patient_id']))

    common_patients = set.intersection(*patient_sets)
    return df[df['patient_id'].isin(common_patients)].reset_index(drop=True)


def merge_user_errors(df: pd.DataFrame, new_col: str = "user_error", drop: bool = True) -> pd.DataFrame:
    """Merge non-overlapping *_rel columns into one column; optionally drop originals."""
    rel_cols: List[str] = [c for c in df.columns if c.endswith("_rel")]
    if not rel_cols:
        raise ValueError("No columns ending with '_rel' found.")
    too_many = df[rel_cols].notna().sum(axis=1) > 1
    if bool(too_many.any()):
        idxs = df.index[too_many].tolist()
        raise ValueError(
            f"Rows have overlapping values in _rel columns at indices: {idxs[:10]}")
    merged = df[rel_cols].bfill(axis=1).iloc[:, 0]
    out = df.copy()
    out[new_col] = merged
    if drop:
        out = out.drop(columns=rel_cols)
    return out


def main():
    # Load the results
    df = pd.read_csv('results.csv')

    participants: Dict[str, Dict[str, int | bool | str]
                       ] = json.load(open('resources/participants.json'))

    df = utils.remove_calibration(df)

    # df = df[df['user_id'].isin(
    #     [uid for uid, info in participants.items() if info['experienced']])].reset_index(drop=True)

    df = df[df["transform_type"] !=
            "TransformType.NONE"].reset_index(drop=True)

    all_rads = [("rad_2", "rad_4"), ("rad_2", "rad_5"), ("rad_4", "rad_5")]

    rads = ['rad_5', 'rad_2']
    df = df[df['user_id'].isin(rads)].reset_index(drop=True)

    task_choices = ["a_vertebralis_r"]  # , "a_vertebralis_l",
    # "a_carotisexterna_l", "a_carotisexterna_r"]
    df = df[df["task_id"].isin(task_choices)].reset_index(drop=True)
    df = df.iloc[:, :16 + 1]
    df = df.drop(df.columns[[1, 2, 6, 9, 11, 13, 15]], axis=1)

    # df = get_common_patients(df, rads)
    df = merge_user_errors(df, new_col="user_error", drop=True)

    # df_grouped = df.groupby(
    #     "patient_id", as_index=False).mean(numeric_only=True)

    """
    user_error = df['user_error'].values
    tre = df['bifurcation_tre'].values

    # combine both and sort by tre
    combined = list(zip(tre, user_error))
    combined_sorted = sorted(combined, key=lambda x: x[0])
    tre_sorted, user_error_sorted = zip(*combined_sorted)

    # plot bifurcation_tre and user_error as two lines
    plt.plot(tre_sorted, label='TRE (mm)')
    plt.plot(user_error_sorted, label='User Error (mm)')
    plt.title('TRE and User Error')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()
    """

    x = 0

    ####################################################
    ####################################################
    ####################################################
    ####################################################
    ####################################################
    ####################################################
    ####################################################
    ####################################################
    ####################################################
    ####################################################
    ####################################################

    df_rad_2 = df[df['user_id'] == rads[0]].reset_index(drop=True)
    # df_rad_2["duration_seconds"] = 50 * \
    # df_rad_2["duration_seconds"] / df_rad_2["duration_seconds"].mean()
    df_rad_4 = df[df['user_id'] == rads[1]].reset_index(drop=True)
    # df_rad_4["duration_seconds"] = 50 * \
    # df_rad_4["duration_seconds"] / df_rad_4["duration_seconds"].mean()

    # subtract columns 'duration_seconds', 'bifurcation_tre' of df_rad_2 - df_rad_4
    # 2 is LINEAR, 4 is NONLINEAR
    df_diff = df_rad_2[['patient_id', 'task_id',
                        'duration_seconds', 'bifurcation_tre']].copy()
    df_diff = df_diff.rename(columns={'duration_seconds': 'duration_seconds_rad_2',
                                      'bifurcation_tre': 'bifurcation_tre_rad_2'})
    df_diff['duration_seconds_rad_4'] = df_rad_4['duration_seconds']
    df_diff['bifurcation_tre_rad_4'] = df_rad_4['bifurcation_tre']
    df_diff['duration_seconds_diff'] = df_diff['duration_seconds_rad_2'] - \
        df_diff['duration_seconds_rad_4']
    df_diff['bifurcation_tre_diff'] = df_diff['bifurcation_tre_rad_2'] - \
        df_diff['bifurcation_tre_rad_4']

    # df_diff = df_diff.groupby(
    #     "patient_id", as_index=False).mean(numeric_only=True)

    duration_diff = df_diff['duration_seconds_diff'].values
    tre_diff = df_diff['bifurcation_tre_diff'].values

    # combine both and sort by tre_diff
    combined = list(zip(tre_diff, duration_diff))
    combined_sorted = sorted(combined, key=lambda x: x[0])
    tre_diff_sorted, duration_diff_sorted = zip(*combined_sorted)

    # plot bifurcation_tre_diff and duration_seconds_diff as two lines
    plt.plot(tre_diff_sorted, label='Difference in TRE (mm)')
    plt.plot(duration_diff_sorted, label='Difference in Duration (seconds)')
    plt.title(f"Difference in TRE and Duration ({rads[0]} - {rads[1]})")
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

    x = 0


if __name__ == "__main__":
    main()
