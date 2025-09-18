import matplotlib.pyplot as plt
import pandas as pd

from typing import Dict

import all_evaluations
import analysis_functions as af


def main():
    # Load the results
    df = pd.read_csv('results.csv')

    participants: Dict[str, Dict[str, int | bool | str]] = {
        "rad_test": {
            "group": 1,
            "experienced": False,
        },
        "rad_1": {
            "group": 1,
            "experienced": False,
        },
        "rad_2": {
            "group": 1,
            "experienced": True,
        },
        "rad_3": {
            "group": 2,
            "experienced": False,
        }
    }

    # get df where "calibration" is not in patient_id
    # df = df[~df['patient_id'].str.contains('calibration')].reset_index(drop=True)

    # where (user_id == 'rad_1' and transform_type == TransformType.NONLINEAR) or (user_id == 'rad_3' and transform_type == TransformType.NONE)
    df = df[(df['user_id'] == 'rad_1') & (df['transform_type'] == 'TransformType.LINEAR') |
            (df['user_id'] == 'rad_3') & (df['transform_type'] == 'TransformType.NONLINEAR') |
            (df['patient_id'].str.contains('calibration_1')) |
            (df['patient_id'].str.contains('calibration_3')) |
            (df['patient_id'].str.contains('calibration_4'))

            ].reset_index(drop=True)
    # only keep rad_1 and rad_3
    df = df[df['user_id'].isin(['rad_1', 'rad_3'])].reset_index(drop=True)

    # where task is neither lymph_node or recurrence
    df = df[~df['task_id'].isin(
        ['lymph_node', 'recurrence'])].reset_index(drop=True)

    # average duration_seconds by user_id and task_id
    avg_duration = df.groupby(['user_id', 'task_id', 'transform_type'])[
        'duration_seconds'].mean().reset_index()

    # drop all columns except for user_id,experienced,group,patient_id,transform_type,task_id,task_index,duration_seconds,a_vertebralis_r_rel,a_vertebralis_l_rel,a_carotisexterna_r_rel,a_carotisexterna_l_rel,bifurcation_tre,
    keep = [
        'patient_id',
        'user_id', 'transform_type',
        'task_id', 'task_index', 'duration_seconds', 'bifurcation_tre',
    ]
    df = df[keep]
    df.to_csv('filtered_results.csv', index=False)

    # opposite of melt on user_id
    calibration_task_indices = {
        f"task_idx_{i:04d}" for i in [21, 22, 23, 24, 33, 34, 35, 36]}
    df['is_calibration'] = df['task_index'].isin(calibration_task_indices)

    # Split calibration vs non-calibration
    calibration_df = df[df['is_calibration']]
    non_calibration_df = df[~df['is_calibration']]

    baseline_means = calibration_df.groupby(
        'user_id')['duration_seconds'].mean()

    non_calibration_df['duration_relative'] = non_calibration_df.apply(
        lambda row: row['duration_seconds'] / baseline_means[row['user_id']],
        axis=1
    )
    # non_calibration_df.to_csv("temp.csv")


    temp = pd.read_csv("temp_2.csv")

    # plot diff_tre vs diff_dur_rel
    plt.scatter(temp['diff_tre'], temp['diff_dur_rel'])
    plt.xlabel('Difference in TRE')
    plt.ylabel('Difference in Relative Duration')
    plt.title('Difference in TRE vs Difference in Relative Duration')
    plt.grid()
    plt.savefig('diff_tre_vs_diff_dur_rel.png')

    x = 0

    # print(f"Loaded results.csv: {len(df)} rows, {len(df.columns)} columns.")

    # # List all available analysis functions
    # funcs = [name for name, obj in inspect.getmembers(af, inspect.isfunction)]
    # print("Available analysis functions:", funcs)

    # df_rad_1 = df[df['user_id'] == 'rad_1'].reset_index(drop=True)
    # df_rad_test = df[df['user_id'] == 'rad_test'].reset_index(drop=True)

    # # Quick smoke-test: run one function and show a snippet
    # desc_1 = af.descriptive_stats(df_rad_1)
    # desc_test = af.descriptive_stats(df_rad_test)

    # desc_1.to_csv('descriptive_stats_1.csv', index=False)
    # desc_test.to_csv('descriptive_stats_test.csv', index=False)

    # print(all_evaluations.plot_duration_by_task_and_transform(
    #     df_rad_1, 'violin', significance=True))
    # print(all_evaluations.plot_duration_by_task_and_transform(
    #     df_rad_test, 'violin', significance=True))

    # all_evaluations.plot_duration_by_task_and_transform(
    #     df[df["user_id"] == "rad_2"], 'violin', significance=True)

    all_evaluations.plot_all_bifurcation_by_transform(
        df, significance=True, value='error')
    all_evaluations.plot_all_bifurcation_by_transform(
        df, significance=True, value='duration')

    all_evaluations.plot_recurrence_accuracy_by_transform(
        df[df["user_id"] == "rad_2"], significance=True)
    af.extract_questionnaire_data(
        '/home/fryderyk/Downloads/Questionnaire Registration Evaluation (Responses) - Form Responses 1(1).csv', part='general')

    vals = af.get_recurrence_vals(df, rad_id='rad_2')
    af.plot_recurrence_vals(vals)

    recurrence_test = af.get_recurrence_vals(df, rad_id='rad_test')
    recurrence_rad_1 = af.get_recurrence_vals(df, rad_id='rad_1')
    df_summary = af.get_recurrence_vals(df)

    af.plot_recurrence_vals(df_summary)

    x = 0


if __name__ == "__main__":
    main()
