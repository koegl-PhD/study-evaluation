#!/usr/bin/env python3

import pandas as pd
import inspect
# assumes your functions are saved in analysis_functions.py
import analysis_functions as af
import all_evaluations


def main():
    # Load the results
    df = pd.read_csv('results.csv')
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

    all_evaluations.plot_duration_by_task_and_transform(
        df[df["user_id"] == "rad_2"], 'violin', significance=True)
    all_evaluations.plot_bifurcation_error_by_transform(
        df[df["user_id"] == "rad_2"], significance=True)
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
