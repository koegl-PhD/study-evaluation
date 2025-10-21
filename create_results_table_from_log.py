import json

from typing import Dict

import pandas as pd

import log_parsing
import study_data_handling
import utils


def main(
        path_gt: str,
        participants: Dict[str, Dict[str, int | bool | str]],
        path_dsc_init: str,
        path_dsc_nifty: str
) -> None:

    df = []
    # """
    for rad_contents in participants.values():

        path_log = str(rad_contents['path_log'])
        path_rt = str(rad_contents['path_rt'])

        df_rad = log_parsing.load_log_to_df(path_log)

        interaction_stats = log_parsing.aggregate_interaction_stats(
            df_rad)

        df_rad = log_parsing.compute_task_duration(df_rad)

        df_rad = study_data_handling.insert_study_results(
            df_rad,
            path_gt,
            path_rt, 5,
            rad_contents)

        df_rad: pd.DataFrame = df_rad.merge(
            right=interaction_stats,
            on=["user_id", "patient_id", "transform_type", "task_id", "task_index"],
            how="left"
        )

        df.append(df_rad)

    df = pd.concat(df, ignore_index=True)

    df = utils.apply_corrections(df)

    df = study_data_handling.add_dsc(df, path_dsc_init, path_dsc_nifty)

    df = utils.combine_tres(df)
    # """

    # df = pd.read_csv('outputs/results.csv')

    df = utils.reorder_columns(df)

    df.to_csv('outputs/results.csv', index=False)

    x = 0


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.width', 0)  # 0 means auto-detect the terminal width

    path_radiologists = r"/data/registrationEvaluation/gt_and_rt/study_results"

    path_dsc_init = r"/data/registrationEvaluation/gt_and_rt/results_dsc_initial.csv"
    path_dsc_nifty = r"/data/registrationEvaluation/gt_and_rt/results_dsc_niftyreg.csv"

    participants: Dict[str, Dict[str, int | bool | str]
                       ] = json.load(open('resources/participants.json'))

    for rad_id in participants.keys():
        participants[rad_id]['path_rt'] = f"{path_radiologists}/{rad_id}"
        participants[rad_id]['path_log'] = f"{participants[rad_id]['path_rt']}/{rad_id}.log"

    path_gt = r"/data/registrationEvaluation/gt_and_rt/SerielleCTs_nii_forHumans_annotations"

    main(path_gt, participants, path_dsc_init, path_dsc_nifty)
