from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.stats import chi2_contingency, f_oneway, kruskal

import log_parsing
import all_evaluations
import study_data_handling
import utils


def main(
        path_gt: str,
        participants: Dict[str, Dict[str, int | bool | str]]
) -> None:

    df = []

    for rad_contents in participants.values():

        path_log = str(rad_contents['path_log'])
        path_rt = str(rad_contents['path_rt'])

        df_rad = log_parsing.load_log_to_df(path_log)

        scroll_stats_old = log_parsing.aggregate_interaction_stats_old(
            df_rad)
        scroll_stats_new = log_parsing.aggregate_interaction_stats_new(
            df_rad)

        if utils.df_equal(scroll_stats_old, scroll_stats_new):
            print(f"Scroll stats are equal for {rad_contents['path_log']}")
        else:
            print(f"Scroll stats differ for {rad_contents['path_log']}")

        df_rad = log_parsing.compute_task_duration_by_index_v2(df_rad)

        df_rad = study_data_handling.insert_study_results(
            df_rad, path_gt, path_rt, 5)

        df_rad.insert(loc=1, column="group", value=rad_contents['group'])
        df_rad.insert(loc=1, column="experienced",
                      value=rad_contents['experienced'])

        df.append(df_rad)

    df = pd.concat(df, ignore_index=True)

    x = 0


res = 83

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.width', 0)  # 0 means auto-detect the terminal width

    path_radiologists = r"/home/fryderyk/Downloads/study_results"

    participants: Dict[str, Dict[str, int | bool | str]] = {
        "rad_test": {
            "group": 1,
            "experienced": False,
        },
        "rad_1": {
            "group": 1,
            "experienced": False,
        }
    }
    for rad_id in participants.keys():
        participants[rad_id]['path_rt'] = f"{path_radiologists}/{rad_id}"
        participants[rad_id]['path_log'] = f"{participants[rad_id]['path_rt']}/{rad_id}.log"

    path_gt = r"/home/fryderyk/Downloads/SerielleCTs_nii_forHumans_annotations"

    main(path_gt, participants)

df[df['user_id'] == 'rad_test'][df['task_id'] ==
                                'recurrence']['recurrence_abs'].value_counts()
