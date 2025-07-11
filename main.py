from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, kruskal

import log_parsing
import all_evaluations
import study_data_handling


def main(
        path_gt: str,
        participants: Dict[str, Dict[str, int | bool | str]]
) -> None:

    df = []

    for rad_contents in participants.values():

        path_log = str(rad_contents['path_log'])
        path_rt = str(rad_contents['path_rt'])

        df_rad = log_parsing.load_log_v2_to_df(path_log)

        df_grouped = log_parsing.compute_scroll_stats_grouped_v2(df_rad)
        # find all indices where two consecutive task_index are the same

        df_scroll = log_parsing.compute_task_scroll_stats_v2(df_rad)

        common_keys = ['user_id', 'patient_id',
                       'transform_type', 'task_id', 'task_index']
        merged = df_grouped.merge(
            df_scroll, how='outer', on=common_keys).fillna(0)

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
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.width', 0)  # 0 means auto-detect the terminal width

    path_radiologists = r"/home/fryderyk/Downloads"

    participants: Dict[str, Dict[str, int | bool | str]] = {
        "rad_test": {
            "group": 1,
            "experienced": False,
        }
    }
    for rad_id in participants.keys():
        participants[rad_id]['path_rt'] = f"{path_radiologists}/{rad_id}"
        participants[rad_id]['path_log'] = f"{participants[rad_id]['path_rt']}/{rad_id}.log"

    path_gt = r"/home/fryderyk/Downloads/SerielleCTs_nii_forHumans_annotations"

    main(path_gt, participants)
