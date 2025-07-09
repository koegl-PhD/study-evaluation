from typing import Tuple, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import log_parsing
import all_evaluations
import study_data_handling

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)  # 0 means auto-detect the terminal width

path_log = r"/home/fryderyk/Downloads/rad_test/rad_test.log"
path_gt = r"/home/fryderyk/Downloads/SerielleCTs_nii_forHumans_annotations"
path_rt = r"/home/fryderyk/Downloads/rad_test"

df = log_parsing.load_log_v2_to_df(path_log)

df_duration = log_parsing.compute_task_duration_by_index_v2(df)

df_duration['result_abs'] = None
df_duration['result_rel'] = None

b = study_data_handling.insert_bifurcations(
    df_duration, path_gt, path_rt, tolerance=2)

# all_evaluations.plot_duration_by_task_and_transform(
#     df, type='bar', significance=True)
# all_evaluations.plot_duration_by_task_and_transform(
#     df, type='violin', significance=True)
# print 5th and 5th last row from the df

stats = all_evaluations.statistical_significance_duration(df)

gt = all_evaluations.get_recurrence_gt()

rt = all_evaluations.get_rad_recurrence(
    r"/home/koeglf/data/registrationStudy/study_output/rad_ihssan_20250704/")


tpr, fpr, tnr, fnr, accuracy, result = all_evaluations.get_recurrence_present_values(
    gt, rt)

# save stats to .csv
path = "/home/fryderyk/Documents/code/registrationViewer/stats.csv"
# stats.to_csv(path, index=False)


x = 0


"""
incorrcet recurrence:
a9ebcF7RKU4 (linear)
6vkfAvGWUPg (none)
I307KZkh1VM (nonlinear)
JlS0Cl1K0 (nonlinear)
7sp2FiVa4WI (none)
2pO8AtRxHAg (none)
TwU508CCA9Y (linear)
"""
