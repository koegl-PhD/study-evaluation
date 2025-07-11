from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, kruskal

import log_parsing
import all_evaluations
import study_data_handling

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 260)
pd.set_option('display.width', 0)  # 0 means auto-detect the terminal width

path_log = r"/home/fryderyk/Downloads/rad_test/rad_test.log"
path_gt = r"/home/fryderyk/Downloads/SerielleCTs_nii_forHumans_annotations"
path_rt = r"/home/fryderyk/Downloads/rad_test"

df = log_parsing.load_log_v2_to_df(path_log)

df = log_parsing.compute_task_duration_by_index_v2(df)

df = study_data_handling.insert_study_results(df, path_gt, path_rt, 5)

x = 0