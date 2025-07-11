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

df_duration = log_parsing.compute_task_duration_by_index_v2(df)

df_bifurcations = study_data_handling.insert_bifurcations(
    df_duration, path_gt, path_rt, tolerance=5)
df_bifurcations = df_bifurcations[~df_bifurcations['patient_id'].str.contains(
    'training')]

df_l = study_data_handling.insert_lymphnodes(
    df_bifurcations, path_gt, path_rt)

df_only_lymph = df_l[df_l['lymph_node_abs'].notna()]

df_clean = df_bifurcations.dropna(subset=['result_abs'])
contingency = pd.crosstab(df_clean['transform_type'], df_clean['result_abs'])
chi2, p, dof, expected = chi2_contingency(contingency)
summary: pd.DataFrame = pd.crosstab(
    df_clean['transform_type'], df_clean['result_abs'])

print(f"Chi2: {chi2:.4f}, p-value: {p:.4f}")
print(summary)

df_clean = df_bifurcations.dropna(subset=['result_rel'])
# group values by transform_type
groups = [group['result_rel'].values for name,
          group in df_clean.groupby('transform_type')]

# ANOVA
f_stat, p_val = f_oneway(*groups)
print(f"ANOVA F: {f_stat:.4f}, p-value: {p_val:.4f}")

# If non-normal: Kruskal-Wallis
h_stat, p_kw = kruskal(*groups)
print(f"Kruskal H: {h_stat:.4f}, p-value: {p_kw:.4f}")

x = 0

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

# df_sorted = df.sort_values(by="result_rel", ascending=False)


path_tre = r"/home/fryderyk/Documents/code/registrationViewer/tres_linear_deformable.csv"
df_tre = pd.read_csv(path_tre)
df_tre = df_tre[~df_tre['patient_id'].str.contains('training')]

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
