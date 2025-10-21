import csv
import pandas as pd
from pathlib import Path

# p_init = r"/data/registrationEvaluation/gt_and_rt/results_dsc_initial.csv"
p_init = r"/data/registrationEvaluation/gt_and_rt/results_dsc_niftyreg.csv"

path_annot = r"/data/registrationEvaluation/gt_and_rt/SerielleCTs_nii_forHumans_annotations"


# read p_init into pandas dataframe
df_init = pd.read_csv(p_init)

# get values from first column
images = sorted(df_init.iloc[:, 0].tolist()[:-4])

root = Path(path_annot)
all_csv_files = [str(p) for p in root.rglob("*.json")
                 if 'points' in p.name and 'b_followup' in p.name]
cases_from_csv_files = []
for case in all_csv_files:
    res = case.split('/')[-1][7:].split('.mrk')[0]
    cases_from_csv_files.append(res)

    print(case)
    print(res)
    print()

# check if each case in cases has a corresponding json file
not_matched = []
for case in cases_from_csv_files:
    if case not in images:
        not_matched.append(case)

not_matched.sort()

# remove all rows from df_init where first column values is not in cases_from_csv_files
df_init = df_init[df_init.iloc[:, 0].isin(cases_from_csv_files)]

df_init.to_csv(p_init, index=False)

x = 0
