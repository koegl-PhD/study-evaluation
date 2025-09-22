import json

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

import all_evaluations
import analysis_functions as af
import utils


def main():
    # Load the results
    df = pd.read_csv('results.csv')

    participants: Dict[str, Dict[str, int | bool | str]
                       ] = json.load(open('resources/participants.json'))

    df_new = utils.remove_calibration(df)

    # keep only experienced radiologists
    df_new = df_new[df_new['user_id'].isin(
        [uid for uid, info in participants.items() if info['experienced']])].reset_index(drop=True)
    # Clean transform_type labels
    df_new["transform_type"] = df_new["transform_type"].str.replace(
        "TransformType.", "")
    # remove rows where transform_type is NONE
    df = df_new[df_new["transform_type"] != "NONE"].reset_index(drop=True)

    tasks = json.load(open('resources/tasks.json'))[:-1]

    with open('outputs/correlation_tre.tex', 'w') as f:

        latex = utils.make_tre_tabular(df, tasks)

        f.write(latex)


if __name__ == "__main__":
    main()
