import glob
import json
import os

from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests


import log_parsing


# mean durations
def plot_duration_by_task_and_transform(df: pd.DataFrame, type: Literal['bar', 'violin'], significance: bool = False) -> Optional[pd.DataFrame]:
    """
    Plot duration_seconds grouped by task_id and transform_type.
    Supports 'bar' (mean ± std) and 'violin' plots.
    Excludes task_ids containing 'training' and orders transform types.
    """

    transform_order = ['TransformType.NONE',
                       'TransformType.LINEAR', 'TransformType.NONLINEAR']
    df['transform_type'] = pd.Categorical(
        df['transform_type'],
        categories=transform_order,
        ordered=True
    )

    plt.figure(figsize=(12, 6))
    if type == 'bar':
        ax = sns.barplot(
            data=df,
            x='task_id',
            y='duration_seconds',
            hue='transform_type',
            # hue_order=transform_order,
            errorbar='sd'
        )

        plt.ylabel('Mean Duration (seconds)')
        plt.title(
            'Mean Duration by Task and Transform Type')

    elif type == 'violin':
        ax = sns.violinplot(
            data=df,
            x='task_id',
            y='duration_seconds',
            hue='transform_type',
            hue_order=transform_order,
            split=False,
            scale='width'
        )
        plt.ylabel('Duration (seconds)')
        plt.title('Duration by Task and Transform Type')

    test_results = None

    if significance:
        test_results = statistical_significance_duration(df)
        pairs = []
        pvalues = []
        for _, row in test_results.iterrows():
            if row['significant']:
                t_id = row['task_id']
                t1, t2 = row['pair'].split(' vs ')
                idx1 = transform_order.index(t1)
                idx2 = transform_order.index(t2)
                if idx1 > idx2:
                    t1, t2 = t2, t1
                pairs.append(((t_id, t1), (t_id, t2)))
                pvalues.append(row['pval_corrected'])

        if pairs != []:
            annotator = Annotator(ax, pairs, data=df,
                                  x='task_id', y='duration_seconds', hue='transform_type')

            line_height = .02 if type == 'bar' else .2

            annotator.configure(
                test=None,
                text_format='star',
                line_height=line_height,
            )
            annotator.set_pvalues_and_annotate(pvalues)

    plt.legend(title='Transform Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    if test_results is not None:
        return test_results
    return None


def statistical_significance_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise t-tests per task_id with Bonferroni correction within each task."""
    results = []

    for task_id, group in df.groupby('task_id'):
        types = group['transform_type'].unique()
        pairs = [(a, b) for i, a in enumerate(types) for b in types[i+1:]]
        pvals = []
        pair_labels = []
        for a, b in pairs:
            data_a = group[group['transform_type'] == a]['duration_seconds']
            data_b = group[group['transform_type'] == b]['duration_seconds']
            stat, pval = ttest_ind(data_a, data_b)
            pvals.append(pval)
            pair_labels.append(f"{a} vs {b}")

        corrected = multipletests(pvals, method='bonferroni')
        for i, pair in enumerate(pair_labels):
            results.append({
                'task_id': task_id,
                'pair': pair,
                'pval_raw': pvals[i],
                'pval_corrected': corrected[1][i],
                'significant': corrected[0][i]
            })

    return pd.DataFrame(results)
