from typing import Optional
from scipy.stats import chi2_contingency
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


def plot_bifurcation_duration_by_transform(
    df: pd.DataFrame,
    participants: Dict[str, Dict[str, int | bool | str]],
    rad_id: str,
    significance: bool = False,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Axes, pd.DataFrame]:
    """
    Return (ax, results) for a violin plot of bifurcation_rel by transform_type; adds significance stars if requested.
    """

    tasks = ['a_vertebralis_r', 'a_vertebralis_l',
             'a_carotisexterna_r', 'a_carotisexterna_l']

    df_rad = df[df["user_id"] == rad_id].copy()

    rad_group = participants[rad_id]['group']
    rad_experienced = participants[rad_id]['experienced']

    df_sub = df_rad[df_rad['task_id'].isin(tasks)].copy()
    df_sub['duration_seconds'] = pd.to_numeric(
        df_sub['duration_seconds'], errors='coerce')
    df_sub = df_sub.dropna(subset=['duration_seconds'])

    order = ['TransformType.NONE',
             'TransformType.LINEAR', 'TransformType.NONLINEAR']
    df_sub['transform_type'] = pd.Categorical(
        df_sub['transform_type'], categories=order, ordered=True)

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
        created_fig = True

    sns.violinplot(
        data=df_sub, x='transform_type', y='duration_seconds',
        order=order, density_norm='width', cut=0, ax=ax
    )
    ax.set_xlabel('Transform Type')
    ax.set_ylabel('Relative Bifurcation')
    ax.set_title(
        f"Bifurcation duration by Transform Type\n{rad_id} - {'Experienced' if rad_experienced else 'Inexperienced'} - Group {rad_group}")

    results = None
    if significance:
        results = statistical_significance_bifurcation(df_sub)
        sig = results[results['significant']]
        if not sig.empty:
            pairs = [tuple(p.split(' vs ')) for p in sig['pair']]
            pvals = sig['pval_corrected'].tolist()
            annot = Annotator(ax, pairs, data=df_sub,
                              x='transform_type', y='duration_seconds', verbose=False)
            annot.configure(test=None, text_format='star', line_height=0.2)
            annot.set_pvalues_and_annotate(pvals)

    if created_fig:
        plt.tight_layout()

    return ax, results


def plot_bifurcation_error_by_transform(
    df: pd.DataFrame,
    participants: Dict[str, Dict[str, int | bool | str]],
    rad_id: str,
    significance: bool = False,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Axes, pd.DataFrame]:
    """
    Return (ax, results) for a violin plot of bifurcation_rel by transform_type; adds significance stars if requested.
    """

    tasks = ['a_vertebralis_r', 'a_vertebralis_l',
             'a_carotisexterna_r', 'a_carotisexterna_l']

    df_rad = df[df["user_id"] == rad_id].copy()

    rad_group = participants[rad_id]['group']
    rad_experienced = participants[rad_id]['experienced']

    df_sub = df_rad[df_rad['task_id'].isin(tasks)].copy()
    df_sub['bifurcation_rel'] = pd.to_numeric(
        df_sub['bifurcation_rel'], errors='coerce')
    df_sub = df_sub.dropna(subset=['bifurcation_rel'])

    order = ['TransformType.NONE',
             'TransformType.LINEAR', 'TransformType.NONLINEAR']
    df_sub['transform_type'] = pd.Categorical(
        df_sub['transform_type'], categories=order, ordered=True)

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
        created_fig = True

    sns.violinplot(
        data=df_sub, x='transform_type', y='bifurcation_rel',
        order=order, density_norm='width', cut=0, ax=ax
    )
    ax.set_xlabel('Transform Type')
    ax.set_ylabel('Relative Bifurcation')
    ax.set_title(
        f"Bifurcation error by Transform Type\n{rad_id} - {'Experienced' if rad_experienced else 'Inexperienced'} - Group {rad_group}")

    results = None
    if significance:
        results = statistical_significance_bifurcation(df_sub)
        sig = results[results['significant']]
        if not sig.empty:
            pairs = [tuple(p.split(' vs ')) for p in sig['pair']]
            pvals = sig['pval_corrected'].tolist()
            annot = Annotator(ax, pairs, data=df_sub,
                              x='transform_type', y='bifurcation_rel', verbose=False)
            annot.configure(test=None, text_format='star', line_height=0.2)
            annot.set_pvalues_and_annotate(pvals)

    ax.axhline(5, linestyle=':', linewidth=1, zorder=10,
               color='red', label='Threshold = 5mm')

    if created_fig:
        plt.tight_layout()

    return ax, results


def statistical_significance_bifurcation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise t-tests on bifurcation_rel across transform_type with Bonferroni correction.
    """
    types = df['transform_type'].cat.categories.tolist()
    pairs = [(a, b) for i, a in enumerate(types) for b in types[i+1:]]
    pvals = []
    labels = []
    for a, b in pairs:
        da = df[df['transform_type'] == a]['bifurcation_rel']
        db = df[df['transform_type'] == b]['bifurcation_rel']
        _, p = ttest_ind(da, db)
        pvals.append(p)
        labels.append(f"{a} vs {b}")

    reject, p_corr, _, _ = multipletests(pvals, method='bonferroni')
    results = [
        {'pair': labels[i], 'pval_raw': pvals[i],
            'pval_corrected': p_corr[i], 'significant': bool(reject[i])}
        for i in range(len(pvals))
    ]
    return pd.DataFrame(results)


def plot_recurrence_accuracy_by_transform(df: pd.DataFrame, significance: bool = False) -> Optional[pd.DataFrame]:
    """
    Plot recurrence correctness rate by transform_type and optionally annotate pairwise significance.
    """
    df_sub = df.dropna(subset=['recurrence']).copy()
    df_sub['correct'] = df_sub['recurrence'].isin(['tp', 'tn']).astype(int)

    order = ['TransformType.NONE',
             'TransformType.LINEAR', 'TransformType.NONLINEAR']
    df_sub['transform_type'] = pd.Categorical(
        df_sub['transform_type'], categories=order, ordered=True)

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df_sub, x='transform_type',
                     y='correct', order=order, ci=95)
    plt.ylabel('Proportion Correct')
    plt.xlabel('Transform Type')
    plt.title('Recurrence Classification Accuracy by Transform Type')

    results: Optional[pd.DataFrame] = None
    if significance:
        # pairwise chi-square tests
        types = order
        pairs = [(types[i], types[j]) for i in range(len(types))
                 for j in range(i+1, len(types))]
        pvals = []
        labels = []
        for a, b in pairs:
            sub = df_sub[df_sub['transform_type'].isin([a, b])]
            cont = pd.crosstab(sub['transform_type'], sub['correct'])
            _, p, _, _ = chi2_contingency(cont)
            pvals.append(p)
            labels.append(f"{a} vs {b}")

        reject, p_corr, _, _ = multipletests(pvals, method='bonferroni')
        results = pd.DataFrame([
            {'pair': labels[i], 'pval_raw': pvals[i],
                'pval_corrected': p_corr[i], 'significant': bool(reject[i])}
            for i in range(len(pvals))
        ])

        sig = results[results['significant']]
        if not sig.empty:
            sig_pairs = [tuple(p.split(' vs ')) for p in sig['pair']]
            sig_pvals = sig['pval_corrected'].tolist()
            annot = Annotator(ax, sig_pairs, data=df_sub,
                              x='transform_type', y='correct')
            annot.configure(test=None, text_format='star', line_height=0.2)
            annot.set_pvalues_and_annotate(sig_pvals)

    plt.tight_layout()
    plt.show()
    return results
