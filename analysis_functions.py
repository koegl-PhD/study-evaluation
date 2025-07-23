from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, median, std, and IQR of numeric metrics grouped by transform_type and experienced. Higher dispersion indicates greater variability."""
    # select only numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg = (
        df.groupby(['transform_type', 'experienced'])[numeric_cols]
        .agg([
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std'),
            ('iqr', lambda s: s.quantile(0.75) - s.quantile(0.25))
        ])
        .reset_index()
    )

    agg = agg.iloc[[1, 0, 2]].reset_index(
        drop=True)  # reorder for better readability
    return agg


def get_recurrence_vals(df: pd.DataFrame, rad_id: str = "") -> pd.DataFrame:
    """
    Creates new dataframe with summary of recurrence values - one row per user_id.
    If rad_id is provided, only that user_id is returned.
    Adds 'total' and 'correct' columns.
    Columns are ordered: user_id, total, correct, tp, tn, fp, fn.
    """
    if rad_id:
        df = df[df['user_id'] == rad_id]
        grouped = [df]
    else:
        grouped = [df[df['user_id'] == uid] for uid in df['user_id'].unique()]

    summaries = []
    for subdf in grouped:
        counts = subdf['recurrence'].value_counts().reindex(
            ['tp', 'tn', 'fp', 'fn'], fill_value=0)
        summary = (
            counts.to_frame().T
            .rename_axis(None)
            .rename_axis(None, axis=1)
            .reset_index(drop=True)
        )
        user_id = subdf['user_id'].iloc[0] if not subdf.empty else rad_id
        total = counts.sum()
        correct = counts['tp'] + counts['tn']

        summary.insert(0, 'user_id', user_id)
        summary.insert(1, 'total', total)
        summary.insert(2, 'correct', correct)

        summaries.append(summary)

    return pd.concat(summaries, ignore_index=True)


def plot_recurrence_vals(df: pd.DataFrame) -> None:
    """
    Plot recurrence values for a specific radiologist or all radiologists.
    """
    import matplotlib.pyplot as plt

    # Assume df_summary is your output DataFrame
    df_plot = df.set_index('user_id')[['tp',  'tn', 'fp', 'fn']]

    df_plot.plot(kind='bar', stacked=True)

    plt.xticks(
        ticks=range(len(df_plot.index)),  # positions
        labels=['Ihssan', 'Hannah'],  # your custom labels
        rotation=45  # optional: rotate labels
    )

    plt.ylabel('Count')
    plt.title('Recurrence Classification per User')
    plt.show()


def plot_bifurcation_rel(df: pd.DataFrame) -> None:

    # Example violin plot
    sns.violinplot(data=df, x='user_id', y='bifurcation_rel')
    plt.axhline(5, color='red', linestyle='--', label='Threshold = 5mm')

    plt.xticks(
        ticks=range(2),  # positions
        labels=['Ihssan', 'Hannah'],  # your custom labels
        rotation=45  # optional: rotate labels
    )

    plt.ylabel('bifurcation_rel distance')
    plt.title('Bifurcation distance to GT per user')
    plt.legend()
    plt.show()


def plot_lymph_node(df: pd.DataFrame) -> None:
    """
    Plot lymph node distances to GT for each user.
    """

    # Group and count
    counts = (
        df.groupby(['user_id', 'lymph_node_abs'])
        .size()
        .unstack(fill_value=0)
    )

    # Reverse order
    counts = counts.iloc[::-1]

    # Plot stacked bar
    counts.plot(kind='bar', stacked=True)

    plt.xticks(
        ticks=range(2),  # positions
        labels=['Ihssan', 'Hannah'],  # your custom labels
        rotation=45  # optional: rotate labels
    )

    plt.ylabel('Count')
    plt.title('Lymph Node Abs (True/False) per User')
    plt.xticks(rotation=0)
    plt.legend(title='lymph_node_abs')
    plt.show()


def plot_lymph_node_time_and_tre(df: pd.DataFrame, user_id: str, sort_by: Literal['tre', 'duration_seconds']) -> None:
    """
    Plot lymph node time and TRE for each user.
    """

    df_lymph = df[
        (df['user_id'] == user_id) &
        (df['task_id'] == 'lymph_node') &
        (df['transform_type'].isin(
            ['TransformType.LINEAR', 'TransformType.NONLINEAR']))
    ].copy()

    # Assign the correct TRE column based on transform_type
    df_lymph['tre'] = df_lymph.apply(
        lambda row: row['lymph_node_tre_linear'] if row['transform_type'] == 'TransformType.LINEAR' else row['lymph_node_tre_nonlinear'],
        axis=1
    )

    # Drop rows with missing data
    df_lymph = df_lymph.dropna(subset=['duration_seconds', 'tre'])

    # Sort by duration_seconds from highest to lowest
    df_lymph = df_lymph.sort_values(
        sort_by, ascending=False).reset_index(drop=True)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Create x-axis positions
    x_positions = range(len(df_lymph))

    # Plot both TRE and duration on the same y-axis
    plt.plot(x_positions, df_lymph['tre'], 'o-', color='tab:red',
             label='TRE (mm)', linewidth=2, markersize=6)
    plt.plot(x_positions, df_lymph['duration_seconds'], 's-',
             color='tab:blue', label='Duration (seconds)', linewidth=2, markersize=6)

    # Set labels and title
    plt.xlabel(f"Cases (sorted by {sort_by}, highest to lowest)")
    plt.ylabel('Value')
    plt.title(f'Lymph Node TRE and Duration for User: {user_id}')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
