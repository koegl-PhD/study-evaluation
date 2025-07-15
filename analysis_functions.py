import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
import pingouin as pg


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
