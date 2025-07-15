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
