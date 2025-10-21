import math

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy import stats
import statsmodels.api as sm


def is_point_in_ROI(point: np.ndarray[Any, Any], center: np.ndarray[Any, Any], size: np.ndarray[Any, Any]) -> bool:
    """
    Check if a 3D point is within a rectangular region of interest (ROI).

    :param point: The point to check (3 floats).
    :param center: The center of the ROI (3 floats).
    :param size: The size of the ROI (3 floats).
    :return: True if the point is within the ROI, False otherwise.
    """
    size_half = size / 2.0
    local_point = point - center

    return bool(np.all(np.abs(local_point) <= size_half))


def did_rad_check_recurrence(path_recurrence: str) -> bool:
    """
    Check if the recurrence annotation file exists and is not empty.

    :param path_recurrence: Path to the recurrence annotation file.
    :return: True if the file exists and is not empty, False otherwise.
    """
    try:
        with open(path_recurrence, 'r') as f:
            data = f.read().strip()
            return data == 'True'
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Recurrence annotation file not found at {path_recurrence}")


def df_equal(df1: pd.DataFrame, df2: pd.DataFrame, drop_last_n: int = 0) -> bool:
    """
    Return True if two DataFrames have the same values, dtypes, indices, and columns.
    """

    if drop_last_n > 0:
        df2 = df2.iloc[:, :-drop_last_n]

    try:
        assert_frame_equal(df1, df2, check_dtype=True, check_like=True)
        return True
    except AssertionError:
        return False


def remove_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where 'patient_id' contains 'calibration'.
    """
    return df[~df['patient_id'].str.contains('calibration')].reset_index(drop=True)


def get_calibration_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get rows where 'patient_id' contains 'calibration'.
    """
    return df[df['patient_id'].str.contains('calibration')].reset_index(drop=True)


def apply_corrections(df: pd.DataFrame) -> pd.DataFrame:

    user = "rad_5"

    # where id is rad_5 for task_index = "task_idx_0074" subtract 45s from duration_seconds
    df.loc[(df['user_id'] == user) & (df['task_index'] ==
                                      "task_idx_0074"), 'duration_seconds'] -= 45
    df.loc[(df['user_id'] == user) & (df['task_index'] ==
                                      "task_idx_0170"), 'duration_seconds'] -= 45

    return df


def count_confusion_values(values: Dict[str, str]) -> Dict[str, Dict[str, int]]:

    result = {key: {"Correct": 0, "Incorrect": 0}
              for key in ["NONE", "LINEAR", "NONLINEAR"]}

    for transform, confusion in values.items():

        # result[transform]["TP"] = confusion.count("tp")
        # result[transform]["TN"] = confusion.count("tn")
        # result[transform]["FP"] = confusion.count("fp")
        # result[transform]["FN"] = confusion.count("fn")

        result[transform]["Correct"] = confusion.count(
            "tp") + confusion.count("tn")
        result[transform]["Incorrect"] = confusion.count(
            "fp") + confusion.count("fn")

    return result


def convert_vals_to_percent(values: Dict[str, float]) -> Dict[str, str]:
    """
    Convert values to percentages.
    """
    return {key: f"{val * 100.0:.3g}%" for key, val in values.items()}


def latex_escape(s: str) -> str:
    """Escape underscores for LaTeX."""
    return s.replace("_", r"\_")


def add_arrow(metric: str) -> str:
    """Add up/down arrow depending on metric type."""
    m = metric.lower()
    if "duration" in m or "distance" in m:   # distance & duration
        return latex_escape(metric) + r" $\downarrow$"
    if "accuracy" in m or "correct" in m:  # correctness
        return latex_escape(metric) + r" $\uparrow$"
    if "workflow" in m or "rel" in m:
        return latex_escape(metric) + r" $\downarrow$"
    return latex_escape(metric)


def json_to_latex_tables(data_mean: Dict[str, Any], data_std: Dict[str, Any], float_fmt: str = "{:.3g}") -> Tuple[str, str]:
    """Return (metrics_table, recurrence_table) LaTeX strings from the nested results dict."""
    metric_rows = []  # (task, metric, NONE, LINEAR, NONLINEAR)
    for (task, metrics_mean), (_, metrics_std) in zip(data_mean.items(), data_std.items()):
        for metric, vals in metrics_mean.items():
            if not isinstance(vals, dict):
                continue
            keys = ["NONE", "LINEAR", "NONLINEAR"]
            if all(k in vals for k in keys):
                fmt_vals = []
                for k in keys:
                    v = vals[k]
                    if isinstance(v, (int, float)):
                        if "recurrence" not in metric.lower():
                            fmt_vals.append(
                                f"{float_fmt.format(v)} ± {float_fmt.format(metrics_std[metric][k])}")
                        else:
                            fmt_vals.append(
                                f"{float_fmt.format(v)}")
                    else:
                        fmt_vals.append("-")
                metric_rows.append(
                    (latex_escape(task), latex_escape(metric), *fmt_vals))

    # Build LaTeX for metrics
    metrics_table = r"""
        \begin{table*}[ht]

        \begin{minipage}{\textwidth}

        \captionsetup{width=\textwidth}
        \caption{Per-task metrics (means/proportions) by transform type for experienced radiologists. Values in \textbf{bold} indicate the best performance per row.}
        \label{tab:task_metric_by_transform}

        \centering
        \begin{tabular}{l l r r r}
        \toprule
        Task & Metric & NONE & LINEAR & NONLINEAR \\
        \midrule
        """
    from collections import defaultdict
    by_task: Dict[str, list] = defaultdict(list)
    for row in metric_rows:
        by_task[row[0]].append(row)

    for task in by_task.keys():
        rows = by_task[task]
        for i, (_, metric, n, l, nl) in enumerate(rows):
            task_cell = task if i == 0 else ""

            n_strip = n.split("±")[0].strip() if "±" in n else n
            n_std = n.split("±")[1].strip() if "±" in n else ""
            l_strip = l.split("±")[0].strip() if "±" in l else l
            l_std = l.split("±")[1].strip() if "±" in l else ""
            nl_strip = nl.split("±")[0].strip() if "±" in nl else nl
            nl_std = nl.split("±")[1].strip() if "±" in nl else ""
            values = [float(n_strip), float(l_strip), float(nl_strip)]
            stds = [float(n_std) if n_std else 0.0, float(l_std)
                    if l_std else 0.0, float(nl_std) if nl_std else 0.0]

            m = metric.lower()
            if ("duration" in m or "distance" in m or "rel" in m
                    or "workflow" in m or "z-score" in m):
                best_val = min(values)
            else:
                best_val = max(values)

            best_idxs = [idx for idx, v in enumerate(values) if math.isclose(
                v, best_val, rel_tol=1e-12, abs_tol=0.0)]

            formatted = []
            for j, (val_stripped, std_val) in enumerate(zip(values, stds)):
                if "recurrence" in task.lower() and "Correctness" in metric:
                    val_str = f"{100*val_stripped:.3g}\\%"
                elif "Correctness" in metric:
                    val_str = f"{100*val_stripped:.3g}\\% ± {100*std_val:.3g}\\%"
                else:
                    val_str = f"{val_stripped:.3g} ± {std_val}"
                if j in best_idxs:
                    val_str = r"\textbf{" + val_str + "}"
                formatted.append(val_str)

            metrics_table += (
                f"{task_cell} & {add_arrow(metric)} & "
                f"{formatted[0]} & {formatted[1]} & {formatted[2]} \\\\\n"
            )
        metrics_table += r"\addlinespace" + "\n"

    metrics_table += r"""\bottomrule
        \end{tabular}
        \end{minipage}
        \end{table*}
        """

    return metrics_table


def make_tre_tabular(df: pd.DataFrame, tasks: List[str]) -> str:
    """Build LaTeX tabular (booktabs) summarizing means and Spearman correlations per task. Expects df already filtered/cleaned."""
    results = [t + "_rel" for t in tasks[:-1]] + ["lymph_node_abs"]
    rows = []

    task_replace = {
        "a_vertebralis_r": "A. Vertebralis R.",
        "a_vertebralis_l": "A. Vertebralis L.",
        "a_carotisexterna_r": "A. Carotis Externa R.",
        "a_carotisexterna_l": "A. Carotis Externa L.",
        "lymph_node": "Lymph Node",
        "recurrence": "Recurrence",
    }

    for task, result in zip(tasks, results):
        gt_error = "lymph_node_tre" if "lymph" in task else "bifurcation_tre"
        df_sub = df.loc[df["task_id"] == task, [
            "duration_seconds", result, gt_error]].dropna(subset=[gt_error])
        if df_sub.empty:
            continue
        r_dur = stats.spearmanr(
            df_sub[gt_error], df_sub["duration_seconds"], nan_policy="omit")
        r_err = stats.spearmanr(
            df_sub[gt_error], df_sub[result], nan_policy="omit")
        sig_dur = r"$^{*}$" if (r_dur.pvalue is not None and r_dur.pvalue <
                                0.05) else ""
        sig_err = r"$^{*}$" if (r_err.pvalue is not None and r_err.pvalue <
                                0.05) else ""
        rows.append((
            task_replace.get(task, task),
            f"{r_dur.correlation:.3g} (p={r_dur.pvalue:.3g}){sig_dur}",
            f"{r_err.correlation:.3g} (p={r_err.pvalue:.3g}){sig_err}",
        ))
    header = r"""
                \begin{table*}[ht]

                \begin{minipage}{\textwidth}

                \captionsetup{width=\textwidth}
                \caption{Spearman correlation between the TRE at the location of the tasks and the duration and error for experienced radiologists.}
                \label{tab:task_metric_by_transform}

                \centering
                \begin{tabular}{l r r}
                \toprule
                Task & $\rho$(TRE,Duration) (p) & $\rho$(TRE,Error) (p) \\
                \midrule
            """
    body = "\n".join(
        [f"{t}& {rd} & {re} \\\\" for t, rd, re in rows])

    footer = r"""
                \bottomrule
                \end{tabular}
                \end{minipage}
                \end{table*}
            """
    return header + "\n" + body + footer


def epsilon_squared_kw(H: float, N: int, k: int) -> float:
    """Kruskal–Wallis epsilon squared effect size."""
    return max(0.0, (H - k + 1) / (N - k))


def fit_piecewise_mixed_grid(
    df: pd.DataFrame,
    y_col: str,
    tre_col: str,
    group_col: str,
    candidates_mm: np.ndarray,
) -> dict:
    """Grid-search one-breakpoint mixed model: Y ~ min(TRE,c) + max(0,TRE-c) + (1|group)."""
    best = {"aic": np.inf, "c": None, "params": None, "result": None}
    # linear baseline for ΔAIC
    lin_exog = sm.add_constant(df[[tre_col]])
    lin_mod = sm.MixedLM(endog=df[y_col], exog=lin_exog, groups=df[group_col]).fit(
        reml=True, disp=False)
    for c in candidates_mm:
        x1 = np.minimum(df[tre_col].values, c)
        x2 = np.maximum(0.0, df[tre_col].values - c)
        exog = sm.add_constant(np.column_stack([x1, x2]))
        try:
            res = sm.MixedLM(endog=df[y_col], exog=exog, groups=df[group_col]).fit(
                reml=True, disp=False)
        except Exception:
            continue
        if res.aic < best["aic"]:
            best = {"aic": float(res.aic), "c": float(
                c), "params": res.params.copy(), "result": res}
    if best["result"] is None:
        return {"ok": False}
    p = best["params"]
    pre_slope = float(p[1])
    post_slope = float(p[1] + p[2])
    return {
        "ok": True,
        "breakpoint_mm": best["c"],
        "intercept": float(p[0]),
        "pre_slope_per_mm": pre_slope,
        "post_slope_per_mm": post_slope,
        "aic_piecewise": best["aic"],
        "aic_linear": float(lin_mod.aic),
        "delta_aic": float(lin_mod.aic - best["aic"]),
    }


def fit_piecewise_fe_grid(
    df: pd.DataFrame,
    y_col: str,
    tre_col: str,
    group_col: str,
    candidates_mm: np.ndarray,
) -> Dict[str, Any]:
    """Grid-search 1-breakpoint FE OLS: Y ~ min(TRE,c)+max(0,TRE-c)+task FEs; returns best c and slopes."""
    # Ensure numeric/categorical consistency
    y = pd.to_numeric(df[y_col], errors="coerce")
    tre = pd.to_numeric(df[tre_col], errors="coerce")
    grp = df[group_col].astype(str)

    # Row mask and aligned series
    mask = ~(y.isna() | tre.isna() | grp.isna())
    y, tre, grp = y[mask], tre[mask], grp[mask]

    # Task fixed effects (drop_first to avoid dummy trap)
    dummies = pd.get_dummies(grp, drop_first=True,
                             prefix=group_col).astype(float)

    # ----- Linear FE baseline (for ΔAIC) with aligned indices -----
    base_lin = pd.DataFrame({"const": 1.0, tre_col: tre}, index=tre.index)
    X_lin = base_lin.join(dummies)
    mask_lin = ~X_lin.isna().any(axis=1)
    X_lin = X_lin.loc[mask_lin].astype(float)
    y_lin = y.loc[mask_lin].to_numpy()
    lin_mod = sm.OLS(y_lin, X_lin).fit()

    # ----- Piecewise FE grid search -----
    best_aic: float = float("inf")
    best_c: float | None = None
    best_res = None

    for c in candidates_mm:
        base_pw = pd.DataFrame(
            {
                "const": 1.0,
                "x1": np.minimum(tre.to_numpy(), c),
                "x2": np.maximum(0.0, tre.to_numpy() - c),
            },
            index=tre.index,
        )
        X = base_pw.join(dummies)
        mask_X = ~X.isna().any(axis=1)
        X = X.loc[mask_X].astype(float)
        y_X = y.loc[mask_X].to_numpy()

        try:
            res = sm.OLS(y_X, X).fit()
        except Exception:
            continue

        if res.aic < best_aic:
            best_aic, best_c, best_res = float(res.aic), float(c), res

    if best_res is None or best_c is None:
        return {"ok": False}

    params = best_res.params
    pre_slope = float(params["x1"])
    post_slope = float(params["x1"] + params["x2"])

    return {
        "ok": True,
        "breakpoint_mm": best_c,
        "intercept": float(params["const"]),
        "pre_slope_per_mm": pre_slope,
        "post_slope_per_mm": post_slope,
        "aic_piecewise": best_aic,
        "aic_linear": float(lin_mod.aic),
        "delta_aic": float(lin_mod.aic - best_aic),
    }


def combine_bifurcaitons(df: pd.DataFrame) -> pd.DataFrame:

    rel_cols = [
        "a_vertebralis_r_rel", "a_vertebralis_l_rel",
        "a_carotisexterna_r_rel", "a_carotisexterna_l_rel"
    ]

    abs_cols = [
        "a_vertebralis_r_abs_5", "a_vertebralis_l_abs_5",
        "a_carotisexterna_r_abs_5", "a_carotisexterna_l_abs_5"
    ]

    df["bifurcation_error"] = df[rel_cols].bfill(axis=1).iloc[:, 0]
    df["bifurcation_abs"] = df[abs_cols].bfill(axis=1).iloc[:, 0]

    # find insertion index = position of first of the original columns
    insert_at = df.columns.get_loc(rel_cols[0])

    # remove originals
    df = df.drop(columns=rel_cols + abs_cols)

    # move new columns to original position
    cols = list(df.columns)
    for col in ["bifurcation_abs", "bifurcation_error"][::-1]:
        cols.insert(insert_at, cols.pop(cols.index(col)))

    df = df[cols]

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:

    df_ori = df.copy()

    df = df_ori.copy()

    cols = list(df.columns)

    # switch columns 8 and 9
    cols.insert(8, cols.pop(cols.index("bifurcation_error")))
    cols.insert(10, cols.pop(cols.index("lymph_node_abs")))
    cols.insert(11, cols.pop(cols.index("recurrence")))
    cols.insert(13, cols.pop(cols.index("lymph_node_tre")))

    df = df[cols]


    return df
