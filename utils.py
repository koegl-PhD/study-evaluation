import math
from typing import Dict, Any, Tuple
from typing import Any, Dict

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


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
    return {key: f"{val * 100.0:.2f}%" for key, val in values.items()}


def latex_escape(s: str) -> str:
    """Escape underscores for LaTeX."""
    return s.replace("_", r"\_")


def add_arrow(metric: str) -> str:
    """Add up/down arrow depending on metric type."""
    m = metric.lower()
    if "duration" in m.lower() or "distance" in m.lower():   # distance & duration
        return latex_escape(metric) + r" $\downarrow$"
    if "accuracy" in m or "correct" in m:  # correctness
        return latex_escape(metric) + r" $\uparrow$"
    return latex_escape(metric)


def json_to_latex_tables(data: Dict[str, Any], float_fmt: str = "{:.2f}") -> Tuple[str, str]:
    """Return (metrics_table, recurrence_table) LaTeX strings from the nested results dict."""
    metric_rows = []  # (task, metric, NONE, LINEAR, NONLINEAR)
    for task, metrics in data.items():
        for metric, vals in metrics.items():
            if task == "recurrence" and metric == "recurrence":
                continue
            if not isinstance(vals, dict):
                continue
            keys = ["NONE", "LINEAR", "NONLINEAR"]
            if all(k in vals for k in keys):
                fmt_vals = []
                for k in keys:
                    v = vals[k]
                    if isinstance(v, bool):
                        fmt_vals.append(float_fmt.format(1.0 if v else 0.0))
                    elif isinstance(v, (int, float)):
                        fmt_vals.append(float_fmt.format(v))
                    else:
                        fmt_vals.append("-")
                metric_rows.append(
                    (latex_escape(task), latex_escape(metric), *fmt_vals))

    # Build LaTeX for metrics
    metrics_table = r"""\begin{table*}[ht]
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

    for task in sorted(by_task.keys()):
        rows = by_task[task]
        for i, (_, metric, n, l, nl) in enumerate(rows):
            task_cell = task if i == 0 else ""

            # convert back to floats for comparison
            values = [float(n), float(l), float(nl)]

            # decide whether smaller or larger is better
            m = metric.lower()
            if "duration" in m or "distance" in m or "rel" in m:
                best_idx = values.index(min(values))   # smaller is better
            else:
                best_idx = values.index(max(values))   # larger is better

            # rebuild values with bold for best
            formatted = []
            for j, val in enumerate(values):
                if "Correctness" in metric:
                    val_str = f"{100*val:.0f}\\%"
                else:
                    val_str = f"{val:.2f}"
                if j == best_idx:
                    val_str = r"\textbf{" + val_str + "}"
                formatted.append(val_str)

            metrics_table += (
                f"{task_cell} & {add_arrow(metric)} & "
                f"{formatted[0]} & {formatted[1]} & {formatted[2]} \\\\\n"
            )
        metrics_table += r"\addlinespace" + "\n"

    metrics_table += r"""\bottomrule
        \end{tabular}
        \caption{Per-task metrics (means/proportions) by transform type.}
        \end{table*}
        """

    # Build LaTeX for recurrence (Correct/Incorrect/Accuracy only)
    rec = data.get("recurrence", {}).get("recurrence", {})
    rec_rows = []
    for tt in ["NONE", "LINEAR", "NONLINEAR"]:
        block = rec.get(tt, {})
        corr = int(block.get("Correct", 0))
        inc = int(block.get("Incorrect", 0))
        total = corr + inc
        acc = (corr / total) if total > 0 else math.nan
        rec_rows.append((tt, corr, inc, acc))

    recurrence_table = r"""\begin{table}[ht]
        \centering
        \begin{tabular}{l r r r}
        \toprule
        Transform & Correct & Incorrect & Accuracy \\
        \midrule
        """
    for tt, c, i, a in rec_rows:
        acc_str = "-" if math.isnan(a) else f"{a*100:.1f}\\%"
        recurrence_table += f"{tt} & {c} & {i} & {acc_str} \\\\\n"

    recurrence_table += r"""\bottomrule
        \end{tabular}
        \caption{Recurrence results by transform type (counts and accuracy).}
        \end{table}
        """

    return metrics_table, recurrence_table
