import json
import math

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

import utils

try:
    import scikit_posthocs as sp
    _HAS_SPKH = True
except Exception:
    _HAS_SPKH = False
    from scipy.stats import mannwhitneyu


NestedDict = Dict[str, Union[float, "NestedDict"]]


def average_nested_dicts(d1: NestedDict, d2: NestedDict) -> NestedDict:
    """Recursively average values of two nested dicts with same structure."""
    result: NestedDict = {}
    for k in d1:
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            result[k] = average_nested_dicts(v1, v2)
        else:
            result[k] = (v1 + v2) / 2
    return result


def _pairwise_dunn_holm(groups: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
    """Pairwise MWU p-values with Holm correction for (NONE, LINEAR, NONLINEAR)."""
    keys = ["NONE", "LINEAR", "NONLINEAR"]
    data = [np.asarray(groups[k], float) for k in keys]
    pairs = [(keys[i], keys[j]) for i in range(3) for j in range(i + 1, 3)]

    raw_ps = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        u = mannwhitneyu(data[i], data[j],
                         alternative="two-sided", method="asymptotic")
        raw_ps.append(u.pvalue)

    _, adj, _, _ = multipletests(raw_ps, method="holm")
    return {pairs[k]: float(adj[k]) for k in range(3)}


def run_kw_dunn_for_metric(df_task: pd.DataFrame, metric_col: str) -> Tuple[float, Dict[Tuple[str, str], float]]:
    """Run Kruskal–Wallis across 3 transforms, then Dunn–Holm pairwise."""
    groups = {
        "NONE": df_task.loc[df_task["transform_type"] == "NONE", metric_col].dropna().values,
        "LINEAR": df_task.loc[df_task["transform_type"] == "LINEAR", metric_col].dropna().values,
        "NONLINEAR": df_task.loc[df_task["transform_type"] == "NONLINEAR", metric_col].dropna().values,
    }
    if any(len(v) == 0 for v in groups.values()):
        return float("nan"), {(a, b): float("nan") for a in groups for b in groups if a < b}

    kw = kruskal(groups["NONE"], groups["LINEAR"],
                 groups["NONLINEAR"], nan_policy="omit")
    pairwise = _pairwise_dunn_holm(groups) if np.isfinite(kw.pvalue) else {
        (a, b): float("nan") for a in groups for b in groups if a < b}
    return float(kw.pvalue), pairwise


def significance_to_latex(df: pd.DataFrame, caption: str = "Pairwise significance results", label: str = "tab:significance") -> str:
    """Convert significance DataFrame to LaTeX table for appendix."""
    df_fmt = df.copy()

    # format numeric p-values
    def fmt_p(p: float) -> str:
        if pd.isna(p):
            return "-"
        if p < 1e-3:
            return f"\\textbf{{{p:.1e}}}"
        elif p < 0.05:
            return f"\\textbf{{{p:.3f}}}"
        else:
            return f"{p:.3f}"

    df_fmt["kw_p"] = df_fmt["kw_p"].apply(fmt_p)
    df_fmt["p_NONE_vs_LINEAR"] = df_fmt["p_NONE_vs_LINEAR"].apply(fmt_p)
    df_fmt["p_NONE_vs_NONLINEAR"] = df_fmt["p_NONE_vs_NONLINEAR"].apply(fmt_p)
    df_fmt["p_LINEAR_vs_NONLINEAR"] = df_fmt["p_LINEAR_vs_NONLINEAR"].apply(
        fmt_p)

    # rename columns for LaTeX
    df_fmt = df_fmt.rename(columns={
        "task": "Task",
        "metric_raw": "Metric",
        "kw_p": "Kruskal–Wallis $p$",
        "p_NONE_vs_LINEAR": "$p_{None, Rigid}$",
        "p_NONE_vs_NONLINEAR": "$p_{None, Deform.}$",
        "p_LINEAR_vs_NONLINEAR": "$p_{Rigid, Deform.}$",
    })

    # convert to LaTeX string
    latex = df_fmt.to_latex(
        index=False,
        escape=False,
        column_format="llcccc",
        caption=caption,
        label=label,
        longtable=False,
    )

    latex = latex.split("\n")

    for idx, row in enumerate(latex):
        row = row.replace("a_vertebralis_r", "A. Vertebralis R.")
        row = row.replace("a_vertebralis_l", "A. Vertebralis L.")
        row = row.replace(
            "a_carotisexterna_r", "A. Carotis E. R.")
        row = row.replace(
            "a_carotisexterna_l", "A. Carotis E. L.")
        row = row.replace("lymph_node", "Lymph Node")
        row = row.replace(" recurrence ", " Recurrence ")

        row = row.replace("bifurcation_error", "Matching error")
        row = row.replace("duration_seconds", "Duration")
        row = row.replace("z_score", "Workflow load")
        row = row.replace("recurrence_abs", "Detection rate")
        row = row.replace("abs", "Localization rate")

        row = row.replace("recurrence", "Recurrence")

        latex[idx] = row

    latex[0] = latex[0].replace("table", "table*")
    latex[-2] = latex[-2].replace("table", "table*")

    latex[7] = latex[7].replace(
        "A. Vertebralis R.", "\\multirow{3}{*}{A. Vertebralis R.}")
    latex[8] = latex[8].replace("A. Vertebralis R.", " ")
    latex[9] = latex[9].replace("A. Vertebralis R.", " ")
    latex.insert(10, '\\addlinespace')

    latex[11] = latex[11].replace(
        "A. Vertebralis L.", "\\multirow{3}{*}{A. Vertebralis L.}")
    latex[12] = latex[12].replace("A. Vertebralis L.", " ")
    latex[13] = latex[13].replace("A. Vertebralis L.", " ")
    latex.insert(14, '\\addlinespace')

    latex[15] = latex[15].replace(
        "A. Carotis E. R.", "\\multirow{3}{*}{A. Carotis E. R.}")
    latex[16] = latex[16].replace("A. Carotis E. R.", " ")
    latex[17] = latex[17].replace("A. Carotis E. R.", " ")
    latex.insert(18, '\\addlinespace')

    latex[19] = latex[19].replace(
        "A. Carotis E. L.", "\\multirow{3}{*}{A. Carotis E. L.}")
    latex[20] = latex[20].replace("A. Carotis E. L.", " ")
    latex[21] = latex[21].replace("A. Carotis E. L.", " ")
    latex.insert(22, '\\addlinespace')

    latex.insert(26, '\\addlinespace')

    latex.insert(19, '\\midrule')
    latex.insert(23, '\\midrule')

    return latex


def calculate_standard_error(s: np.ndarray) -> float:
    p = s.mean()
    n = s.shape[0]

    se = math.sqrt(p * (1 - p) / n)

    return se


def main():
    # Load the results
    df = pd.read_csv('outputs/results.csv')

    participants: Dict[str, Dict[str, int | bool | str]
                       ] = json.load(open('resources/participants.json'))
    task_metric_map: Dict[str, List[str]] = json.load(
        open('resources/task_metric_map.json'))
    tasks = json.load(open('resources/tasks.json'))

    df = utils.remove_calibration(df)

    df = utils.compute_workflow_z(df)

    means_all = []
    stds_all = []

    for experienced in [True]:  # [True, False]:

        # keep only (in)experienced radiologists
        # df_new = df[df['user_id'].isin(
        # [uid for uid, info in participants.items() if info['experienced'] == experienced])].reset_index(drop=True)
        df_new = df.copy()
        # remove all rows where transform_type is not NONE and where synchronised_count is 0
        df_new = df_new[~((df_new['transform_type'] != 'TransformType.NONE') & (
            df_new['synchronised_count'] == 0))].reset_index(drop=True)

        # Clean transform_type labels
        df_new["transform_type"] = df_new["transform_type"].str.replace(
            "TransformType.", "")
        # Extract numeric task index for sorting
        df_new["task_idx_num"] = df_new["task_index"].str.replace(
            "task_idx_", "").astype(int)

        # --- Significance testing (Kruskal–Wallis + Dunn–Holm) ---
        sig_rows: List[Dict[str, Union[str, float]]] = []

        for task in tasks:
            df_task = df_new[df_new["task_id"] == task].reset_index(drop=True)

            metric_cols: List[str] = task_metric_map[task] + \
                task_metric_map["common"] + ["z_score"]
            for metric_col in metric_cols:
                kw_p, pair_ps = run_kw_dunn_for_metric(df_task, metric_col)
                sig_rows.append({
                    "task": task,
                    "metric_raw": metric_col,
                    "kw_p": kw_p,
                    "p_NONE_vs_LINEAR": pair_ps.get(("NONE", "LINEAR"), np.nan),
                    "p_NONE_vs_NONLINEAR": pair_ps.get(("NONE", "NONLINEAR"), np.nan),
                    "p_LINEAR_vs_NONLINEAR": pair_ps.get(("LINEAR", "NONLINEAR"), np.nan),
                })

        sig_df = pd.DataFrame(sig_rows)
        sig_df.to_csv("outputs/new/significance_all.csv", index=False)

        sig_latex = significance_to_latex(
            sig_df, caption=f"Pairwise significance results for all radiologists", label=f"tab:significance_none_rigid_deformable")

        with open(f"outputs/new/significance_table.tex", "w") as f:
            f.write("\n".join(sig_latex))

        # --- End significance testing ---

        means = {}
        stds = {}

        for task in tasks:

            df_filtered = df_new[df_new['task_id']
                                 == task].reset_index(drop=True)

            # accumulators for combined workflow load z-score
            wl_vals = {"NONE": [], "LINEAR": [], "NONLINEAR": []}
            wl_std = None
            wl_count = 0

            for res in task_metric_map[task] + task_metric_map["common"] + task_metric_map["workflow"]:

                std = None
                if task == "recurrence" and res == "recurrence":
                    temp1 = df_filtered.groupby("transform_type")[res].sum()
                    temp1 = utils.count_confusion_values(temp1.to_dict())

                    temp = {
                        "NONE": temp1["NONE"]["Correct"] / (temp1["NONE"]["Correct"] + temp1["NONE"]["Incorrect"]),
                        "LINEAR": temp1["LINEAR"]["Correct"] / (temp1["LINEAR"]["Correct"] + temp1["LINEAR"]["Incorrect"]),
                        "NONLINEAR": temp1["NONLINEAR"]["Correct"] / (temp1["NONLINEAR"]["Correct"] + temp1["NONLINEAR"]["Incorrect"]),
                    }

                    # zero std
                    std = {"NONE": 0.0, "LINEAR": 0.0, "NONLINEAR": 0.0}
                elif res.endswith("_abs_5"):
                    temp = df_filtered.groupby("transform_type")[res].mean()
                    temp = utils.convert_vals_to_percent(temp.to_dict())
                    std = {}
                    std["NONE"] = df_filtered[df_filtered["transform_type"] ==
                                              "NONE"][res].std()
                    std["LINEAR"] = df_filtered[df_filtered["transform_type"] ==
                                                "LINEAR"][res].std()
                    std["NONLINEAR"] = df_filtered[df_filtered["transform_type"] ==
                                                   "NONLINEAR"][res].std()
                elif res in task_metric_map["workflow"]:
                    temp = df_filtered.groupby("transform_type")[res].mean().reindex([
                        "NONE", "LINEAR", "NONLINEAR"])
                    vals = temp.values.astype(float)

                    # z-scores across transform types (handle zero variance)
                    std_across_types = np.nanstd(vals, ddof=0)
                    zscores = np.zeros_like(vals) if not np.isfinite(
                        std_across_types) or std_across_types == 0 else (vals - np.nanmean(vals)) / std_across_types
                    zdict = dict(
                        zip(["NONE", "LINEAR", "NONLINEAR"], map(float, zscores)))

                    for k in wl_vals:
                        wl_vals[k].append(zdict[k])
                    wl_count += 1

                    continue
                else:
                    temp = df_filtered.groupby("transform_type")[res].mean()

                    # change order to "NONE", "LINEAR", "NONLINEAR
                    temp = temp.reindex(["NONE", "LINEAR", "NONLINEAR"])
                    temp = temp.to_dict()
                    std = {}

                    if (task.lower() == 'lymph_node' or task.lower() == 'recurrence') and (res.lower().find('abs') != -1 or res.lower().find('recurrence') != -1):
                        std_none = calculate_standard_error(
                            df_filtered[df_filtered["transform_type"] == "NONE"][res].astype(int))
                        std_linear = calculate_standard_error(
                            df_filtered[df_filtered["transform_type"] == "LINEAR"][res].astype(int))
                        std_nonlinear = calculate_standard_error(
                            df_filtered[df_filtered["transform_type"] == "NONLINEAR"][res].astype(int))
                    else:
                        std_none = df_filtered[df_filtered["transform_type"] ==
                                               "NONE"][res].std()
                        std_linear = df_filtered[df_filtered["transform_type"] ==
                                                 "LINEAR"][res].std()
                        std_nonlinear = df_filtered[df_filtered["transform_type"] ==
                                                    "NONLINEAR"][res].std()
                    std["NONE"] = std_none
                    std["LINEAR"] = std_linear
                    std["NONLINEAR"] = std_nonlinear

                # dict of dicts
                if task not in means:
                    means[task] = {}
                if task not in stds:
                    stds[task] = {}

                if "duration" in res:
                    res = "Duration (s)"
                elif "bifurcation_error" in res:
                    res = "Distance (mm)"
                elif "abs" in res or "recurrence" in res.lower():
                    res = "Correctness (\\%)"

                means[task][res] = temp
                stds[task][res] = std

            # add combined workflow load (average of per-metric z-scores)
            if any(len(v) for v in wl_vals.values()):
                means[task]["Workflow load (z-score)"] = {k: float(
                    np.mean(wl_vals[k])) if wl_vals[k] else float("nan") for k in wl_vals}
                stds[task]["Workflow load (z-score)"] = {k: float(
                    np.std(wl_vals[k], ddof=0)) if wl_vals[k] else 0.0 for k in wl_vals}

        x = 0

        means = {
            "A. Vertebralis R.": means["a_vertebralis_r"],
            "A. Vertebralis L.": means["a_vertebralis_l"],
            "A. Carotis Externa R.": means["a_carotisexterna_r"],
            "A. Carotis Externa L.": means["a_carotisexterna_l"],
            "Lymph Node": means["lymph_node"],
            "Recurrence": means["recurrence"],
        }
        stds = {
            "A. Vertebralis R.": stds["a_vertebralis_r"],
            "A. Vertebralis L.": stds["a_vertebralis_l"],
            "A. Carotis Externa R.": stds["a_carotisexterna_r"],
            "A. Carotis Externa L.": stds["a_carotisexterna_l"],
            "Lymph Node": stds["lymph_node"],
            "Recurrence": stds["recurrence"],
        }

        means_all.append(means)
        stds_all.append(stds)

    # means = average_nested_dicts(means_all[0], means_all[1])
    # stds = average_nested_dicts(stds_all[0], stds_all[1])

    # metrics_tex = utils.json_to_latex_tables(means_all[1], stds_all[1])
    # with open("outputs/new/metrics_table_inexperienced.tex", "w") as f:
    #     f.write(metrics_tex)
    # metrics_tex = utils.json_to_latex_tables(means_all[0], stds_all[0])
    # with open("outputs/new/metrics_table_experienced.tex", "w") as f:
    #     f.write(metrics_tex)
    metrics_tex = utils.json_to_latex_tables(means_all[0], stds_all[0])
    with open("outputs/new/metrics_table_all.tex", "w") as f:
        f.write(metrics_tex)

    x = 0


if __name__ == "__main__":
    main()
