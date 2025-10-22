from typing import Any, Dict, List, Tuple
import warnings
import math

from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import numpy as np
from scipy import stats
import statsmodels.api as sm  # add this import

import utils


def summarize_model_base(model: Any, name: str, predictor: str, response: str) -> None:
    """Concise mixed model summary with % and variance components."""
    return
    coef = model.params[predictor]
    se = model.bse[predictor]
    pval = model.pvalues[predictor]
    ci_low, ci_high = model.conf_int().loc[predictor]

    pct_change = (np.exp(coef) - 1) * 100
    ci_low_pct = (np.exp(ci_low) - 1) * 100
    ci_high_pct = (np.exp(ci_high) - 1) * 100

    var_radiologist = float(np.asarray(model.cov_re)[
                            0, 0]) if model.cov_re is not None else np.nan

    vcomp_names = getattr(model.model, "vcomp_names", None)
    if vcomp_names is not None and len(model.vcomp) == len(vcomp_names):
        vcomp_map = dict(zip(vcomp_names, model.vcomp))
        var_case = float(vcomp_map.get("case", np.nan))
    else:
        var_case = float(model.vcomp[0]) if len(model.vcomp) == 1 else np.nan

    residual_var = float(model.scale)

    print(f"\n📊 {name}")
    print(f"Fixed effect ({predictor}): {coef:.3f} ± {se:.3f}, p={pval:.3e}")
    print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}] → ~{pct_change:.1f}% change "
          f"(95% CI [{ci_low_pct:.1f}%, {ci_high_pct:.1f}%])")
    # every time tre increases by 2.718 (e), the outcome changes by pct_change % (e^coef)
    # or
    print(
        f"With 95 % confidence, every 1-unit increase in {predictor} raises the expected {response} by somewhere in [ci_low, mean, ci_high] [{ci_low_pct:.1f}%, {pct_change:.1f}%, {ci_high_pct:.1f}%]. p={pval:.3e}")

    total_var = var_radiologist + var_case + residual_var
    print("\nRandom-effect variances:")
    print(
        f"  Radiologist: {var_radiologist:.3f} ({var_radiologist/total_var*100:.1f}%)")
    print(f"  Case:        {var_case:.3f} ({var_case/total_var*100:.1f}%)")
    print(
        f"  Residual:    {residual_var:.3f} ({residual_var/total_var*100:.1f}%)")
    # Each random-effect variance (like 0.019, 0.287, 0.162) measures how much variability there is in the model’s predicted values that comes from
    #   - radiologist (group variance)
    #   - case (variance component)
    #   - pure residual noise (residual variance)
    print(f"{var_radiologist/total_var*100:.1f}% of the variance in {response} is due to differences between radiologists.")
    print(f"{var_case/total_var*100:.1f}% of the variance in {response} is due to differences between cases.")
    print(f"{residual_var/total_var*100:.1f}% of the variance in {response} is unexplained (residual).")


def summarize_model_all_tasks(model: Any, name: str, predictor: str, response: str) -> None:
    """Summarize mixed model with tasks: global effect, per-task slopes if interactions, and variance components."""
    params = model.params
    bse = model.bse
    pvals = model.pvalues
    conf = model.conf_int()
    cov = model.cov_params()

    def pct(e: float) -> float:
        return (np.exp(e) - 1.0) * 100.0

    print(f"\n📊 {name}")

    # 1) Global fixed effect (if present)
    if predictor in params.index:
        coef = float(params[predictor])
        se = float(bse[predictor])
        pval = float(pvals[predictor])
        lo, hi = map(float, conf.loc[predictor])
        print(
            f"Global fixed effect ({predictor}): {coef:.3f} ± {se:.3f}, p={pval:.3e} "
            f"→ ~{pct(coef):.1f}% (95% CI [{pct(lo):.1f}%, {pct(hi):.1f}%])"
        )
    else:
        coef = 0.0  # reference slope will be built only from interactions
        print(
            f"No global {predictor} main effect term (only task-specific interactions).")

    # 2) Detect task interactions and compute per-task slopes
    # Handles both "predictor:C(task_id)[T.xxx]" and "C(task_id)[T.xxx]:predictor"
    inter_terms = [
        k for k in params.index if predictor in k and "C(" in k and ":" in k]
    if inter_terms:
        # Extract task levels robustly
        def extract_task(term: str) -> str:
            if "[T." in term and "]" in term:
                start = term.index("[T.") + 3
                end = term.index("]", start)
                return term[start:end]
            # Fallback: try last bracketed section
            if "[" in term and "]" in term:
                return term[term.rindex("[")+1:term.rindex("]")]
            return term

        task_levels = sorted({extract_task(t) for t in inter_terms})

        rows: List[Tuple[str, float, float, float, float]] = []
        for t in task_levels:
            # Find the exact interaction key (either order)
            key_a = f"{predictor}:C(task_id)[T.{t}]"
            key_b = f"C(task_id)[T.{t}]:{predictor}"
            inter_key = key_a if key_a in params.index else (
                key_b if key_b in params.index else None)
            if inter_key is None:
                continue

            b_inter = float(params[inter_key])
            # Per-task slope = global + interaction (if global exists)
            b_task = coef + b_inter

            # SE for linear combination
            var = 0.0
            if predictor in cov.index:
                var += float(cov.loc[predictor, predictor])
                if inter_key in cov.index:
                    var += float(cov.loc[inter_key, inter_key]) + \
                        2.0 * float(cov.loc[predictor, inter_key])
            else:
                # No global term → variance comes from interaction only
                var += float(cov.loc[inter_key, inter_key])
            se_task = float(np.sqrt(max(var, 0.0)))

            # Wald p-value for per-task slope
            z = b_task / se_task if se_task > 0 else np.nan
            p_task = float(
                2.0 * (1.0 - 0.5 * (1 + np.math.erf(abs(z) / np.sqrt(2))))) if np.isfinite(z) else np.nan

            # 95% CI
            lo_task = b_task - 1.96 * se_task
            hi_task = b_task + 1.96 * se_task

            rows.append((t, b_task, se_task, lo_task, hi_task))

        if rows:
            df_tasks = pd.DataFrame(
                rows, columns=["task_id", "coef", "se", "ci_low", "ci_high"])
            df_tasks["pct"] = df_tasks["coef"].apply(pct)
            df_tasks["pct_low"] = df_tasks["ci_low"].apply(pct)
            df_tasks["pct_high"] = df_tasks["ci_high"].apply(pct)
            # Print compact table
            print("\nPer-task slopes (effect of predictor on response):")
            for _, r in df_tasks.iterrows():
                print(
                    f"  {r['task_id']}: {r['coef']:.3f} ± {r['se']:.3f} "
                    f"→ ~{r['pct']:.1f}% (95% CI [{r['pct_low']:.1f}%, {r['pct_high']:.1f}%])"
                )
        else:
            print("\nNo task interaction terms detected after parsing.")
    else:
        print("\nNo task interactions in the fixed-effects part; single global slope applies across tasks.")

    # 3) Variance components (radiologist, case, task if present)
    var_radiologist = float(np.asarray(model.cov_re)[
                            0, 0]) if model.cov_re is not None else np.nan
    vcomp_names = getattr(model.model, "vcomp_names", None)
    var_case = np.nan
    var_task = np.nan
    if vcomp_names is not None and len(model.vcomp) == len(vcomp_names):
        vmap: Dict[str, float] = dict(
            zip(vcomp_names, map(float, model.vcomp)))
        var_case = vmap.get("case", np.nan)
        var_task = vmap.get("task", np.nan)
    residual_var = float(model.scale)

    parts = [v for v in [var_radiologist, var_case,
                         var_task, residual_var] if np.isfinite(v)]
    total = float(sum(parts)) if parts else np.nan

    def share(v: float) -> float:
        return (v / total * 100.0) if (np.isfinite(v) and np.isfinite(total) and total > 0) else np.nan

    print("\nRandom-effect variances:")
    print(
        f"  Radiologist: {var_radiologist:.3f} ({share(var_radiologist):.1f}%)")
    if np.isfinite(var_case):
        print(f"  Case:        {var_case:.3f} ({share(var_case):.1f}%)")
    if np.isfinite(var_task):
        print(f"  Task:        {var_task:.3f} ({share(var_task):.1f}%)")
    print(f"  Residual:    {residual_var:.3f} ({share(residual_var):.1f}%)")


def is_complex_model_better(base_model: Any, slope_model: Any) -> bool:
    """
    Compare two nested mixed models using LRT and AIC.
    Returns True if slope_model fits significantly better, else False.
    """
    # Likelihood Ratio Test
    lr_stat = 2 * (slope_model.llf - base_model.llf)
    p_lr = stats.chi2.sf(lr_stat, df=1)

    # AIC difference (negative means slope model is better)
    aic_diff = slope_model.aic - base_model.aic

    # print(f"LRT χ²={lr_stat:.2f}, p={p_lr:.3f}")
    # print(f"AIC base={base_model.aic:.1f}, slope={slope_model.aic:.1f} (ΔAIC={aic_diff:.1f})")

    # Decision logic
    if p_lr < 0.05 and aic_diff < -2:
        # print("Slope model significantly better.")
        return True
    else:
        # print("No significant improvement — keep base model.")
        return False


def _pct_change_from_coef(coef: float) -> float:
    return (np.exp(coef) - 1.0) * 100.0


def _extract_effect(model: Any, predictor: str) -> Tuple[float, float, float, float]:
    """
    Return (% change, % CI low, % CI high, p-value) for predictor.
    """
    beta = float(model.params[predictor])
    pval = float(model.pvalues[predictor])
    ci_low, ci_high = map(float, model.conf_int().loc[predictor])
    pct = _pct_change_from_coef(beta)
    pct_low = _pct_change_from_coef(ci_low)
    pct_high = _pct_change_from_coef(ci_high)
    return pct, pct_low, pct_high, pval


def _fmt_pct_ci(pct: float, lo: float, hi: float, negate: bool = False) -> str:
    # +314 % (148–590 %)
    if negate:
        return f"{-pct:.0f}\\% [{-hi:.0f}--{-lo:.0f}\\%]"
    return f"{pct:+.0f}\\% [{lo:.0f}--{hi:.0f}\\%]"


def _fmt_p_latex(p: float) -> str:
    # "< 10^{-3}" for very small p, else "x.x×10^{-k}" in math mode, else 3 sig figs
    # if p < 1e-3:
    #     return "$<10^{-3}$"
    exp = int(math.floor(math.log10(p))) if p > 0 else 0
    if p < 0.1:
        mant = p / (10 ** exp)
        return f"${mant:.2f}\\times10^{{{exp}}}$"
    return f"{p:.3f}"


def build_latex_table_duration(
    per_task: List[Tuple[str, float, float, float, float]],
    combined: Tuple[float, float, float, float],
    extra_rows: List[Tuple[str, float, float, float, float]],
    predictor: str,
    output_var: str = "duration"
) -> str:
    """LaTeX table with per-task % change (95% CI) & p, a 4-row-spanning combined column, plus extra rows (e.g., lymph/recurrence)."""
    order = ["a_vertebralis_r", "a_vertebralis_l",
             "a_carotisexterna_r", "a_carotisexterna_l"]
    if "lymph_node" in [t[0] for t in per_task]:
        order.append("lymph_node")
    if "recurrence" in [t[0] for t in per_task]:
        order.append("recurrence")
    pt_map: Dict[str, Tuple[float, float, float, float]] = {
        t: (pct, lo, hi, p) for t, pct, lo, hi, p in per_task
    }
    rows = [(t, *pt_map[t]) for t in order if t in pt_map]
    comb_pct, comb_lo, comb_hi, comb_p = combined

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{Effect of registration accuracy (log({predictor})) on task {output_var}. Entries show percent change per log-unit with 95\\% CI and p-values. Combined columns report a pooled model over vertebralis/carotis tasks.}}")
    lines.append(f"  \\label{{tab:duration_effects_{predictor}_{output_var}}}")
    lines.append("  \\begin{tabular}{@{}lcccc@{}}")
    lines.append("    \\toprule")
    lines.append(
        "    \\textbf{Task} & \\textbf{\\% change (95\\% CI)} & \\textbf{p} & \\multicolumn{2}{c}{\\textbf{Combined (vertebralis + carotis)}}\\\\")
    lines.append("    \\cmidrule{4-5}")
    lines.append("    & & & \\% change (95\\% CI) & p \\\\")
    lines.append("    \\midrule")
    for i, (task, pct, lo, hi, p) in enumerate(rows):
        if i == 0:
            lines.append(
                f"    {task} & {_fmt_pct_ci(pct, lo, hi)} & {_fmt_p_latex(p)} & "
                f"\\multirow{{4}}{{*}}{{{_fmt_pct_ci(comb_pct, comb_lo, comb_hi)}}} & "
                f"\\multirow{{4}}{{*}}{{{_fmt_p_latex(comb_p)}}} \\\\"
            )
        else:
            if "lymph_node" in task:
                lines.append("    \\addlinespace")
            if "recurrence" in task:
                lines.append(
                    f"    {task} & {_fmt_pct_ci(pct, lo, hi, negate=True)} & {_fmt_p_latex(p)} & & \\\\")
            else:
                lines.append(
                    f"    {task} & {_fmt_pct_ci(pct, lo, hi)} & {_fmt_p_latex(p)} & & \\\\")
    if extra_rows:
        lines.append("    \\addlinespace")
        for label, pct, lo, hi, p in extra_rows:
            lines.append(
                f"    {label} & {_fmt_pct_ci(pct, lo, hi)} & {_fmt_p_latex(p)} & & \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main() -> None:

    df = pd.read_csv('outputs/results.csv')
    df = utils.remove_calibration(df)

    df = df[df["transform_type"] !=
            "TransformType.NONE"].reset_index(drop=True)

    # remove rows where synchronised_duration_seconds is 0.0
    df = df[df["synchronised_duration_seconds"] != 0.0].reset_index(drop=True)

    task_subsets = [
        "a_vertebralis_r",
        "a_vertebralis_l",
        "a_carotisexterna_r",
        "a_carotisexterna_l",
        "lymph_node",
        "recurrence",
        ["a_vertebralis_r", "a_vertebralis_l",
            "a_carotisexterna_r", "a_carotisexterna_l"]
    ]

    print()
    print()

    # remove last 20 columns from df
    df = df.iloc[:, :-20]

    # show all columns
    pd.set_option('display.max_columns', None)
    # print all columns in one line
    pd.set_option('display.width', 1000)

    # 1. Check distributions -  all are non normal and heavily left sewed
    """
    for col in ['dsc', 'bifurcation_error', 'duration_seconds', 'tre']:
        sns.histplot(df[col], kde=True)
        plt.title(col)
        plt.show()
        stat, p = stats.shapiro(df[col].dropna())
        print(f"{col}: Shapiro p={p:.4f}")
    """

    # 2. log-transform (since right-skewed)
    df['log_tre'] = np.log1p(df['tre'])
    df['log_dsc'] = np.log1p(df['dsc'].max() - df['dsc'])
    df['log_bif_error'] = np.log1p(df['bifurcation_error'])
    df['log_duration'] = np.log1p(df['duration_seconds'])

    # remvoe uncessecasry tasks
    df_ori = df.copy()
    for predictor in ['log_tre', 'log_dsc']:
        for output_var in ['log_duration', 'log_bif_error']:
            per_task_duration: List[Tuple[str, float, float]] = []
            combined_duration: Tuple[float, float] = (
                float("nan"), float("nan"))
            print()
            for task_subset in task_subsets:
                if not isinstance(task_subset, list) and task_subset == "recurrence" and predictor == "log_tre":
                    continue  # skip tre+recurrence (only dsc makes sense)
                if not isinstance(task_subset, list) and task_subset == "recurrence" and output_var == "log_bif_error":
                    # skip bifurcation+recurrence (only duration makes sense)
                    continue
                if not isinstance(task_subset, list) and task_subset == "lymph_node" and output_var == "log_bif_error":
                    # skip bifurcation+lymph node (only duration makes sense)
                    continue

                # print()
                # print(f"=== Analyzing task: {task_subset} ===")
                task_subset = [task_subset] if isinstance(
                    task_subset, str) else task_subset

                df = df_ori.copy()
                df = df[df["task_id"].isin(task_subset)].reset_index(drop=True)

                # 3. Mixed-effects models with radiologist + case random intercepts
                # (A) Duration model
                if len(task_subset) == 1:
                    model_dur = smf.mixedlm(
                        f"{output_var} ~ {predictor}",
                        data=df,
                        groups=df["user_id"],
                        re_formula="~1",
                        vc_formula={"case": "0 + C(patient_id)"}
                    ).fit(reml=False)
                    # check if residual is bell shaped
                    plt.close()
                    sns.histplot(model_dur.resid, kde=True)
                    summarize_model_base(model_dur, "Task Duration (log_duration)",
                                         predictor, "duration seconds")
                    pct, lo, hi, p = _extract_effect(model_dur, predictor)
                    per_task_duration.append((task_subset[0], pct, lo, hi, p))
                else:
                    model_acc_all = smf.mixedlm(
                        f"{output_var} ~ {predictor}",
                        data=df,
                        groups=df["user_id"],
                        re_formula="~1",
                        vc_formula={
                            "case": "0 + C(patient_id)",
                            "task": "0 + C(task_id)"
                        }
                    ).fit(reml=False)
                    plt.close()
                    sns.histplot(model_acc_all.resid, kde=True)
                    summarize_model_all_tasks(model_acc_all, "Radiologist Duration Full Model",
                                              predictor, "bifurcation error")
                    comb_pct, comb_lo, comb_hi, comb_p = _extract_effect(
                        model_acc_all, predictor)
                    combined_duration = (comb_pct, comb_lo, comb_hi, comb_p)

            # lymph node detection (boolean column: 'abs')
            task_subset = [
                "lymph_node",
            ]
            df = df_ori[df_ori["task_id"].isin(
                task_subset)].reset_index(drop=True)

            df["abs"] = df["abs"].astype(int)
            model_lymph = smf.glm(
                f'Q("abs") ~ {predictor}',
                data=df[df["task_id"] == "lymph_node"],
                family=sm.families.Binomial()
            ).fit()

            print(model_lymph.summary())

            # recurrence detection (boolean column: 'recurrence_abs')
            task_subset = [
                "recurrence",
            ]
            df = df_ori[df_ori["task_id"].isin(
                task_subset)].reset_index(drop=True)
            df["recurrence_abs"] = df["recurrence_abs"].astype(int)
            model_recur = smf.glm(
                "recurrence_abs ~ log_dsc",
                data=df[df["task_id"] == "recurrence"],
                family=sm.families.Binomial()
            ).fit()

            print(model_recur.summary())

            ly_pct, ly_lo, ly_hi, ly_p = _extract_effect(
                model_lymph, predictor)
            rc_pct, rc_lo, rc_hi, rc_p = _extract_effect(
                model_recur, "log_dsc")

            extra_rows: List[Tuple[str, float, float, float, float]] = []
            if output_var == 'log_bif_error':
                if predictor == 'log_dsc':
                    extra_rows = [
                        ("Lymph node", ly_pct, ly_lo, ly_hi, ly_p),
                        ("Recurrence", rc_pct, rc_lo, rc_hi, rc_p),
                    ]
                else:
                    extra_rows = [
                        ("Lymph node", ly_pct, ly_lo, ly_hi, ly_p),
                    ]

            ovar = "duration" if output_var == 'log_duration' else "accuracy"
            latex_table = build_latex_table_duration(
                per_task_duration, combined_duration, extra_rows, predictor[-3:], ovar)

            latex_table = latex_table.replace(
                "a_vertebralis_r", "A. Vertebralis R.")
            latex_table = latex_table.replace(
                "a_vertebralis_l", "A. Vertebralis L.")
            latex_table = latex_table.replace(
                "a_carotisexterna_r", "A. Carotis Externa R.")
            latex_table = latex_table.replace(
                "a_carotisexterna_l", "A. Carotis Externa L.")
            latex_table = latex_table.replace("lymph_node", "Lymph Node")
            latex_table = latex_table.replace("recurrence", "Recurrence")

            with open(f"outputs/mixed_models_table_{predictor}_{ovar}.tex", "w") as f:
                f.write(latex_table)

    x = 0


"""
task_subset = [
    "lymph_node",
    "recurrence",
]
df = df_ori[df_ori["task_id"].isin(task_subset)].reset_index(drop=True)

# combine recurrence_abs and abs into detection_success
df["detection_success"] = df[["abs", "recurrence_abs"]].bfill(axis=1).iloc[:, 0]

model_detection = smf.glm(
    "detection_success ~ log_dsc * C(task_id)",
    data=df[df["task_id"].isin(["lymph_node", "recurrence"])],
    family=sm.families.Binomial()
).fit()
print(model_detection.summary())

x = 0
"""

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    main()
