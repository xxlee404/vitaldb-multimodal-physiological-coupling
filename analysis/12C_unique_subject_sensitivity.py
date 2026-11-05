# -*- coding: utf-8 -*-
"""
VitalDB Multimodal Physiological Coupling Study
STEP 12C - Unique-Subject Robustness Check for Physiological Characterization

Purpose
-------
Final robustness check for the exploratory physiological-characterization analysis.

STEP 12B used all cases and handled repeated cases with subject-cluster robust SE.
STEP 12C removes repeated-case dependence entirely by retaining ONE deterministic
case per subject and repeating the same fully adjusted 12 x 12 context analysis.

This step DOES NOT:
- use postoperative outcomes
- select or remove coupling features
- change the adjustment set
- search for new interactions
- tune any model

The aim is only to determine whether the STEP 12B characterization pattern
remains qualitatively similar in a unique-subject sensitivity cohort.

One-case-per-subject rule
-------------------------
If analysis_case_order exists:
    retain the lowest analysis_case_order within each subject.
Otherwise:
    retain the lowest caseid within each subject.

Inference
---------
- Rank-based inverse-normal transform of each coupling feature
- Same simultaneous adjustment set as STEP 12B
- HC3 robust standard errors
- Omnibus Wald tests for each context term
- Benjamini-Hochberg FDR across all 144 tests
- Descriptive delta adjusted R^2

Primary robustness summaries
----------------------------
1) How many STEP 12B FDR-significant tests remain FDR-significant in STEP 12C?
2) How many STEP 12B moderate-effect findings remain significant?
3) Correlation of delta adjusted R^2 between STEP 12B and STEP 12C
4) Domain-level persistence
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm, rankdata, spearmanr
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PATHS
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data_processed"
RESULT_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures" / "12C_unique_subject_robustness"

INPUT_DATA = PROCESSED_DIR / "07D_preoutcome_modeling_dataset_frozen.csv"
INPUT_12B = RESULT_DIR / "12B_adjusted_context_omnibus_tests.csv"

OUT_OMNIBUS = RESULT_DIR / "12C_unique_subject_adjusted_omnibus_tests.csv"
OUT_COMPARE = RESULT_DIR / "12C_comparison_with_STEP12B.csv"
OUT_SUMMARY = RESULT_DIR / "12C_unique_subject_robustness_summary.csv"
OUT_DOMAIN = RESULT_DIR / "12C_domain_robustness_summary.csv"
OUT_LEVEL_COUNTS = RESULT_DIR / "12C_context_level_counts.csv"

FIG_DELTA = FIG_DIR / "12C_delta_adjR2_12B_vs_12C.png"
FIG_MATRIX = FIG_DIR / "12C_unique_subject_delta_adjR2_heatmap.png"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. LOCKED DEFINITIONS
# =============================================================================
COUPLING12 = [
    "art_pleth_lag_median_sec",
    "art_pleth_lag_iqr_sec",
    "art_pleth_xcorr_median",
    "art_pleth_xcorr_iqr",
    "art_pleth_xcorr_lag_median_sec",
    "art_pleth_amplitude_spearman",
    "hr_map_corr_median",
    "hr_map_corr_iqr",
    "map_etco2_corr_median",
    "map_etco2_corr_iqr",
    "hr_etco2_corr_median",
    "hr_etco2_corr_iqr",
]

DOMAIN_MAP = {
    "art_pleth_lag_median_sec": "ART_PLETH",
    "art_pleth_lag_iqr_sec": "ART_PLETH",
    "art_pleth_xcorr_median": "ART_PLETH",
    "art_pleth_xcorr_iqr": "ART_PLETH",
    "art_pleth_xcorr_lag_median_sec": "ART_PLETH",
    "art_pleth_amplitude_spearman": "ART_PLETH",
    "hr_map_corr_median": "HR_MAP",
    "hr_map_corr_iqr": "HR_MAP",
    "map_etco2_corr_median": "MAP_ETCO2",
    "map_etco2_corr_iqr": "MAP_ETCO2",
    "hr_etco2_corr_median": "HR_ETCO2",
    "hr_etco2_corr_iqr": "HR_ETCO2",
}

CONTINUOUS_CONTEXT = ["age", "bmi"]

CATEGORICAL_CONTEXT = [
    "sex",
    "asa",
    "emop",
    "department",
    "ane_type",
    "optype",
    "approach",
    "position",
    "preop_htn",
    "preop_dm",
]

ALL_CONTEXT = CONTINUOUS_CONTEXT + CATEGORICAL_CONTEXT

# =============================================================================
# 3. HELPERS
# =============================================================================
def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan)

    valid = np.isfinite(p)
    idx = np.where(valid)[0]

    if len(idx) == 0:
        return q

    pv = p[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    unsorted = np.empty(m, dtype=float)
    unsorted[order] = adjusted
    q[idx] = unsorted

    return q


def inverse_normal_transform(series):
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=float)

    mask = x.notna()
    n = int(mask.sum())

    if n == 0:
        return out

    ranks = rankdata(
        x.loc[mask].to_numpy(dtype=float),
        method="average",
    )

    probs = (ranks - 0.375) / (n + 0.25)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)

    out.loc[mask] = norm.ppf(probs)
    return out


def standardize(series):
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=1)

    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index, dtype=float)

    return (x - x.mean()) / sd


def term_to_formula(term):
    if term in CONTINUOUS_CONTEXT:
        return f"{term}_z"

    return f"C({term})"


def get_term_columns(design_info, term_label):
    if term_label not in design_info.term_name_slices:
        return []

    sl = design_info.term_name_slices[term_label]
    return list(range(sl.start, sl.stop))


def robust_wald_for_term(result, design_info, term_label):
    cols = get_term_columns(design_info, term_label)

    if len(cols) == 0:
        return np.nan, np.nan, 0

    p = len(result.params)
    R = np.zeros((len(cols), p), dtype=float)

    for r, c in enumerate(cols):
        R[r, c] = 1.0

    test = result.wald_test(R, scalar=True)

    stat = float(np.asarray(test.statistic).squeeze())
    p_value = float(np.asarray(test.pvalue).squeeze())

    return stat, p_value, len(cols)


def reduced_formula(full_terms, dropped_term):
    kept = [x for x in full_terms if x != dropped_term]

    if not kept:
        return "coupling_z ~ 1"

    return "coupling_z ~ " + " + ".join(kept)


def effect_magnitude(delta_adj_r2):
    x = max(0.0, float(delta_adj_r2))

    if x >= 0.06:
        return "large"
    if x >= 0.02:
        return "moderate"
    if x >= 0.005:
        return "small"
    return "very_small"


# =============================================================================
# 4. LOAD DATA
# =============================================================================
print("=" * 112)
print("VitalDB Multimodal Physiological Coupling Study")
print("STEP 12C - Unique-Subject Robustness Check")
print("=" * 112)

for path in [INPUT_DATA, INPUT_12B]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

df = pd.read_csv(INPUT_DATA)
df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

step12b = pd.read_csv(INPUT_12B)
step12b.columns = (
    step12b.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

required = ["caseid", "subjectid"] + COUPLING12 + ALL_CONTEXT

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        "Required columns missing:\n"
        + "\n".join(missing)
    )

print(f"\nOriginal cases: {len(df):,}")
print(f"Original unique subjects: {df['subjectid'].nunique():,}")
print("Postoperative outcomes loaded/inspected: NO")

# =============================================================================
# 5. DETERMINISTIC ONE CASE PER SUBJECT
# =============================================================================
sort_columns = ["subjectid"]

if "analysis_case_order" in df.columns:
    sort_columns.append("analysis_case_order")
    selection_rule = (
        "lowest analysis_case_order within subject"
    )
else:
    sort_columns.append("caseid")
    selection_rule = (
        "lowest caseid within subject"
    )

unique_df = (
    df.sort_values(sort_columns)
    .drop_duplicates(
        subset=["subjectid"],
        keep="first",
    )
    .copy()
)

print(f"\nUnique-subject cases retained: {len(unique_df):,}")
print(f"Unique subjects retained: {unique_df['subjectid'].nunique():,}")
print(f"Selection rule: {selection_rule}")
print(
    f"Repeated case rows removed: "
    f"{len(df) - len(unique_df):,}"
)

# =============================================================================
# 6. CONTEXT LEVEL COUNTS
# =============================================================================
level_rows = []

for context in CATEGORICAL_CONTEXT:
    counts = (
        unique_df[context]
        .value_counts(dropna=False)
    )

    for level, n in counts.items():
        level_rows.append({
            "context_variable": context,
            "level": str(level),
            "n": int(n),
        })

level_df = pd.DataFrame(level_rows)
level_df.to_csv(
    OUT_LEVEL_COUNTS,
    index=False,
    encoding="utf-8-sig",
)

# =============================================================================
# 7. PREPARE ADJUSTMENT DATA
# =============================================================================
work = unique_df.copy()

for c in CONTINUOUS_CONTEXT:
    work[f"{c}_z"] = standardize(work[c])

adjustment_columns = (
    [f"{c}_z" for c in CONTINUOUS_CONTEXT]
    + CATEGORICAL_CONTEXT
)

complete_mask = work[
    adjustment_columns
].notna().all(axis=1)

complete_n = int(complete_mask.sum())

print(f"\nComplete-case unique subjects: {complete_n:,}")
print(
    f"Excluded for missing adjustment covariates: "
    f"{len(work) - complete_n:,}"
)

FULL_TERMS = (
    [f"{c}_z" for c in CONTINUOUS_CONTEXT]
    + [f"C({c})" for c in CATEGORICAL_CONTEXT]
)

FULL_FORMULA = (
    "coupling_z ~ "
    + " + ".join(FULL_TERMS)
)

# =============================================================================
# 8. REPEAT ADJUSTED MODELS
# =============================================================================
rows = []

print("\n" + "=" * 112)
print("UNIQUE-SUBJECT ADJUSTED MODELS")
print("=" * 112)

for i, feature in enumerate(COUPLING12, start=1):
    temp = work.loc[complete_mask].copy()

    temp["coupling_z"] = inverse_normal_transform(
        temp[feature]
    )

    temp = temp.dropna(
        subset=["coupling_z"]
    ).copy()

    model = smf.ols(
        formula=FULL_FORMULA,
        data=temp,
    )

    ordinary = model.fit()

    robust = model.fit(
        cov_type="HC3"
    )

    full_r2 = float(ordinary.rsquared)
    full_adj_r2 = float(ordinary.rsquared_adj)

    design_info = model.data.design_info

    print(
        f"{i:02d}/{len(COUPLING12)} "
        f"{feature:<38} | "
        f"N={len(temp):,} | "
        f"adj.R2={full_adj_r2:.4f}"
    )

    for context in ALL_CONTEXT:
        term = term_to_formula(context)

        stat, p_value, df_term = robust_wald_for_term(
            robust,
            design_info,
            term,
        )

        reduced = reduced_formula(
            FULL_TERMS,
            term,
        )

        reduced_result = smf.ols(
            formula=reduced,
            data=temp,
        ).fit()

        delta_r2 = (
            full_r2
            - float(reduced_result.rsquared)
        )

        delta_adj_r2 = (
            full_adj_r2
            - float(reduced_result.rsquared_adj)
        )

        beta = np.nan

        if context in CONTINUOUS_CONTEXT:
            param_name = f"{context}_z"

            if param_name in robust.params.index:
                beta = float(
                    robust.params[param_name]
                )

        rows.append({
            "coupling_feature": feature,
            "domain": DOMAIN_MAP[feature],
            "context_variable": context,
            "n_unique_subjects": len(temp),
            "wald_chi2": stat,
            "wald_df": df_term,
            "p_value_HC3": p_value,
            "delta_R2": float(delta_r2),
            "delta_adjusted_R2": float(delta_adj_r2),
            "effect_magnitude": effect_magnitude(
                delta_adj_r2
            ),
            "standardized_beta_if_single_df": beta,
            "full_model_adjusted_R2": full_adj_r2,
        })

omnibus = pd.DataFrame(rows)

omnibus["q_value_FDR"] = benjamini_hochberg(
    omnibus["p_value_HC3"].to_numpy(dtype=float)
)

omnibus["FDR_lt_0_05"] = (
    omnibus["q_value_FDR"] < 0.05
)

omnibus["FDR_lt_0_01"] = (
    omnibus["q_value_FDR"] < 0.01
)

omnibus = omnibus.sort_values(
    [
        "q_value_FDR",
        "delta_adjusted_R2",
    ],
    ascending=[
        True,
        False,
    ]
).reset_index(drop=True)

omnibus.to_csv(
    OUT_OMNIBUS,
    index=False,
    encoding="utf-8-sig",
)

# =============================================================================
# 9. COMPARE STEP 12B vs STEP 12C
# =============================================================================
keys = [
    "coupling_feature",
    "domain",
    "context_variable",
]

needed_12b = [
    "coupling_feature",
    "domain",
    "context_variable",
    "q_value_FDR",
    "FDR_lt_0_05",
    "FDR_lt_0_01",
    "delta_adjusted_R2",
    "incremental_effect_magnitude",
    "standardized_beta_if_single_df",
]

missing_12b = [
    c for c in needed_12b
    if c not in step12b.columns
]

if missing_12b:
    raise ValueError(
        "STEP 12B comparison columns missing:\n"
        + "\n".join(missing_12b)
    )

compare = step12b[needed_12b].merge(
    omnibus[
        [
            "coupling_feature",
            "domain",
            "context_variable",
            "q_value_FDR",
            "FDR_lt_0_05",
            "FDR_lt_0_01",
            "delta_adjusted_R2",
            "effect_magnitude",
            "standardized_beta_if_single_df",
        ]
    ],
    on=keys,
    how="inner",
    suffixes=("_12B", "_12C"),
    validate="one_to_one",
)

compare["significant_both_q05"] = (
    compare["FDR_lt_0_05_12B"]
    & compare["FDR_lt_0_05_12C"]
)

compare["significant_both_q01"] = (
    compare["FDR_lt_0_01_12B"]
    & compare["FDR_lt_0_01_12C"]
)

compare["12B_moderate_or_large"] = (
    compare["incremental_effect_magnitude"]
    .isin(["moderate", "large"])
)

compare["12B_moderate_persists_q05"] = (
    compare["12B_moderate_or_large"]
    & compare["FDR_lt_0_05_12C"]
)

# For continuous predictors only, compare beta directions.
def same_beta_direction(row):
    b = row["standardized_beta_if_single_df_12B"]
    c = row["standardized_beta_if_single_df_12C"]

    if pd.isna(b) or pd.isna(c):
        return np.nan

    return bool(np.sign(b) == np.sign(c))

compare["same_continuous_beta_direction"] = compare.apply(
    same_beta_direction,
    axis=1,
)

compare.to_csv(
    OUT_COMPARE,
    index=False,
    encoding="utf-8-sig",
)

# =============================================================================
# 10. ROBUSTNESS SUMMARY
# =============================================================================
n_tests = len(compare)

b_sig = int(
    compare["FDR_lt_0_05_12B"].sum()
)

c_sig = int(
    compare["FDR_lt_0_05_12C"].sum()
)

both_sig = int(
    compare["significant_both_q05"].sum()
)

b_mod = int(
    compare["12B_moderate_or_large"].sum()
)

b_mod_persist = int(
    compare["12B_moderate_persists_q05"].sum()
)

rho_delta, p_delta = spearmanr(
    compare["delta_adjusted_R2_12B"],
    compare["delta_adjusted_R2_12C"],
    nan_policy="omit",
)

continuous_compare = compare.loc[
    compare["context_variable"].isin(
        CONTINUOUS_CONTEXT
    )
].copy()

direction_valid = continuous_compare[
    "same_continuous_beta_direction"
].dropna()

direction_match = int(
    direction_valid.sum()
)

direction_total = len(direction_valid)

summary = pd.DataFrame([{
    "total_compared_tests": n_tests,
    "step12B_FDR_q05_n": b_sig,
    "step12C_FDR_q05_n": c_sig,
    "FDR_q05_significant_in_both_n": both_sig,
    "step12B_significant_persistence_pct":
        (both_sig / b_sig * 100.0)
        if b_sig else np.nan,
    "step12B_moderate_or_large_n": b_mod,
    "step12B_moderate_or_large_persist_q05_n":
        b_mod_persist,
    "moderate_effect_persistence_pct":
        (b_mod_persist / b_mod * 100.0)
        if b_mod else np.nan,
    "delta_adjR2_spearman_rho_12B_vs_12C":
        float(rho_delta),
    "delta_adjR2_spearman_p":
        float(p_delta),
    "continuous_beta_direction_match_n":
        direction_match,
    "continuous_beta_direction_total_n":
        direction_total,
    "continuous_beta_direction_match_pct":
        (direction_match / direction_total * 100.0)
        if direction_total else np.nan,
}])

summary.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig",
)

# =============================================================================
# 11. DOMAIN ROBUSTNESS
# =============================================================================
domain_rows = []

for domain, sub in compare.groupby("domain"):
    b_domain_sig = int(
        sub["FDR_lt_0_05_12B"].sum()
    )

    both_domain = int(
        sub["significant_both_q05"].sum()
    )

    domain_rows.append({
        "domain": domain,
        "tests": len(sub),
        "step12B_q05_n": b_domain_sig,
        "step12C_q05_n": int(
            sub["FDR_lt_0_05_12C"].sum()
        ),
        "significant_both_q05_n":
            both_domain,
        "persistence_pct":
            (
                both_domain
                / b_domain_sig
                * 100.0
            )
            if b_domain_sig
            else np.nan,
        "max_delta_adjR2_12B":
            float(
                sub[
                    "delta_adjusted_R2_12B"
                ].max()
            ),
        "max_delta_adjR2_12C":
            float(
                sub[
                    "delta_adjusted_R2_12C"
                ].max()
            ),
    })

domain_df = pd.DataFrame(domain_rows)

domain_df.to_csv(
    OUT_DOMAIN,
    index=False,
    encoding="utf-8-sig",
)

# =============================================================================
# 12. FIGURES
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(
    compare["delta_adjusted_R2_12B"],
    compare["delta_adjusted_R2_12C"],
)

minimum = float(
    min(
        compare["delta_adjusted_R2_12B"].min(),
        compare["delta_adjusted_R2_12C"].min(),
    )
)

maximum = float(
    max(
        compare["delta_adjusted_R2_12B"].max(),
        compare["delta_adjusted_R2_12C"].max(),
    )
)

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
)

ax.set_xlabel(
    "STEP 12B delta adjusted R²"
)

ax.set_ylabel(
    "STEP 12C unique-subject delta adjusted R²"
)

ax.set_title(
    "Robustness of Context Effect Sizes"
)

fig.tight_layout()

fig.savefig(
    FIG_DELTA,
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)

matrix = omnibus.pivot(
    index="coupling_feature",
    columns="context_variable",
    values="delta_adjusted_R2",
)

ordered_context = [
    c for c in ALL_CONTEXT
    if c in matrix.columns
]

matrix = matrix[ordered_context]

fig, ax = plt.subplots(
    figsize=(11, 8)
)

im = ax.imshow(
    matrix.to_numpy(dtype=float),
    aspect="auto",
)

ax.set_xticks(
    np.arange(matrix.shape[1])
)

ax.set_yticks(
    np.arange(matrix.shape[0])
)

ax.set_xticklabels(
    matrix.columns,
    rotation=45,
    ha="right",
    fontsize=8,
)

ax.set_yticklabels(
    matrix.index,
    fontsize=8,
)

ax.set_title(
    "Unique-Subject Sensitivity: Delta Adjusted R²"
)

fig.colorbar(
    im,
    ax=ax,
    fraction=0.03,
    pad=0.02,
)

fig.tight_layout()

fig.savefig(
    FIG_MATRIX,
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)

# =============================================================================
# 13. PRINT RESULTS
# =============================================================================
print("\n" + "=" * 112)
print("STEP 12C ROBUSTNESS SUMMARY")
print("=" * 112)

print(
    f"\nCompared tests: {n_tests}"
)

print(
    f"STEP 12B FDR q<0.05: {b_sig}"
)

print(
    f"STEP 12C FDR q<0.05: {c_sig}"
)

print(
    f"Significant in BOTH: {both_sig}"
)

print(
    f"Persistence of STEP 12B significant findings: "
    f"{(both_sig / b_sig * 100.0) if b_sig else np.nan:.1f}%"
)

print(
    f"\nSTEP 12B moderate/large effects: "
    f"{b_mod}"
)

print(
    f"Moderate/large effects remaining q<0.05 in STEP 12C: "
    f"{b_mod_persist}"
)

print(
    f"Moderate-effect persistence: "
    f"{(b_mod_persist / b_mod * 100.0) if b_mod else np.nan:.1f}%"
)

print(
    f"\nSpearman correlation of delta adjusted R² "
    f"(12B vs 12C): rho={rho_delta:.4f}, p={p_delta:.4g}"
)

print(
    f"Continuous beta direction agreement: "
    f"{direction_match}/{direction_total} "
    f"({(direction_match / direction_total * 100.0) if direction_total else np.nan:.1f}%)"
)

print("\n" + "=" * 112)
print("DOMAIN ROBUSTNESS")
print("=" * 112)

print(
    domain_df.to_string(
        index=False,
        formatters={
            "persistence_pct":
                lambda x: f"{x:.1f}",
            "max_delta_adjR2_12B":
                lambda x: f"{x:+.5f}",
            "max_delta_adjR2_12C":
                lambda x: f"{x:+.5f}",
        }
    )
)

print("\n" + "=" * 112)
print("TOP 20 ROBUST ASSOCIATIONS")
print("=" * 112)

robust_top = (
    compare.loc[
        compare["significant_both_q05"]
    ]
    .sort_values(
        [
            "delta_adjusted_R2_12B",
            "q_value_FDR_12B",
        ],
        ascending=[
            False,
            True,
        ]
    )
    .head(20)
)

if len(robust_top) == 0:
    print("No associations significant in both STEP 12B and STEP 12C.")
else:
    print(
        robust_top[
            [
                "coupling_feature",
                "domain",
                "context_variable",
                "delta_adjusted_R2_12B",
                "q_value_FDR_12B",
                "delta_adjusted_R2_12C",
                "q_value_FDR_12C",
                "incremental_effect_magnitude",
            ]
        ].to_string(
            index=False,
            formatters={
                "delta_adjusted_R2_12B":
                    lambda x: f"{x:+.5f}",
                "q_value_FDR_12B":
                    lambda x: f"{x:.4g}",
                "delta_adjusted_R2_12C":
                    lambda x: f"{x:+.5f}",
                "q_value_FDR_12C":
                    lambda x: f"{x:.4g}",
            }
        )
    )

print("\n" + "=" * 112)
print("STEP 12C PUBLICATION ROBUSTNESS DECISION")
print("=" * 112)

persistence_pct = (
    both_sig / b_sig * 100.0
    if b_sig
    else 0.0
)

moderate_persistence_pct = (
    b_mod_persist / b_mod * 100.0
    if b_mod
    else 0.0
)

if (
    persistence_pct >= 70.0
    and moderate_persistence_pct >= 75.0
    and rho_delta >= 0.75
):
    decision = "ROBUST CHARACTERIZATION SIGNAL"
    reason = (
        "Most adjusted STEP 12B findings persist after restricting to one case "
        "per subject, including most moderate-effect findings, with strong "
        "agreement in effect-size ranking."
    )

elif (
    persistence_pct >= 50.0
    and rho_delta >= 0.60
):
    decision = "MODERATELY ROBUST CHARACTERIZATION SIGNAL"
    reason = (
        "A substantial proportion of adjusted findings persist in the "
        "unique-subject sensitivity analysis."
    )

else:
    decision = "LIMITED ROBUSTNESS"
    reason = (
        "Too many STEP 12B findings attenuate when analysis is restricted "
        "to one case per subject."
    )

print(f"\nDecision: {decision}")
print(f"Reason: {reason}")

print("\nGenerated files:")
for path in [
    OUT_OMNIBUS,
    OUT_COMPARE,
    OUT_SUMMARY,
    OUT_DOMAIN,
    OUT_LEVEL_COUNTS,
    FIG_DELTA,
    FIG_MATRIX,
]:
    print(path)

print(
    """
NEXT:
Send:
1) STEP 12C ROBUSTNESS SUMMARY
2) DOMAIN ROBUSTNESS
3) TOP 20 ROBUST ASSOCIATIONS
4) STEP 12C PUBLICATION ROBUSTNESS DECISION

If robustness is adequate, stop statistical exploration and move directly
to manuscript tables/figures and Results/Discussion drafting.
"""
)