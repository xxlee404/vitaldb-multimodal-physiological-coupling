# -*- coding: utf-8 -*-
"""
VitalDB Multimodal Physiological Coupling Study
STEP 12B - Adjusted Physiological Characterization with Subject-Cluster Robust Inference

Context
-------
STEP 12A found multiple exploratory associations between the 12 locked
physiological coupling features and clinical/surgical context variables.

Purpose
-------
Test whether those associations remain after simultaneous adjustment for
other measured clinical/surgical context variables.

IMPORTANT
---------
- Postoperative outcomes are NOT loaded or inspected.
- All 12 locked coupling features are retained.
- No feature selection is performed.
- No interaction hunting is performed.
- This remains an exploratory/characterization analysis.

Repeated cases
--------------
Repeated cases from the same subject are handled using cluster-robust
sandwich standard errors clustered by subjectid.

Response transformation
-----------------------
Each coupling feature is rank-based inverse-normal transformed before modeling.
This provides a common standardized scale and reduces sensitivity to skewed,
bounded, or discrete coupling distributions.

Adjusted model for each coupling feature
----------------------------------------
coupling_z ~
    age_z + bmi_z
    + C(sex)
    + C(asa)
    + C(emop)
    + C(department)
    + C(ane_type)
    + C(optype)
    + C(approach)
    + C(position)
    + C(preop_htn)
    + C(preop_dm)

Inference
---------
- Cluster-robust Wald test for each context term
- Benjamini-Hochberg FDR across all feature-context omnibus tests
- Descriptive incremental adjusted R^2 for each term:
      full model adjusted R^2 - reduced model adjusted R^2

The delta adjusted R^2 is descriptive only; the cluster-robust Wald/FDR
result is the inferential result.
"""

from pathlib import Path
import warnings
import re

import numpy as np
import pandas as pd

from scipy.stats import norm, rankdata
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PATHS
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data_processed"
RESULT_DIR = PROJECT_DIR / "results"

INPUT_PATH = PROCESSED_DIR / "07D_preoutcome_modeling_dataset_frozen.csv"

OUT_OMNIBUS = RESULT_DIR / "12B_adjusted_context_omnibus_tests.csv"
OUT_COEFFICIENTS = RESULT_DIR / "12B_adjusted_model_coefficients.csv"
OUT_SUMMARY = RESULT_DIR / "12B_adjusted_characterization_summary.csv"
OUT_GROUP_DESCRIPTIVES = RESULT_DIR / "12B_context_group_descriptives.csv"
OUT_MODEL_DIAGNOSTICS = RESULT_DIR / "12B_model_diagnostics.csv"

# =============================================================================
# 2. LOCKED COUPLING FEATURES
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
    q = np.full(len(p), np.nan, dtype=float)

    valid = np.isfinite(p)
    idx = np.where(valid)[0]

    if len(idx) == 0:
        return q

    pv = p[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    unsorted = np.empty(m, dtype=float)
    unsorted[order] = adj
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

    # Blom-like plotting position
    probs = (ranks - 0.375) / (n + 0.25)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)

    out.loc[mask] = norm.ppf(probs)
    return out


def standardize(series):
    x = pd.to_numeric(series, errors="coerce")
    mean = x.mean()
    sd = x.std(ddof=1)

    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index, dtype=float)

    return (x - mean) / sd


def term_to_formula(term):
    if term in CONTINUOUS_CONTEXT:
        return f"{term}_z"
    return f"C({term})"


def get_term_columns(design_info, term_label):
    """
    Return design-matrix column indices belonging to a formula term.
    statsmodels/patsy term names should match e.g. 'age_z' or 'C(sex)'.
    """
    slices = design_info.term_name_slices

    if term_label not in slices:
        return []

    sl = slices[term_label]
    return list(range(sl.start, sl.stop))


def robust_wald_for_term(result, design_info, term_label):
    col_indices = get_term_columns(design_info, term_label)

    if len(col_indices) == 0:
        return np.nan, np.nan, 0

    p = len(result.params)
    R = np.zeros((len(col_indices), p), dtype=float)

    for row_i, col_i in enumerate(col_indices):
        R[row_i, col_i] = 1.0

    test = result.wald_test(R, scalar=True)

    statistic = float(np.asarray(test.statistic).squeeze())
    p_value = float(np.asarray(test.pvalue).squeeze())

    return statistic, p_value, len(col_indices)


def reduced_formula(full_terms, dropped_term):
    kept = [t for t in full_terms if t != dropped_term]

    if len(kept) == 0:
        return "coupling_z ~ 1"

    return "coupling_z ~ " + " + ".join(kept)


# =============================================================================
# 4. LOAD DATA
# =============================================================================
print("=" * 112)
print("VitalDB Multimodal Physiological Coupling Study")
print("STEP 12B - Adjusted Physiological Characterization")
print("=" * 112)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)
df.columns = (
    df.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

required = (
    ["caseid", "subjectid"]
    + COUPLING12
    + ALL_CONTEXT
)

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        "Required columns are missing:\n"
        + "\n".join(missing)
    )

print(f"\nCases: {len(df):,}")
print(f"Unique subjects: {df['subjectid'].nunique():,}")
print(f"Locked coupling features: {len(COUPLING12)}")
print(f"Adjusted context variables: {len(ALL_CONTEXT)}")
print("Postoperative outcomes loaded/inspected: NO")
print("Repeated-case handling: cluster-robust SE by subjectid")

# =============================================================================
# 5. PREPARE COMMON CLINICAL DATA
# =============================================================================
work = df.copy()

for c in CONTINUOUS_CONTEXT:
    work[f"{c}_z"] = standardize(work[c])

# Keep categorical variables as-is.
# Complete-case set is defined using ALL adjustment variables.
adjustment_columns = (
    [f"{c}_z" for c in CONTINUOUS_CONTEXT]
    + CATEGORICAL_CONTEXT
)

common_complete_mask = work[
    adjustment_columns + ["subjectid"]
].notna().all(axis=1)

common_n = int(common_complete_mask.sum())
common_subjects = int(
    work.loc[common_complete_mask, "subjectid"].nunique()
)

print(f"\nCommon complete-case rows for adjustment: {common_n:,}")
print(f"Common complete-case subjects: {common_subjects:,}")
print(
    f"Rows excluded for missing adjustment covariates: "
    f"{len(work) - common_n:,}"
)

# =============================================================================
# 6. FORMULA
# =============================================================================
FULL_TERMS = (
    [f"{c}_z" for c in CONTINUOUS_CONTEXT]
    + [f"C({c})" for c in CATEGORICAL_CONTEXT]
)

FULL_FORMULA = (
    "coupling_z ~ "
    + " + ".join(FULL_TERMS)
)

print("\nFull adjusted formula:")
print(FULL_FORMULA)

# =============================================================================
# 7. RUN ADJUSTED MODELS
# =============================================================================
omnibus_rows = []
coef_rows = []
diagnostic_rows = []

print("\n" + "=" * 112)
print("ADJUSTED MODELS")
print("=" * 112)

for feature_index, feature in enumerate(COUPLING12, start=1):

    temp = work.loc[common_complete_mask].copy()

    temp["coupling_z"] = inverse_normal_transform(
        temp[feature]
    )

    temp = temp.dropna(
        subset=["coupling_z", "subjectid"]
    ).copy()

    # Fit ordinary OLS for model fit metrics.
    ols_model = smf.ols(
        formula=FULL_FORMULA,
        data=temp,
    )

    ols_result = ols_model.fit()

    # Cluster-robust covariance by subjectid.
    robust_result = ols_model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": temp["subjectid"],
            "use_correction": True,
        },
    )

    design_info = ols_model.data.design_info

    full_adj_r2 = float(ols_result.rsquared_adj)
    full_r2 = float(ols_result.rsquared)

    diagnostic_rows.append({
        "coupling_feature": feature,
        "domain": DOMAIN_MAP[feature],
        "n_cases": len(temp),
        "n_subjects": temp["subjectid"].nunique(),
        "model_df": int(len(robust_result.params)),
        "R2": full_r2,
        "adjusted_R2": full_adj_r2,
    })

    print(
        f"{feature_index:02d}/{len(COUPLING12)} "
        f"{feature:<38} | "
        f"N={len(temp):,} | "
        f"subjects={temp['subjectid'].nunique():,} | "
        f"adj.R2={full_adj_r2:.4f}"
    )

    # ---------------------------------------------------------
    # Coefficients
    # ---------------------------------------------------------
    params = robust_result.params
    bse = robust_result.bse
    pvals = robust_result.pvalues

    conf = robust_result.conf_int()

    for name in params.index:
        coef_rows.append({
            "coupling_feature": feature,
            "domain": DOMAIN_MAP[feature],
            "parameter": name,
            "coefficient": float(params[name]),
            "robust_se": float(bse[name]),
            "ci_lower": float(conf.loc[name, 0]),
            "ci_upper": float(conf.loc[name, 1]),
            "p_value": float(pvals[name]),
        })

    # ---------------------------------------------------------
    # Omnibus term tests + descriptive delta adjusted R2
    # ---------------------------------------------------------
    for context in ALL_CONTEXT:

        formula_term = term_to_formula(context)

        stat, p_value, df_term = robust_wald_for_term(
            robust_result,
            design_info,
            formula_term,
        )

        # Reduced model fit on identical complete-case rows.
        reduced = reduced_formula(
            FULL_TERMS,
            formula_term,
        )

        reduced_result = smf.ols(
            formula=reduced,
            data=temp,
        ).fit()

        delta_adj_r2 = (
            full_adj_r2
            - float(reduced_result.rsquared_adj)
        )

        delta_r2 = (
            full_r2
            - float(reduced_result.rsquared)
        )

        # For single-df continuous terms, capture standardized beta.
        single_beta = np.nan

        if context in CONTINUOUS_CONTEXT:
            param_name = f"{context}_z"

            if param_name in params.index:
                single_beta = float(params[param_name])

        omnibus_rows.append({
            "coupling_feature": feature,
            "domain": DOMAIN_MAP[feature],
            "context_variable": context,
            "context_type": (
                "continuous"
                if context in CONTINUOUS_CONTEXT
                else "categorical"
            ),
            "n_cases": len(temp),
            "n_subjects": temp["subjectid"].nunique(),
            "wald_chi2": stat,
            "wald_df": df_term,
            "p_value_cluster_robust": p_value,
            "delta_R2": float(delta_r2),
            "delta_adjusted_R2": float(delta_adj_r2),
            "standardized_beta_if_single_df": single_beta,
            "full_model_adjusted_R2": full_adj_r2,
        })

# =============================================================================
# 8. MULTIPLE TESTING CORRECTION
# =============================================================================
omnibus = pd.DataFrame(omnibus_rows)

omnibus["q_value_FDR"] = benjamini_hochberg(
    omnibus["p_value_cluster_robust"].to_numpy(dtype=float)
)

omnibus["FDR_lt_0_05"] = (
    omnibus["q_value_FDR"] < 0.05
)

omnibus["FDR_lt_0_01"] = (
    omnibus["q_value_FDR"] < 0.01
)

# Effect-size magnitude based on descriptive incremental adjusted R2.
def delta_r2_magnitude(value):
    if pd.isna(value):
        return "unknown"

    value = max(0.0, float(value))

    if value >= 0.06:
        return "large"
    if value >= 0.02:
        return "moderate"
    if value >= 0.005:
        return "small"
    return "very_small"


omnibus["incremental_effect_magnitude"] = (
    omnibus["delta_adjusted_R2"]
    .apply(delta_r2_magnitude)
)

omnibus = omnibus.sort_values(
    [
        "q_value_FDR",
        "delta_adjusted_R2",
        "coupling_feature",
    ],
    ascending=[
        True,
        False,
        True,
    ]
).reset_index(drop=True)

omnibus.to_csv(
    OUT_OMNIBUS,
    index=False,
    encoding="utf-8-sig",
)

coef_df = pd.DataFrame(coef_rows)

coef_df.to_csv(
    OUT_COEFFICIENTS,
    index=False,
    encoding="utf-8-sig",
)

diagnostics = pd.DataFrame(diagnostic_rows)

diagnostics.to_csv(
    OUT_MODEL_DIAGNOSTICS,
    index=False,
    encoding="utf-8-sig",
)

# =============================================================================
# 9. DESCRIPTIVE GROUP SUMMARIES
# =============================================================================
desc_rows = []

for feature in COUPLING12:
    for context in CATEGORICAL_CONTEXT:
        temp = df[[feature, context]].dropna().copy()

        for level, sub in temp.groupby(
            context,
            observed=False,
        ):
            values = pd.to_numeric(
                sub[feature],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                continue

            desc_rows.append({
                "coupling_feature": feature,
                "domain": DOMAIN_MAP[feature],
                "context_variable": context,
                "level": str(level),
                "n": len(values),
                "median": float(values.median()),
                "q1": float(values.quantile(0.25)),
                "q3": float(values.quantile(0.75)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            })

pd.DataFrame(desc_rows).to_csv(
    OUT_GROUP_DESCRIPTIVES,
    index=False,
    encoding="utf-8-sig",
)

# =============================================================================
# 10. SUMMARIES
# =============================================================================
summary_rows = []

for feature in COUPLING12:
    sub = omnibus.loc[
        omnibus["coupling_feature"] == feature
    ].copy()

    sig = sub.loc[
        sub["FDR_lt_0_05"]
    ].copy()

    if len(sub) == 0:
        continue

    strongest = sub.sort_values(
        [
            "delta_adjusted_R2",
            "q_value_FDR",
        ],
        ascending=[
            False,
            True,
        ]
    ).iloc[0]

    summary_rows.append({
        "coupling_feature": feature,
        "domain": DOMAIN_MAP[feature],
        "adjusted_tests_n": len(sub),
        "FDR_significant_n": int(
            sub["FDR_lt_0_05"].sum()
        ),
        "FDR_0_01_n": int(
            sub["FDR_lt_0_01"].sum()
        ),
        "strongest_context_by_delta_adjR2":
            strongest["context_variable"],
        "strongest_delta_adjusted_R2":
            float(strongest["delta_adjusted_R2"]),
        "strongest_effect_magnitude":
            strongest["incremental_effect_magnitude"],
        "strongest_q_value":
            float(strongest["q_value_FDR"]),
        "full_model_adjusted_R2":
            float(strongest["full_model_adjusted_R2"]),
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig",
)

# =============================================================================
# 11. CONSOLE OUTPUT
# =============================================================================
print("\n" + "=" * 112)
print("STEP 12B OVERALL ADJUSTED SUMMARY")
print("=" * 112)

print(f"\nTotal adjusted omnibus tests: {len(omnibus):,}")
print(
    f"FDR q<0.05: "
    f"{int(omnibus['FDR_lt_0_05'].sum()):,}"
)
print(
    f"FDR q<0.01: "
    f"{int(omnibus['FDR_lt_0_01'].sum()):,}"
)

print("\nPer-feature adjusted summary:")
print(
    summary.to_string(
        index=False,
        formatters={
            "strongest_delta_adjusted_R2":
                lambda x: f"{x:+.5f}",
            "strongest_q_value":
                lambda x: f"{x:.4g}",
            "full_model_adjusted_R2":
                lambda x: f"{x:.4f}",
        }
    )
)

print("\n" + "=" * 112)
print("TOP 30 ADJUSTED FDR-SIGNIFICANT ASSOCIATIONS")
print("=" * 112)

top = omnibus.loc[
    omnibus["FDR_lt_0_05"]
].sort_values(
    [
        "delta_adjusted_R2",
        "q_value_FDR",
    ],
    ascending=[
        False,
        True,
    ]
).head(30)

if len(top) == 0:
    print("No adjusted FDR-significant associations.")
else:
    print(
        top[
            [
                "coupling_feature",
                "domain",
                "context_variable",
                "wald_chi2",
                "wald_df",
                "p_value_cluster_robust",
                "q_value_FDR",
                "delta_adjusted_R2",
                "incremental_effect_magnitude",
                "standardized_beta_if_single_df",
            ]
        ].to_string(
            index=False,
            formatters={
                "wald_chi2":
                    lambda x: f"{x:.3f}",
                "p_value_cluster_robust":
                    lambda x: f"{x:.4g}",
                "q_value_FDR":
                    lambda x: f"{x:.4g}",
                "delta_adjusted_R2":
                    lambda x: f"{x:+.5f}",
                "standardized_beta_if_single_df":
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:+.4f}"
                    ),
            }
        )
    )

print("\n" + "=" * 112)
print("DOMAIN-LEVEL COUNTS")
print("=" * 112)

domain_summary = (
    omnibus
    .groupby("domain")
    .agg(
        tests=("context_variable", "size"),
        FDR_q05=("FDR_lt_0_05", "sum"),
        FDR_q01=("FDR_lt_0_01", "sum"),
        max_delta_adjR2=("delta_adjusted_R2", "max"),
    )
    .reset_index()
)

print(
    domain_summary.to_string(
        index=False,
        formatters={
            "max_delta_adjR2":
                lambda x: f"{x:+.5f}",
        }
    )
)

print("\n" + "=" * 112)
print("STEP 12B INTERPRETATION RULE")
print("=" * 112)

print("""
A coherent characterization signal is supported if:
1) clinically interpretable coupling-context associations remain FDR-significant
   after simultaneous adjustment, AND
2) the associations cluster into plausible physiological domains rather than
   appearing as isolated scattered findings, AND
3) effect sizes remain non-trivial after adjustment.

Because this analysis is exploratory, even robust associations should be framed
as physiological characterization, not causal effects.

Postoperative outcome was not used.
The failed incremental ICU-prediction hypothesis remains closed.
""")

print("Generated files:")
for path in [
    OUT_OMNIBUS,
    OUT_COEFFICIENTS,
    OUT_SUMMARY,
    OUT_GROUP_DESCRIPTIVES,
    OUT_MODEL_DIAGNOSTICS,
]:
    print(path)

print(
    """
NEXT:
Send:
1) STEP 12B OVERALL ADJUSTED SUMMARY
2) Per-feature adjusted summary
3) TOP 30 ADJUSTED FDR-SIGNIFICANT ASSOCIATIONS
4) DOMAIN-LEVEL COUNTS

Then we can make the final publication decision for the physiological
characterization direction.
"""
)