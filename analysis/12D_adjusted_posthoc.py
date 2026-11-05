# -*- coding: utf-8 -*-
"""
VitalDB Multimodal Physiological Coupling Study
STEP 12D - Adjusted Post-hoc Interpretation of Robust Moderate Associations

Purpose
-------
Final interpretability analysis ONLY.

STEP 12B/12C already established robust physiological-characterization signals.
This script does not search for new findings. It automatically selects only
associations that were:

1) FDR q<0.05 in BOTH STEP 12B and STEP 12C, AND
2) moderate/large effect in STEP 12B.

For categorical contexts (e.g., optype, approach):
- Fit the SAME fully adjusted model as STEP 12B
- Use subject-cluster robust covariance
- Perform all pairwise adjusted contrasts between levels
- Apply Benjamini-Hochberg FDR WITHIN each prespecified feature-context family

For continuous contexts (e.g., age):
- Report the adjusted standardized beta from the same model
- Also report an interpretable adjusted contrast for +10 years (age only)
  on the rank-normalized coupling scale.

IMPORTANT
---------
- No postoperative outcomes are loaded.
- No new coupling feature is added.
- No feature selection beyond the prespecified robust-moderate filter.
- No interaction terms are added.
- No attempt is made to improve prediction.
"""

from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd

from scipy.stats import norm, rankdata
from patsy import build_design_matrices
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PATHS
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data_processed"
RESULT_DIR = PROJECT_DIR / "results"

INPUT_DATA = PROCESSED_DIR / "07D_preoutcome_modeling_dataset_frozen.csv"
INPUT_COMPARE = RESULT_DIR / "12C_comparison_with_STEP12B.csv"

OUT_SELECTED = RESULT_DIR / "12D_prespecified_robust_moderate_targets.csv"
OUT_PAIRWISE = RESULT_DIR / "12D_adjusted_pairwise_comparisons.csv"
OUT_CONTINUOUS = RESULT_DIR / "12D_adjusted_continuous_effects.csv"
OUT_SUMMARY = RESULT_DIR / "12D_posthoc_interpretation_summary.csv"

# =============================================================================
# 2. LOCKED DEFINITIONS
# =============================================================================
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

# Same adjustment model as STEP 12B.
FULL_TERMS = (
    [f"{c}_z" for c in CONTINUOUS_CONTEXT]
    + [f"C({c})" for c in CATEGORICAL_CONTEXT]
)

FULL_FORMULA = (
    "coupling_z ~ "
    + " + ".join(FULL_TERMS)
)

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


def standardize_with_reference(series):
    x = pd.to_numeric(series, errors="coerce")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))

    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Cannot standardize variable with zero/non-finite SD.")

    z = (x - mean) / sd
    return z, mean, sd


def sorted_levels(series):
    values = pd.Series(series).dropna().unique().tolist()
    return sorted(values, key=lambda x: str(x))


def make_reference_row(temp):
    """
    Construct one representative row for design-matrix contrasts.
    With no interactions, categorical pairwise differences are invariant
    to values of the other covariates, but we use sensible reference values.
    """
    row = {}

    # standardized continuous covariates fixed at mean (=0)
    for c in CONTINUOUS_CONTEXT:
        row[f"{c}_z"] = 0.0

    # categorical variables fixed to their most frequent observed level
    for c in CATEGORICAL_CONTEXT:
        mode = temp[c].mode(dropna=True)
        if len(mode) == 0:
            raise ValueError(f"No valid level for {c}")
        row[c] = mode.iloc[0]

    return row


def contrast_design_vector(design_info, row_a, row_b):
    dm = build_design_matrices(
        [design_info],
        pd.DataFrame([row_a, row_b]),
        return_type="dataframe",
    )[0]

    vec_a = dm.iloc[0].to_numpy(dtype=float)
    vec_b = dm.iloc[1].to_numpy(dtype=float)

    # A minus B
    return vec_a - vec_b


# =============================================================================
# 4. LOAD INPUTS
# =============================================================================
print("=" * 112)
print("VitalDB Multimodal Physiological Coupling Study")
print("STEP 12D - Adjusted Post-hoc Interpretation")
print("=" * 112)

for path in [INPUT_DATA, INPUT_COMPARE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

df = pd.read_csv(INPUT_DATA)
df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

compare = pd.read_csv(INPUT_COMPARE)
compare.columns = (
    compare.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

print(f"\nCases: {len(df):,}")
print(f"Unique subjects: {df['subjectid'].nunique():,}")
print("Postoperative outcomes loaded/inspected: NO")

# =============================================================================
# 5. SELECT ONLY PRESPECIFIED ROBUST MODERATE/LARGE TARGETS
# =============================================================================
required_compare = [
    "coupling_feature",
    "domain",
    "context_variable",
    "FDR_lt_0_05_12B",
    "FDR_lt_0_05_12C",
    "incremental_effect_magnitude",
    "delta_adjusted_R2_12B",
    "delta_adjusted_R2_12C",
]

missing_compare = [
    c for c in required_compare
    if c not in compare.columns
]

if missing_compare:
    raise ValueError(
        "STEP 12C comparison file is missing columns:\n"
        + "\n".join(missing_compare)
    )

selected = compare.loc[
    compare["FDR_lt_0_05_12B"].astype(bool)
    & compare["FDR_lt_0_05_12C"].astype(bool)
    & compare["incremental_effect_magnitude"].isin(
        ["moderate", "large"]
    )
].copy()

selected = selected.sort_values(
    "delta_adjusted_R2_12B",
    ascending=False,
).reset_index(drop=True)

selected.to_csv(
    OUT_SELECTED,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 112)
print("PRESPECIFIED ROBUST MODERATE/LARGE TARGETS")
print("=" * 112)

print(f"\nSelected target associations: {len(selected)}")

if len(selected) == 0:
    raise RuntimeError(
        "No robust moderate/large associations were found."
    )

print(
    selected[
        [
            "coupling_feature",
            "domain",
            "context_variable",
            "delta_adjusted_R2_12B",
            "delta_adjusted_R2_12C",
            "incremental_effect_magnitude",
        ]
    ].to_string(
        index=False,
        formatters={
            "delta_adjusted_R2_12B": lambda x: f"{x:+.5f}",
            "delta_adjusted_R2_12C": lambda x: f"{x:+.5f}",
        }
    )
)

# =============================================================================
# 6. PREPARE COMMON ADJUSTMENT DATA
# =============================================================================
work = df.copy()

age_z, age_mean, age_sd = standardize_with_reference(work["age"])
bmi_z, bmi_mean, bmi_sd = standardize_with_reference(work["bmi"])

work["age_z"] = age_z
work["bmi_z"] = bmi_z

adjustment_columns = (
    ["age_z", "bmi_z"]
    + CATEGORICAL_CONTEXT
    + ["subjectid"]
)

common_complete_mask = work[
    adjustment_columns
].notna().all(axis=1)

print(
    f"\nCommon complete-case rows: "
    f"{int(common_complete_mask.sum()):,}"
)

print(
    f"Common complete-case subjects: "
    f"{work.loc[common_complete_mask, 'subjectid'].nunique():,}"
)

print(f"Age mean={age_mean:.3f}, SD={age_sd:.3f}")
print(f"BMI mean={bmi_mean:.3f}, SD={bmi_sd:.3f}")

# =============================================================================
# 7. FIT ONE ADJUSTED MODEL PER SELECTED TARGET FEATURE
# =============================================================================
pairwise_rows = []
continuous_rows = []
summary_rows = []

# Cache model fit by coupling feature because one feature could have multiple contexts.
model_cache = {}

for _, target in selected.iterrows():
    feature = target["coupling_feature"]
    context = target["context_variable"]
    domain = target["domain"]

    if feature not in work.columns:
        raise ValueError(f"Missing coupling feature: {feature}")

    if context not in ALL_CONTEXT:
        raise ValueError(f"Unexpected context variable: {context}")

    if feature not in model_cache:
        temp = work.loc[common_complete_mask].copy()

        temp["coupling_z"] = inverse_normal_transform(
            temp[feature]
        )

        temp = temp.dropna(
            subset=["coupling_z", "subjectid"]
        ).copy()

        model = smf.ols(
            formula=FULL_FORMULA,
            data=temp,
        )

        robust = model.fit(
            cov_type="cluster",
            cov_kwds={
                "groups": temp["subjectid"],
                "use_correction": True,
            },
        )

        model_cache[feature] = {
            "temp": temp,
            "model": model,
            "result": robust,
            "design_info": model.data.design_info,
        }

    temp = model_cache[feature]["temp"]
    result = model_cache[feature]["result"]
    design_info = model_cache[feature]["design_info"]

    # -------------------------------------------------------------------------
    # Categorical target -> all adjusted pairwise contrasts
    # -------------------------------------------------------------------------
    if context in CATEGORICAL_CONTEXT:
        levels = sorted_levels(temp[context])

        print(
            f"\n{feature} × {context}: "
            f"{len(levels)} levels, "
            f"{len(levels) * (len(levels) - 1) // 2} pairwise comparisons"
        )

        reference = make_reference_row(temp)

        family_rows = []

        for level_a, level_b in itertools.combinations(levels, 2):
            row_a = reference.copy()
            row_b = reference.copy()

            row_a[context] = level_a
            row_b[context] = level_b

            contrast = contrast_design_vector(
                design_info,
                row_a,
                row_b,
            )

            test = result.t_test(contrast)

            estimate = float(
                np.asarray(test.effect).squeeze()
            )

            se = float(
                np.asarray(test.sd).squeeze()
            )

            p_value = float(
                np.asarray(test.pvalue).squeeze()
            )

            ci = np.asarray(
                test.conf_int(alpha=0.05)
            ).squeeze()

            n_a = int(
                (temp[context] == level_a).sum()
            )

            n_b = int(
                (temp[context] == level_b).sum()
            )

            family_rows.append({
                "coupling_feature": feature,
                "domain": domain,
                "context_variable": context,
                "level_A": str(level_a),
                "level_B": str(level_b),
                "n_A": n_a,
                "n_B": n_b,
                "adjusted_difference_A_minus_B_z": estimate,
                "robust_SE": se,
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "p_value": p_value,
            })

        family_df = pd.DataFrame(family_rows)

        family_df["q_value_FDR_within_family"] = benjamini_hochberg(
            family_df["p_value"].to_numpy(dtype=float)
        )

        family_df["FDR_lt_0_05"] = (
            family_df["q_value_FDR_within_family"] < 0.05
        )

        family_df["direction"] = np.where(
            family_df["adjusted_difference_A_minus_B_z"] > 0,
            "A_higher_than_B",
            np.where(
                family_df["adjusted_difference_A_minus_B_z"] < 0,
                "A_lower_than_B",
                "equal",
            )
        )

        pairwise_rows.extend(
            family_df.to_dict("records")
        )

        sig_n = int(
            family_df["FDR_lt_0_05"].sum()
        )

        strongest = family_df.iloc[
            np.argmax(
                np.abs(
                    family_df[
                        "adjusted_difference_A_minus_B_z"
                    ].to_numpy(dtype=float)
                )
            )
        ]

        summary_rows.append({
            "coupling_feature": feature,
            "domain": domain,
            "context_variable": context,
            "context_type": "categorical",
            "levels_n": len(levels),
            "pairwise_tests_n": len(family_df),
            "FDR_significant_pairwise_n": sig_n,
            "strongest_pair": (
                f"{strongest['level_A']} vs {strongest['level_B']}"
            ),
            "strongest_adjusted_difference_z": float(
                strongest["adjusted_difference_A_minus_B_z"]
            ),
            "strongest_pair_q_value": float(
                strongest["q_value_FDR_within_family"]
            ),
            "delta_adjusted_R2_12B": float(
                target["delta_adjusted_R2_12B"]
            ),
            "delta_adjusted_R2_12C": float(
                target["delta_adjusted_R2_12C"]
            ),
        })

    # -------------------------------------------------------------------------
    # Continuous target -> standardized beta + interpretable unit contrast
    # -------------------------------------------------------------------------
    elif context in CONTINUOUS_CONTEXT:
        param = f"{context}_z"

        if param not in result.params.index:
            raise RuntimeError(
                f"Parameter not found in fitted model: {param}"
            )

        beta = float(result.params[param])
        se = float(result.bse[param])
        p_value = float(result.pvalues[param])

        ci = result.conf_int().loc[param]

        if context == "age":
            unit_change = 10.0
            standardized_change = unit_change / age_sd
            unit_label = "+10 years"
        elif context == "bmi":
            unit_change = 5.0
            standardized_change = unit_change / bmi_sd
            unit_label = "+5 kg/m²"
        else:
            unit_change = np.nan
            standardized_change = np.nan
            unit_label = ""

        effect_for_unit = (
            beta * standardized_change
            if np.isfinite(standardized_change)
            else np.nan
        )

        # Linear scaling of CI for same positive unit change
        ci_unit_lower = (
            float(ci.iloc[0]) * standardized_change
            if np.isfinite(standardized_change)
            else np.nan
        )

        ci_unit_upper = (
            float(ci.iloc[1]) * standardized_change
            if np.isfinite(standardized_change)
            else np.nan
        )

        continuous_rows.append({
            "coupling_feature": feature,
            "domain": domain,
            "context_variable": context,
            "standardized_beta_per_1SD_context": beta,
            "robust_SE": se,
            "ci_lower_beta": float(ci.iloc[0]),
            "ci_upper_beta": float(ci.iloc[1]),
            "p_value": p_value,
            "context_mean": (
                age_mean if context == "age" else bmi_mean
            ),
            "context_SD": (
                age_sd if context == "age" else bmi_sd
            ),
            "interpretable_change": unit_label,
            "adjusted_change_in_coupling_z": effect_for_unit,
            "ci_lower_change": ci_unit_lower,
            "ci_upper_change": ci_unit_upper,
            "delta_adjusted_R2_12B": float(
                target["delta_adjusted_R2_12B"]
            ),
            "delta_adjusted_R2_12C": float(
                target["delta_adjusted_R2_12C"]
            ),
        })

        summary_rows.append({
            "coupling_feature": feature,
            "domain": domain,
            "context_variable": context,
            "context_type": "continuous",
            "levels_n": np.nan,
            "pairwise_tests_n": np.nan,
            "FDR_significant_pairwise_n": np.nan,
            "strongest_pair": unit_label,
            "strongest_adjusted_difference_z": effect_for_unit,
            "strongest_pair_q_value": p_value,
            "delta_adjusted_R2_12B": float(
                target["delta_adjusted_R2_12B"]
            ),
            "delta_adjusted_R2_12C": float(
                target["delta_adjusted_R2_12C"]
            ),
        })

# =============================================================================
# 8. SAVE OUTPUTS
# =============================================================================
pairwise_df = pd.DataFrame(pairwise_rows)

if len(pairwise_df):
    pairwise_df = pairwise_df.sort_values(
        [
            "coupling_feature",
            "context_variable",
            "q_value_FDR_within_family",
            "p_value",
        ]
    ).reset_index(drop=True)

pairwise_df.to_csv(
    OUT_PAIRWISE,
    index=False,
    encoding="utf-8-sig",
)

continuous_df = pd.DataFrame(continuous_rows)

continuous_df.to_csv(
    OUT_CONTINUOUS,
    index=False,
    encoding="utf-8-sig",
)

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig",
)

# =============================================================================
# 9. CONSOLE OUTPUT
# =============================================================================
print("\n" + "=" * 112)
print("STEP 12D POST-HOC SUMMARY")
print("=" * 112)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "strongest_adjusted_difference_z":
                lambda x: (
                    ""
                    if pd.isna(x)
                    else f"{x:+.4f}"
                ),
            "strongest_pair_q_value":
                lambda x: (
                    ""
                    if pd.isna(x)
                    else f"{x:.4g}"
                ),
            "delta_adjusted_R2_12B":
                lambda x: f"{x:+.5f}",
            "delta_adjusted_R2_12C":
                lambda x: f"{x:+.5f}",
        }
    )
)

# =============================================================================
# 10. PRINT CONTINUOUS EFFECTS
# =============================================================================
print("\n" + "=" * 112)
print("ADJUSTED CONTINUOUS EFFECTS")
print("=" * 112)

if len(continuous_df) == 0:
    print("No continuous robust-moderate target.")
else:
    print(
        continuous_df[
            [
                "coupling_feature",
                "context_variable",
                "standardized_beta_per_1SD_context",
                "ci_lower_beta",
                "ci_upper_beta",
                "p_value",
                "interpretable_change",
                "adjusted_change_in_coupling_z",
                "ci_lower_change",
                "ci_upper_change",
            ]
        ].to_string(
            index=False,
            formatters={
                "standardized_beta_per_1SD_context":
                    lambda x: f"{x:+.4f}",
                "ci_lower_beta":
                    lambda x: f"{x:+.4f}",
                "ci_upper_beta":
                    lambda x: f"{x:+.4f}",
                "p_value":
                    lambda x: f"{x:.4g}",
                "adjusted_change_in_coupling_z":
                    lambda x: f"{x:+.4f}",
                "ci_lower_change":
                    lambda x: f"{x:+.4f}",
                "ci_upper_change":
                    lambda x: f"{x:+.4f}",
            }
        )
    )

# =============================================================================
# 11. PRINT TOP SIGNIFICANT PAIRWISE RESULTS
# =============================================================================
print("\n" + "=" * 112)
print("TOP 40 FDR-SIGNIFICANT ADJUSTED PAIRWISE COMPARISONS")
print("=" * 112)

if len(pairwise_df) == 0:
    print("No categorical post-hoc targets.")
else:
    significant = pairwise_df.loc[
        pairwise_df["FDR_lt_0_05"]
    ].copy()

    significant["abs_difference"] = np.abs(
        significant["adjusted_difference_A_minus_B_z"]
    )

    significant = significant.sort_values(
        [
            "abs_difference",
            "q_value_FDR_within_family",
        ],
        ascending=[
            False,
            True,
        ]
    ).head(40)

    if len(significant) == 0:
        print(
            "No pairwise comparison remained significant "
            "after within-family FDR correction."
        )
    else:
        print(
            significant[
                [
                    "coupling_feature",
                    "context_variable",
                    "level_A",
                    "level_B",
                    "n_A",
                    "n_B",
                    "adjusted_difference_A_minus_B_z",
                    "ci_lower",
                    "ci_upper",
                    "q_value_FDR_within_family",
                    "direction",
                ]
            ].to_string(
                index=False,
                formatters={
                    "adjusted_difference_A_minus_B_z":
                        lambda x: f"{x:+.4f}",
                    "ci_lower":
                        lambda x: f"{x:+.4f}",
                    "ci_upper":
                        lambda x: f"{x:+.4f}",
                    "q_value_FDR_within_family":
                        lambda x: f"{x:.4g}",
                }
            )
        )

# =============================================================================
# 12. FINAL RULE
# =============================================================================
print("\n" + "=" * 112)
print("STEP 12D FINAL ANALYSIS RULE")
print("=" * 112)

print("""
STEP 12D is the final interpretability analysis.

Use these results only to explain the direction and group structure of the
already-established robust moderate associations.

Do NOT:
- add new features,
- change the analysis window,
- create new subgroups,
- tune prediction models,
- search for additional interactions,
- promote a post-hoc result to a new confirmatory hypothesis.

After STEP 12D, statistical exploration should stop and manuscript preparation
should begin.
""")

print("Generated files:")
for path in [
    OUT_SELECTED,
    OUT_PAIRWISE,
    OUT_CONTINUOUS,
    OUT_SUMMARY,
]:
    print(path)

print(
    """
NEXT:
Send:
1) PRESPECIFIED ROBUST MODERATE/LARGE TARGETS
2) STEP 12D POST-HOC SUMMARY
3) ADJUSTED CONTINUOUS EFFECTS
4) TOP 40 FDR-SIGNIFICANT ADJUSTED PAIRWISE COMPARISONS

Then we can translate the final findings into the exact BSPC Results section,
tables, figure captions, and Discussion.
"""
)