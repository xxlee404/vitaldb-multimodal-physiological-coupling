# -*- coding: utf-8 -*-
"""
VitalDB Multimodal Physiological Coupling Study
STEP 07D - Freeze Clinical Baseline (Model A) Before Outcome Modeling

Purpose
-------
1) Freeze the exact primary clinical covariate block BEFORE any outcome is loaded.
2) Audit missingness and basic redundancy/association without using outcomes.
3) Preserve the previously frozen signal blocks:
   - Primary conventional = 14
   - Sensitivity conventional = 20
   - Coupling = 12
4) Save a clean pre-outcome modeling dataset and a written analysis specification.

IMPORTANT
---------
No postoperative outcome is loaded or inspected in this script.
No variable is selected based on prediction performance.

Primary clinical baseline:
- age
- sex
- bmi
- asa
- emop
- department
- ane_type
- optype
- approach
- position
- preop_htn
- preop_dm

height and weight are not used in the PRIMARY clinical block because BMI already
summarizes body size and including all three would introduce unnecessary redundancy.
They remain available in the source dataset for sensitivity analyses if needed.
"""

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PATHS
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data_processed"
RESULT_DIR = PROJECT_DIR / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "07C_preoutcome_analysis_dataset.csv"

OUT_DATA = PROCESSED_DIR / "07D_preoutcome_modeling_dataset_frozen.csv"
OUT_CLINICAL_AUDIT = RESULT_DIR / "07D_modelA_clinical_audit.csv"
OUT_NUMERIC_CORR = RESULT_DIR / "07D_modelA_numeric_spearman.csv"
OUT_CATEGORICAL_ASSOC = RESULT_DIR / "07D_modelA_categorical_cramers_v.csv"
OUT_HIGH_ASSOC = RESULT_DIR / "07D_modelA_high_association_pairs.csv"
OUT_SPEC_JSON = RESULT_DIR / "07D_FINAL_preoutcome_model_specification.json"
OUT_SPEC_TXT = RESULT_DIR / "07D_FINAL_preoutcome_model_specification.txt"

# =============================================================================
# 2. FROZEN FEATURE BLOCKS
# =============================================================================

MODEL_A_CLINICAL = [
    "age",
    "sex",
    "bmi",
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

EXCLUDED_FROM_PRIMARY_CLINICAL = {
    "height": "BMI retained as primary body-size representation.",
    "weight": "BMI retained as primary body-size representation.",
}

CONVENTIONAL_PRIMARY14 = [
    "hr_mean", "hr_std", "hr_min", "hr_max",
    "map_mean", "map_std", "map_min", "map_max",
    "spo2_mean", "spo2_min",
    "etco2_mean", "etco2_std", "etco2_min", "etco2_max",
]

CONVENTIONAL_SENSITIVITY20 = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
    "map_mean", "map_std", "map_min", "map_max", "map_range",
    "spo2_mean", "spo2_std", "spo2_min", "spo2_max", "spo2_range",
    "etco2_mean", "etco2_std", "etco2_min", "etco2_max", "etco2_range",
]

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

# Numerical variables for simple Spearman audit.
NUMERIC_AUDIT = [
    "age",
    "bmi",
    "asa",
    "emop",
    "preop_htn",
    "preop_dm",
]

# Categorical/ordinal association audit.
CATEGORICAL_AUDIT = [
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

REVIEW_ASSOC = 0.70
STRONG_ASSOC = 0.85
VERY_STRONG_ASSOC = 0.90

# Final future ML settings, frozen before outcomes.
CV_SPLITTER = "StratifiedGroupKFold"
CV_FOLDS = 5
CV_SHUFFLE = True
CV_RANDOM_STATE = 2026
GROUP_COLUMN = "subjectid"

# =============================================================================
# 3. HELPERS
# =============================================================================

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V, pairwise complete observations."""
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(tmp) == 0:
        return np.nan

    table = pd.crosstab(tmp["x"], tmp["y"])

    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan

    observed = table.to_numpy(dtype=float)
    n = observed.sum()

    if n <= 1:
        return np.nan

    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n

    valid = expected > 0
    chi2 = np.sum(((observed - expected) ** 2)[valid] / expected[valid])

    phi2 = chi2 / n
    r, k = observed.shape

    # Bias correction
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    denom = min(kcorr - 1, rcorr - 1)

    if denom <= 0:
        return np.nan

    return float(math.sqrt(phi2corr / denom))


def association_level(value: float) -> str:
    if pd.isna(value):
        return "NA"
    value = abs(float(value))
    if value >= VERY_STRONG_ASSOC:
        return "VERY_STRONG"
    if value >= STRONG_ASSOC:
        return "STRONG"
    if value >= REVIEW_ASSOC:
        return "REVIEW"
    return "OK"


# =============================================================================
# 4. LOAD DATA
# =============================================================================
print("=" * 100)
print("VitalDB Multimodal Physiological Coupling Study")
print("STEP 07D - Freeze Clinical Baseline (Model A) Before Outcome Modeling")
print("=" * 100)

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"입력 파일이 없습니다:\n{INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

required = (
    ["caseid", "subjectid"]
    + MODEL_A_CLINICAL
    + CONVENTIONAL_PRIMARY14
    + COUPLING12
)

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        "필수 컬럼이 없습니다:\n" + "\n".join(missing)
    )

print(f"\nCases: {len(df):,}")
print(f"Unique subjects: {df['subjectid'].nunique():,}")
print(f"Frozen Model A clinical covariates: {len(MODEL_A_CLINICAL)}")
print(f"Frozen conventional primary features: {len(CONVENTIONAL_PRIMARY14)}")
print(f"Frozen coupling features: {len(COUPLING12)}")
print("Outcome variables loaded/inspected: NO")

# =============================================================================
# 5. SHOW FINAL MODEL A
# =============================================================================
print("\n" + "=" * 100)
print("FINAL FROZEN MODEL A CLINICAL COVARIATES")
print("=" * 100)

for i, feature in enumerate(MODEL_A_CLINICAL, start=1):
    print(f"{i:02d}. {feature}")

print("\nExcluded from PRIMARY clinical block:")
for feature, reason in EXCLUDED_FROM_PRIMARY_CLINICAL.items():
    print(f"- {feature}: {reason}")

# =============================================================================
# 6. CLINICAL MISSINGNESS / LEVEL AUDIT
# =============================================================================
print("\n" + "=" * 100)
print("MODEL A CLINICAL AUDIT")
print("=" * 100)

audit_rows = []

for feature in MODEL_A_CLINICAL:
    s = df[feature]
    row = {
        "feature": feature,
        "missing_n": int(s.isna().sum()),
        "missing_pct": float(s.isna().mean() * 100.0),
        "unique_n": int(s.nunique(dropna=True)),
        "dtype": str(s.dtype),
    }

    if pd.api.types.is_numeric_dtype(s):
        numeric = pd.to_numeric(s, errors="coerce")
        row["median"] = float(numeric.median()) if numeric.notna().any() else np.nan
        row["q1"] = float(numeric.quantile(0.25)) if numeric.notna().any() else np.nan
        row["q3"] = float(numeric.quantile(0.75)) if numeric.notna().any() else np.nan
    else:
        row["median"] = np.nan
        row["q1"] = np.nan
        row["q3"] = np.nan

    audit_rows.append(row)

    print(
        f"{feature:<15} "
        f"missing={row['missing_n']:>4} "
        f"({row['missing_pct']:.2f}%) | "
        f"unique={row['unique_n']:>3} | "
        f"dtype={row['dtype']}"
    )

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(OUT_CLINICAL_AUDIT, index=False, encoding="utf-8-sig")

# =============================================================================
# 7. NUMERIC SPEARMAN AUDIT
# =============================================================================
print("\n" + "=" * 100)
print("MODEL A NUMERIC SPEARMAN AUDIT")
print("=" * 100)

num = df[NUMERIC_AUDIT].apply(pd.to_numeric, errors="coerce")
numeric_corr = num.corr(method="spearman")
numeric_corr.to_csv(OUT_NUMERIC_CORR, encoding="utf-8-sig")

numeric_pairs = []

for i in range(len(NUMERIC_AUDIT)):
    for j in range(i + 1, len(NUMERIC_AUDIT)):
        f1 = NUMERIC_AUDIT[i]
        f2 = NUMERIC_AUDIT[j]
        rho = numeric_corr.loc[f1, f2]

        if pd.isna(rho):
            continue

        numeric_pairs.append({
            "association_type": "spearman",
            "feature_1": f1,
            "feature_2": f2,
            "association": float(rho),
            "abs_association": abs(float(rho)),
            "flag_level": association_level(rho),
        })

numeric_pairs_df = pd.DataFrame(numeric_pairs)

num_review = numeric_pairs_df.loc[
    numeric_pairs_df["abs_association"] >= REVIEW_ASSOC
].sort_values("abs_association", ascending=False)

if len(num_review) == 0:
    print("No numeric Model A pairs with |rho| >= 0.70.")
else:
    print(num_review.to_string(index=False))

# =============================================================================
# 8. CATEGORICAL CRAMER'S V AUDIT
# =============================================================================
print("\n" + "=" * 100)
print("MODEL A CATEGORICAL CRAMER'S V AUDIT")
print("=" * 100)

cat_matrix = pd.DataFrame(
    np.eye(len(CATEGORICAL_AUDIT)),
    index=CATEGORICAL_AUDIT,
    columns=CATEGORICAL_AUDIT,
    dtype=float,
)

cat_pairs = []

for i in range(len(CATEGORICAL_AUDIT)):
    for j in range(i + 1, len(CATEGORICAL_AUDIT)):
        f1 = CATEGORICAL_AUDIT[i]
        f2 = CATEGORICAL_AUDIT[j]

        v = cramers_v(df[f1], df[f2])
        cat_matrix.loc[f1, f2] = v
        cat_matrix.loc[f2, f1] = v

        if pd.isna(v):
            continue

        cat_pairs.append({
            "association_type": "cramers_v",
            "feature_1": f1,
            "feature_2": f2,
            "association": float(v),
            "abs_association": abs(float(v)),
            "flag_level": association_level(v),
        })

cat_matrix.to_csv(OUT_CATEGORICAL_ASSOC, encoding="utf-8-sig")

cat_pairs_df = pd.DataFrame(cat_pairs)

cat_review = cat_pairs_df.loc[
    cat_pairs_df["abs_association"] >= REVIEW_ASSOC
].sort_values("abs_association", ascending=False)

if len(cat_review) == 0:
    print("No categorical Model A pairs with Cramer's V >= 0.70.")
else:
    print(cat_review.to_string(index=False))

# =============================================================================
# 9. COMBINE HIGH-ASSOCIATION PAIRS
# =============================================================================
all_pairs = pd.concat(
    [numeric_pairs_df, cat_pairs_df],
    ignore_index=True,
)

high_pairs = all_pairs.loc[
    all_pairs["abs_association"] >= REVIEW_ASSOC
].sort_values(
    ["abs_association", "feature_1", "feature_2"],
    ascending=[False, True, True],
)

high_pairs.to_csv(OUT_HIGH_ASSOC, index=False, encoding="utf-8-sig")

max_assoc = (
    float(all_pairs["abs_association"].max())
    if len(all_pairs)
    else np.nan
)

n70 = int((all_pairs["abs_association"] >= REVIEW_ASSOC).sum())
n85 = int((all_pairs["abs_association"] >= STRONG_ASSOC).sum())
n90 = int((all_pairs["abs_association"] >= VERY_STRONG_ASSOC).sum())

# =============================================================================
# 10. SAVE FROZEN PRE-OUTCOME MODELING DATASET
# =============================================================================
# Keep all existing columns because this is still a no-outcome dataset.
# The feature blocks are defined explicitly in the specification files.
df.to_csv(
    OUT_DATA,
    index=False,
    encoding="utf-8-sig",
)

# =============================================================================
# 11. FINAL MODEL SPECIFICATION
# =============================================================================
spec = {
    "stage": "STEP 07D - FINAL PRE-OUTCOME MODEL FREEZE",
    "outcomes_inspected": False,
    "cohort": {
        "cases": int(len(df)),
        "unique_subjects": int(df["subjectid"].nunique()),
    },
    "model_A_clinical": MODEL_A_CLINICAL,
    "excluded_primary_clinical": EXCLUDED_FROM_PRIMARY_CLINICAL,
    "model_B": {
        "description": "Model A + 14 primary conventional vital-sign features",
        "conventional_features": CONVENTIONAL_PRIMARY14,
    },
    "model_C": {
        "description": "Model B + all 12 locked coupling features",
        "coupling_features": COUPLING12,
    },
    "sensitivity_models": {
        "B20": (
            "Model A + all 20 original conventional vital-sign features"
        ),
        "C20": (
            "Model A + all 20 original conventional vital-sign features "
            "+ all 12 coupling features"
        ),
        "conventional20": CONVENTIONAL_SENSITIVITY20,
    },
    "missing_data_preprocessing": {
        "numeric": (
            "Median imputation fitted on training fold only. "
            "This applies to ASA and any other numeric missing values."
        ),
        "categorical": (
            "Most-frequent imputation fitted on training fold only, "
            "then one-hot encoding with unknown categories ignored."
        ),
    },
    "modeling_preprocessing": {
        "logistic_regression": (
            "Numeric scaling with StandardScaler fitted on training fold only."
        ),
        "tree_models": (
            "No scaling required; imputation/encoding still fit inside folds."
        ),
    },
    "cross_validation": {
        "splitter": CV_SPLITTER,
        "n_splits": CV_FOLDS,
        "shuffle": CV_SHUFFLE,
        "random_state": CV_RANDOM_STATE,
        "group_column": GROUP_COLUMN,
        "rule": (
            "All cases from the same subjectid remain in the same fold."
        ),
    },
    "primary_hypothesis": (
        "Adding the 12 locked multimodal physiological coupling features "
        "to a clinical + conventional vital-sign baseline provides "
        "incremental predictive information."
    ),
    "primary_incremental_comparison": "Model C versus Model B",
    "secondary_comparison": "Model B versus Model A",
    "outcome_driven_feature_selection": False,
}

OUT_SPEC_JSON.write_text(
    json.dumps(spec, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

spec_text = f"""
VitalDB Multimodal Physiological Coupling Study
STEP 07D - FINAL PRE-OUTCOME MODEL SPECIFICATION
============================================================

OUTCOMES INSPECTED
------------------
NO

COHORT
------
Cases: {len(df):,}
Unique subjects: {df['subjectid'].nunique():,}

MODEL A - CLINICAL
------------------
{chr(10).join(MODEL_A_CLINICAL)}

height and weight are excluded from the PRIMARY clinical block because
BMI is retained as the primary body-size representation.

MODEL B - CLINICAL + CONVENTIONAL SIGNAL
----------------------------------------
Model A plus 14 primary conventional features:
{chr(10).join(CONVENTIONAL_PRIMARY14)}

MODEL C - CLINICAL + CONVENTIONAL + COUPLING
---------------------------------------------
Model B plus all 12 locked coupling features:
{chr(10).join(COUPLING12)}

PRIMARY HYPOTHESIS
------------------
Do the locked coupling features add predictive information beyond the
clinical + conventional vital-sign baseline?

PRIMARY COMPARISON
------------------
Model C versus Model B

SECONDARY COMPARISON
--------------------
Model B versus Model A

SENSITIVITY ANALYSIS
--------------------
B20 = Model A + all original 20 conventional features
C20 = B20 + all 12 coupling features

CROSS-VALIDATION
----------------
StratifiedGroupKFold
n_splits = 5
shuffle = True
random_state = 2026
group = subjectid

All repeated cases from the same subject remain in the same fold.

PREPROCESSING
-------------
Numeric missing values:
- training-fold median imputation

Categorical missing values:
- training-fold most-frequent imputation
- one-hot encoding
- unknown categories ignored

Logistic regression:
- StandardScaler fitted on training fold only

Tree models:
- no scaling required

NO outcome-driven feature selection is permitted.
"""

OUT_SPEC_TXT.write_text(
    spec_text.strip() + "\n",
    encoding="utf-8",
)

# =============================================================================
# 12. SUMMARY
# =============================================================================
features_with_missing = int((audit_df["missing_n"] > 0).sum())
max_missing_pct = float(audit_df["missing_pct"].max())

print("\n" + "=" * 100)
print("STEP 07D FINAL PRE-OUTCOME SUMMARY")
print("=" * 100)

print(f"\nCases: {len(df):,}")
print(f"Unique subjects: {df['subjectid'].nunique():,}")
print(f"Model A clinical variables: {len(MODEL_A_CLINICAL)}")
print(f"Model B added conventional features: {len(CONVENTIONAL_PRIMARY14)}")
print(f"Model C added coupling features: {len(COUPLING12)}")

print(f"\nClinical variables with missing values: {features_with_missing}")
print(f"Maximum clinical missingness: {max_missing_pct:.2f}%")

print(f"\nMaximum audited clinical association: {max_assoc:.4f}")
print(f"Association >= 0.70 pairs: {n70}")
print(f"Association >= 0.85 pairs: {n85}")
print(f"Association >= 0.90 pairs: {n90}")

print("\nFINAL FROZEN MODEL STRUCTURE:")
print("Model A = Clinical 12")
print("Model B = Model A + Conventional 14")
print("Model C = Model B + Coupling 12")
print("Primary incremental test = Model C versus Model B")
print("Outcome variables inspected: NO")

print("\n" + "=" * 100)
print("STEP 07D DECISION")
print("=" * 100)

if max_missing_pct < 5.0 and n90 == 0:
    print("\nDecision: READY FOR OUTCOME MERGE")
    print(
        "Reason: Clinical baseline is complete enough and was frozen "
        "without using postoperative outcomes."
    )
elif max_missing_pct < 10.0:
    print("\nDecision: READY WITH CAUTION")
    print(
        "Reason: Clinical missingness is manageable, but review any very "
        "strong pre-outcome clinical associations before modeling."
    )
else:
    print("\nDecision: REVIEW")
    print("Reason: Clinical missingness/association requires review.")

print("\nGenerated files:")
for p in [
    OUT_DATA,
    OUT_CLINICAL_AUDIT,
    OUT_NUMERIC_CORR,
    OUT_CATEGORICAL_ASSOC,
    OUT_HIGH_ASSOC,
    OUT_SPEC_JSON,
    OUT_SPEC_TXT,
]:
    print(p)

print(
    """
NEXT:
Send:
1) MODEL A NUMERIC SPEARMAN AUDIT
2) MODEL A CATEGORICAL CRAMER'S V AUDIT
3) STEP 07D FINAL PRE-OUTCOME SUMMARY
4) STEP 07D DECISION

If READY, the next step will merge the prespecified postoperative outcome
and inspect its prevalence WITHOUT changing any frozen features.
"""
)