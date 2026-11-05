# -*- coding: utf-8 -*-
"""
VitalDB Multimodal Physiological Coupling Study
STEP 07B - Outcome-Independent Conventional / Coupling Feature Audit

Purpose
-------
1) Audit redundancy among the 20 locked conventional vital-sign features.
2) Audit overlap between the 20 conventional features and 12 locked coupling features.
3) Confirm finite/missing/variance structure.
4) Record the pre-outcome model/CV specification.
5) Do NOT load, merge, or inspect postoperative outcomes.

This script does NOT automatically remove any feature.
Any redundancy decision must be made using only signal-processing/statistical meaning
before outcome modeling.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PATHS
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data_processed"
RESULT_DIR = PROJECT_DIR / "results"
FIGURE_DIR = PROJECT_DIR / "figures" / "07B_feature_block_audit"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "07_analysis_features_without_outcomes.csv"

OUT_CONV_AUDIT = RESULT_DIR / "07B_conventional_feature_audit.csv"
OUT_CONV_CORR = RESULT_DIR / "07B_conventional_spearman_matrix.csv"
OUT_CONV_PAIRS = RESULT_DIR / "07B_conventional_correlation_pairs.csv"
OUT_CONV_HIGH = RESULT_DIR / "07B_conventional_high_correlation_pairs.csv"

OUT_CROSS_CORR = RESULT_DIR / "07B_conventional_vs_coupling_spearman_matrix.csv"
OUT_CROSS_PAIRS = RESULT_DIR / "07B_conventional_vs_coupling_pairs.csv"
OUT_CROSS_HIGH = RESULT_DIR / "07B_conventional_vs_coupling_high_pairs.csv"

OUT_ALL_CORR = RESULT_DIR / "07B_all32_spearman_matrix.csv"
OUT_CLINICAL_INVENTORY = RESULT_DIR / "07B_clinical_candidate_inventory.csv"
OUT_SUMMARY = RESULT_DIR / "07B_feature_block_audit_summary.csv"
OUT_SPEC_JSON = RESULT_DIR / "07B_preoutcome_analysis_specification.json"
OUT_SPEC_TXT = RESULT_DIR / "07B_preoutcome_analysis_specification.txt"

FIG_CONV = FIGURE_DIR / "07B_conventional_spearman_heatmap.png"
FIG_ALL = FIGURE_DIR / "07B_all32_spearman_heatmap.png"

# =============================================================================
# 2. LOCKED FEATURE DEFINITIONS
# =============================================================================
CONVENTIONAL_FEATURES = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
    "map_mean", "map_std", "map_min", "map_max", "map_range",
    "spo2_mean", "spo2_std", "spo2_min", "spo2_max", "spo2_range",
    "etco2_mean", "etco2_std", "etco2_min", "etco2_max", "etco2_range",
]

COUPLING_FEATURES = [
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

ALL_SIGNAL_FEATURES = CONVENTIONAL_FEATURES + COUPLING_FEATURES

# Thresholds fixed before outcomes.
REVIEW_RHO = 0.70
STRONG_RHO = 0.85
VERY_STRONG_RHO = 0.90
NEAR_ZERO_STD = 1e-12

# Future CV rule fixed before outcomes.
CV_FOLDS = 5
CV_RANDOM_STATE = 2026
CV_GROUP_COLUMN = "subjectid"

# =============================================================================
# 3. HELPERS
# =============================================================================
def pair_table_from_corr(corr: pd.DataFrame, block_name: str) -> pd.DataFrame:
    rows = []
    cols = list(corr.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = corr.iloc[i, j]
            if pd.isna(rho):
                continue

            abs_rho = abs(float(rho))

            if abs_rho >= VERY_STRONG_RHO:
                level = "VERY_STRONG"
            elif abs_rho >= STRONG_RHO:
                level = "STRONG"
            elif abs_rho >= REVIEW_RHO:
                level = "REVIEW"
            else:
                level = "OK"

            rows.append({
                "block": block_name,
                "feature_1": cols[i],
                "feature_2": cols[j],
                "spearman_rho": float(rho),
                "abs_rho": abs_rho,
                "flag_level": level,
            })

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(
            ["abs_rho", "feature_1", "feature_2"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    return out


def cross_pair_table(corr: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for conventional in corr.index:
        for coupling in corr.columns:
            rho = corr.loc[conventional, coupling]
            if pd.isna(rho):
                continue

            abs_rho = abs(float(rho))

            if abs_rho >= VERY_STRONG_RHO:
                level = "VERY_STRONG"
            elif abs_rho >= STRONG_RHO:
                level = "STRONG"
            elif abs_rho >= REVIEW_RHO:
                level = "REVIEW"
            else:
                level = "OK"

            rows.append({
                "conventional_feature": conventional,
                "coupling_feature": coupling,
                "spearman_rho": float(rho),
                "abs_rho": abs_rho,
                "flag_level": level,
            })

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(
            ["abs_rho", "conventional_feature", "coupling_feature"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    return out


def save_heatmap(corr: pd.DataFrame, title: str, path: Path, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 4. LOAD DATA
# =============================================================================
print("=" * 100)
print("VitalDB Multimodal Physiological Coupling Study")
print("STEP 07B - Outcome-Independent Conventional / Coupling Feature Audit")
print("=" * 100)

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"입력 파일이 없습니다:\n{INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

required_id = ["caseid", "subjectid"]
missing_id = [c for c in required_id if c not in df.columns]
if missing_id:
    raise ValueError(f"필수 ID 컬럼이 없습니다: {missing_id}")

missing_features = [f for f in ALL_SIGNAL_FEATURES if f not in df.columns]
if missing_features:
    raise ValueError(
        "필수 feature가 없습니다:\n" + "\n".join(missing_features)
    )

print(f"\nInput cases: {len(df):,}")
print(f"Unique subjects: {df['subjectid'].nunique():,}")
print(f"Conventional features: {len(CONVENTIONAL_FEATURES)}")
print(f"Coupling features: {len(COUPLING_FEATURES)}")
print("Outcome variables: NOT USED")

# =============================================================================
# 5. BASIC AUDIT
# =============================================================================
print("\n" + "=" * 100)
print("BASIC FEATURE AUDIT")
print("=" * 100)

audit_rows = []

for feature in CONVENTIONAL_FEATURES:
    values = pd.to_numeric(df[feature], errors="coerce")
    arr = values.to_numpy(dtype=float)
    inf_n = int(np.isinf(arr).sum())
    finite = values.replace([np.inf, -np.inf], np.nan)

    missing_n = int(values.isna().sum())
    unique_n = int(finite.nunique(dropna=True))
    std_value = float(finite.std(ddof=1)) if finite.notna().sum() > 1 else np.nan
    near_zero = bool(pd.notna(std_value) and abs(std_value) <= NEAR_ZERO_STD)

    audit_rows.append({
        "feature": feature,
        "missing_n": missing_n,
        "missing_pct": missing_n / len(df) * 100.0,
        "inf_n": inf_n,
        "unique_n": unique_n,
        "std": std_value,
        "near_zero_variance": near_zero,
    })

    print(
        f"{feature:<20} "
        f"missing={missing_n:>4} | "
        f"inf={inf_n:>3} | "
        f"unique={unique_n:>4} | "
        f"std={std_value:.6f}"
    )

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(OUT_CONV_AUDIT, index=False, encoding="utf-8-sig")

numeric_df = df[ALL_SIGNAL_FEATURES].apply(pd.to_numeric, errors="coerce")
numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

# =============================================================================
# 6. CONVENTIONAL-CONVENTIONAL CORRELATION
# =============================================================================
print("\n" + "=" * 100)
print("CONVENTIONAL FEATURE PAIRWISE SPEARMAN CORRELATION")
print("=" * 100)

conv_corr = numeric_df[CONVENTIONAL_FEATURES].corr(method="spearman")
conv_corr.to_csv(OUT_CONV_CORR, encoding="utf-8-sig")

conv_pairs = pair_table_from_corr(conv_corr, "conventional")
conv_pairs.to_csv(OUT_CONV_PAIRS, index=False, encoding="utf-8-sig")

conv_high = conv_pairs.loc[conv_pairs["abs_rho"] >= REVIEW_RHO].copy()
conv_high.to_csv(OUT_CONV_HIGH, index=False, encoding="utf-8-sig")

if len(conv_high) == 0:
    print("\nNo conventional pairs with |rho| >= 0.70.")
else:
    print("\nConventional pairs with |rho| >= 0.70:")
    print(
        conv_high[
            ["feature_1", "feature_2", "spearman_rho", "abs_rho", "flag_level"]
        ].to_string(
            index=False,
            formatters={
                "spearman_rho": lambda x: f"{x:.4f}",
                "abs_rho": lambda x: f"{x:.4f}",
            }
        )
    )

conv_max = float(conv_pairs["abs_rho"].max()) if len(conv_pairs) else np.nan
conv_n70 = int((conv_pairs["abs_rho"] >= REVIEW_RHO).sum()) if len(conv_pairs) else 0
conv_n85 = int((conv_pairs["abs_rho"] >= STRONG_RHO).sum()) if len(conv_pairs) else 0
conv_n90 = int((conv_pairs["abs_rho"] >= VERY_STRONG_RHO).sum()) if len(conv_pairs) else 0

print(f"\nConventional maximum |rho| = {conv_max:.4f}")
print(f"|rho| >= 0.70 pairs = {conv_n70}")
print(f"|rho| >= 0.85 pairs = {conv_n85}")
print(f"|rho| >= 0.90 pairs = {conv_n90}")

save_heatmap(
    conv_corr,
    "Conventional Vital Features - Spearman Correlation",
    FIG_CONV,
    figsize=(11, 9),
)

# =============================================================================
# 7. CONVENTIONAL VS COUPLING CORRELATION
# =============================================================================
print("\n" + "=" * 100)
print("CONVENTIONAL VS COUPLING SPEARMAN CORRELATION")
print("=" * 100)

all_corr = numeric_df[ALL_SIGNAL_FEATURES].corr(method="spearman")
all_corr.to_csv(OUT_ALL_CORR, encoding="utf-8-sig")

cross_corr = all_corr.loc[CONVENTIONAL_FEATURES, COUPLING_FEATURES]
cross_corr.to_csv(OUT_CROSS_CORR, encoding="utf-8-sig")

cross_pairs = cross_pair_table(cross_corr)
cross_pairs.to_csv(OUT_CROSS_PAIRS, index=False, encoding="utf-8-sig")

cross_high = cross_pairs.loc[cross_pairs["abs_rho"] >= REVIEW_RHO].copy()
cross_high.to_csv(OUT_CROSS_HIGH, index=False, encoding="utf-8-sig")

if len(cross_high) == 0:
    print("\nNo conventional-vs-coupling pairs with |rho| >= 0.70.")
else:
    print("\nConventional-vs-coupling pairs with |rho| >= 0.70:")
    print(
        cross_high[
            [
                "conventional_feature",
                "coupling_feature",
                "spearman_rho",
                "abs_rho",
                "flag_level",
            ]
        ].to_string(
            index=False,
            formatters={
                "spearman_rho": lambda x: f"{x:.4f}",
                "abs_rho": lambda x: f"{x:.4f}",
            }
        )
    )

cross_max = float(cross_pairs["abs_rho"].max()) if len(cross_pairs) else np.nan
cross_n70 = int((cross_pairs["abs_rho"] >= REVIEW_RHO).sum()) if len(cross_pairs) else 0
cross_n85 = int((cross_pairs["abs_rho"] >= STRONG_RHO).sum()) if len(cross_pairs) else 0
cross_n90 = int((cross_pairs["abs_rho"] >= VERY_STRONG_RHO).sum()) if len(cross_pairs) else 0

print(f"\nCross-block maximum |rho| = {cross_max:.4f}")
print(f"|rho| >= 0.70 pairs = {cross_n70}")
print(f"|rho| >= 0.85 pairs = {cross_n85}")
print(f"|rho| >= 0.90 pairs = {cross_n90}")

save_heatmap(
    all_corr,
    "All 32 Signal Features - Spearman Correlation",
    FIG_ALL,
    figsize=(16, 14),
)

# =============================================================================
# 8. SUBJECT STRUCTURE
# =============================================================================
print("\n" + "=" * 100)
print("SUBJECT / REPEATED-CASE STRUCTURE")
print("=" * 100)

subject_counts = df.groupby("subjectid").size()

single_subjects = int((subject_counts == 1).sum())
repeated_subjects = int((subject_counts > 1).sum())
extra_rows = int((subject_counts - 1).clip(lower=0).sum())
max_cases_per_subject = int(subject_counts.max())

print(f"Cases: {len(df):,}")
print(f"Unique subjects: {subject_counts.size:,}")
print(f"Single-case subjects: {single_subjects:,}")
print(f"Repeated-case subjects: {repeated_subjects:,}")
print(f"Extra repeated case rows: {extra_rows:,}")
print(f"Maximum cases per subject: {max_cases_per_subject}")
print("Future train/validation split: GROUP BY subjectid")

# =============================================================================
# 9. CLINICAL CANDIDATE INVENTORY
# =============================================================================
clinical_candidates = [
    "age",
    "sex",
    "height",
    "weight",
    "bmi",
    "asa",
    "emop",
    "department",
    "ane_type",
    "optype",
    "dx",
    "opname",
]

inventory_rows = []

for col in clinical_candidates:
    present = col in df.columns

    if present:
        s = df[col]
        inventory_rows.append({
            "candidate": col,
            "present": True,
            "missing_n": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100.0),
            "unique_n": int(s.nunique(dropna=True)),
            "dtype": str(s.dtype),
        })
    else:
        inventory_rows.append({
            "candidate": col,
            "present": False,
            "missing_n": np.nan,
            "missing_pct": np.nan,
            "unique_n": np.nan,
            "dtype": "",
        })

clinical_inventory_df = pd.DataFrame(inventory_rows)
clinical_inventory_df.to_csv(
    OUT_CLINICAL_INVENTORY,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 100)
print("CLINICAL CANDIDATE INVENTORY")
print("=" * 100)
print(clinical_inventory_df.to_string(index=False))

# =============================================================================
# 10. PRE-OUTCOME SPECIFICATION
# =============================================================================
specification = {
    "study_stage": "STEP 07B - pre-outcome feature-block audit",
    "outcomes_inspected": False,
    "analysis_window": "first 30 minutes after operation start",
    "conventional_features_locked_n": len(CONVENTIONAL_FEATURES),
    "conventional_features": CONVENTIONAL_FEATURES,
    "coupling_features_locked_n": len(COUPLING_FEATURES),
    "coupling_features": COUPLING_FEATURES,
    "feature_selection_rule": (
        "No outcome-driven feature selection. No feature is automatically removed "
        "in STEP 07B. Any redundancy handling must be decided from signal-processing "
        "or statistical meaning before outcomes are merged."
    ),
    "planned_feature_blocks": {
        "Model_B_signal_baseline": CONVENTIONAL_FEATURES,
        "Model_C_signal_plus_coupling": CONVENTIONAL_FEATURES + COUPLING_FEATURES,
        "Model_A_clinical": (
            "To be frozen from available non-outcome clinical covariates "
            "before outcome modeling."
        ),
    },
    "cross_validation": {
        "splitter": "StratifiedGroupKFold",
        "n_splits": CV_FOLDS,
        "shuffle": True,
        "random_state": CV_RANDOM_STATE,
        "group_column": CV_GROUP_COLUMN,
        "rule": "All cases from the same subjectid must remain in the same fold.",
    },
    "preprocessing": {
        "fit_inside_each_training_fold": True,
        "numeric_missing": "median imputation fitted on training fold only",
        "logistic_regression_scaling": "StandardScaler fitted on training fold only",
        "categorical_if_used": "one-hot encoding fitted on training fold only",
        "tree_models_scaling": "not required",
    },
    "primary_incremental_comparison": (
        "Compare conventional signal features versus conventional + coupling features "
        "using identical subject-grouped folds and preprocessing."
    ),
    "correlation_thresholds": {
        "review": REVIEW_RHO,
        "strong": STRONG_RHO,
        "very_strong": VERY_STRONG_RHO,
    },
}

with open(OUT_SPEC_JSON, "w", encoding="utf-8") as f:
    json.dump(specification, f, ensure_ascii=False, indent=2)

spec_txt = f"""
VitalDB Multimodal Physiological Coupling Study
STEP 07B - PRE-OUTCOME ANALYSIS SPECIFICATION
============================================================

OUTCOMES INSPECTED
------------------
NO.

LOCKED SIGNAL FEATURE BLOCKS
----------------------------
Conventional vital features: {len(CONVENTIONAL_FEATURES)}
Coupling features: {len(COUPLING_FEATURES)}
Total signal features for Model C: {len(ALL_SIGNAL_FEATURES)}

PRIMARY SIGNAL COMPARISON
-------------------------
Model B: all 20 locked conventional vital features.
Model C: Model B + all 12 locked coupling features.

No outcome-driven feature selection is allowed.

REDUNDANCY REVIEW THRESHOLDS
----------------------------
Review:       |rho| >= {REVIEW_RHO:.2f}
Strong:       |rho| >= {STRONG_RHO:.2f}
Very strong:  |rho| >= {VERY_STRONG_RHO:.2f}

No feature is automatically removed in STEP 07B.
Any redundancy decision must be made BEFORE outcomes are merged.

FUTURE CROSS-VALIDATION
-----------------------
Splitter: StratifiedGroupKFold
Folds: {CV_FOLDS}
Shuffle: True
Random state: {CV_RANDOM_STATE}
Group: subjectid

All repeated cases from one subject must stay in the same fold.

PREPROCESSING
-------------
All preprocessing must be fitted inside each training fold.
Numeric missing values: training-fold median imputation.
Logistic regression: training-fold StandardScaler.
Categorical clinical variables (if used): training-fold one-hot encoding.
Tree-based models: no scaling required.

IMPORTANT
---------
The final clinical feature block (Model A) will be frozen from available,
non-outcome clinical covariates BEFORE postoperative outcomes are merged.
"""

OUT_SPEC_TXT.write_text(spec_txt.strip() + "\n", encoding="utf-8")

# =============================================================================
# 11. SUMMARY / DECISION
# =============================================================================
missing_feature_count = int((audit_df["missing_n"] > 0).sum())
inf_feature_count = int((audit_df["inf_n"] > 0).sum())
near_zero_count = int(audit_df["near_zero_variance"].sum())

summary = pd.DataFrame([{
    "input_cases": len(df),
    "unique_subjects": df["subjectid"].nunique(),
    "conventional_features": len(CONVENTIONAL_FEATURES),
    "coupling_features": len(COUPLING_FEATURES),
    "conventional_features_with_missing": missing_feature_count,
    "conventional_features_with_inf": inf_feature_count,
    "conventional_near_zero_variance": near_zero_count,
    "conventional_max_abs_rho": conv_max,
    "conventional_pairs_ge_070": conv_n70,
    "conventional_pairs_ge_085": conv_n85,
    "conventional_pairs_ge_090": conv_n90,
    "cross_block_max_abs_rho": cross_max,
    "cross_pairs_ge_070": cross_n70,
    "cross_pairs_ge_085": cross_n85,
    "cross_pairs_ge_090": cross_n90,
    "repeated_case_subjects": repeated_subjects,
    "max_cases_per_subject": max_cases_per_subject,
    "outcomes_used": False,
}])

summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("STEP 07B AUDIT SUMMARY")
print("=" * 100)

print(f"\nInput cases: {len(df):,}")
print(f"Unique subjects: {df['subjectid'].nunique():,}")
print(f"Conventional features with missing values: {missing_feature_count}")
print(f"Conventional features with inf values: {inf_feature_count}")
print(f"Conventional near-zero variance features: {near_zero_count}")

print(f"\nConventional maximum |rho|: {conv_max:.4f}")
print(f"Conventional |rho| >= 0.70 pairs: {conv_n70}")
print(f"Conventional |rho| >= 0.85 pairs: {conv_n85}")
print(f"Conventional |rho| >= 0.90 pairs: {conv_n90}")

print(f"\nConventional-vs-coupling maximum |rho|: {cross_max:.4f}")
print(f"Cross-block |rho| >= 0.70 pairs: {cross_n70}")
print(f"Cross-block |rho| >= 0.85 pairs: {cross_n85}")
print(f"Cross-block |rho| >= 0.90 pairs: {cross_n90}")

print(f"\nRepeated-case subjects: {repeated_subjects}")
print(f"Maximum cases per subject: {max_cases_per_subject}")
print("Outcome variables inspected: NO")

print("\n" + "=" * 100)
print("STEP 07B DECISION")
print("=" * 100)

if (
    missing_feature_count == 0
    and inf_feature_count == 0
    and near_zero_count == 0
    and cross_n85 == 0
):
    if conv_n85 == 0:
        decision = "VERY GOOD"
        reason = (
            "Complete conventional feature block with no strong within-block or "
            "cross-block redundancy."
        )
    else:
        decision = "GOOD - REVIEW CONVENTIONAL REDUNDANCY"
        reason = (
            "Signal blocks are complete and coupling adds non-duplicative structure, "
            "but strong redundancy exists among conventional summary features. "
            "Review using statistical/signal meaning before outcomes."
        )
else:
    decision = "REVIEW"
    reason = (
        "Missing/non-finite/near-zero features or strong conventional-vs-coupling "
        "redundancy require review before outcome modeling."
    )

print(f"\nDecision: {decision}")
print(f"Reason: {reason}")

print("\nGenerated files:")
for p in [
    OUT_CONV_AUDIT,
    OUT_CONV_CORR,
    OUT_CONV_PAIRS,
    OUT_CONV_HIGH,
    OUT_CROSS_CORR,
    OUT_CROSS_PAIRS,
    OUT_CROSS_HIGH,
    OUT_ALL_CORR,
    OUT_CLINICAL_INVENTORY,
    OUT_SUMMARY,
    OUT_SPEC_JSON,
    OUT_SPEC_TXT,
    FIG_CONV,
    FIG_ALL,
]:
    print(p)

print(
    """
NEXT:
Send the console sections below for review BEFORE merging outcomes:

1) CONVENTIONAL FEATURE PAIRWISE SPEARMAN CORRELATION
   - especially all pairs with |rho| >= 0.70

2) CONVENTIONAL VS COUPLING SPEARMAN CORRELATION
   - especially all pairs with |rho| >= 0.70

3) CLINICAL CANDIDATE INVENTORY

4) STEP 07B AUDIT SUMMARY / DECISION
"""
)