# ============================================================
# VitalDB Multimodal Physiological Coupling Study
# STEP 06
#
# Outcome-Independent Audit of Locked Coupling Features
#
# PURPOSE
# ------------------------------------------------------------
# STEP 05에서 QC PASS한 cohort를 대상으로, postoperative
# outcome을 전혀 사용하지 않고 12개 locked coupling feature의
# 분석 적합성을 점검한다.
#
# 1. 결측 / inf 확인
# 2. 분포 / 극단값 확인
# 3. Spearman 상관성 / 중복성 확인
# 4. 반복 case / unique subject 구조 확인
# 5. 최종 분석 cohort 고정
#
# IMPORTANT
# ------------------------------------------------------------
# - 이 단계에서는 ICU, mortality 등 outcome을 사용하지 않는다.
# - 상관성이 높다고 자동으로 feature를 삭제하지 않는다.
# - |rho| >= 0.70 : review flag
# - |rho| >= 0.85 : strong redundancy flag
# - strong redundancy가 있으면 outcome을 보기 전에 feature
#   유지/제외 여부를 결정한다.
# - 반복 수술 case는 제거하지 않고 향후 ML에서 subjectid를
#   group으로 사용해 data leakage를 방지한다.
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data_processed"
RESULT_DIR = PROJECT_DIR / "results"
FIGURE_DIR = PROJECT_DIR / "figures" / "06_feature_audit"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. INPUT / OUTPUT FILES
# ============================================================

INPUT_PATH = (
    PROCESSED_DIR
    / "05_locked_coupling_features_qc_pass.csv"
)

FEATURE_AUDIT_PATH = (
    RESULT_DIR
    / "06_locked_feature_audit.csv"
)

CORR_MATRIX_PATH = (
    RESULT_DIR
    / "06_locked_feature_spearman_matrix.csv"
)

CORR_PAIRS_PATH = (
    RESULT_DIR
    / "06_locked_feature_correlation_pairs.csv"
)

HIGH_CORR_PATH = (
    RESULT_DIR
    / "06_high_correlation_pairs.csv"
)

SUBJECT_STRUCTURE_PATH = (
    RESULT_DIR
    / "06_subject_case_structure.csv"
)

OUTLIER_FLAG_PATH = (
    RESULT_DIR
    / "06_feature_extreme_value_flags.csv"
)

FINAL_COHORT_PATH = (
    PROCESSED_DIR
    / "06_final_analysis_cohort_locked.csv"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "06_feature_audit_summary.csv"
)

HEATMAP_PATH = (
    FIGURE_DIR
    / "06_spearman_correlation_heatmap.png"
)


# ============================================================
# 3. LOCKED PRIMARY FEATURE SET
# ============================================================

LOCKED_FEATURES = [
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


# ============================================================
# 4. PREDECLARED AUDIT THRESHOLDS
# ============================================================

REVIEW_CORR_THRESHOLD = 0.70
STRONG_CORR_THRESHOLD = 0.85
VERY_STRONG_CORR_THRESHOLD = 0.90

NEAR_ZERO_STD_THRESHOLD = 1e-8

# 극단값은 제거하지 않고 flag만 생성
EXTREME_LOWER_Q = 0.01
EXTREME_UPPER_Q = 0.99


# ============================================================
# 5. START
# ============================================================

print("=" * 92)
print("VitalDB Multimodal Physiological Coupling Study")
print("STEP 06 - Outcome-Independent Locked Feature Audit")
print("=" * 92)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"STEP 05 QC PASS 파일이 없습니다:\n{INPUT_PATH}"
    )


df = pd.read_csv(INPUT_PATH)

df.columns = (
    df.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)


# ============================================================
# 6. REQUIRED COLUMN CHECK
# ============================================================

required_columns = ["caseid", "subjectid"] + LOCKED_FEATURES

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "필수 column이 없습니다: "
        + ", ".join(missing_columns)
    )


# ============================================================
# 7. BASIC ID CLEANING
# ============================================================

df["caseid"] = pd.to_numeric(
    df["caseid"],
    errors="coerce"
)

df["subjectid"] = pd.to_numeric(
    df["subjectid"],
    errors="coerce"
)

for feature in LOCKED_FEATURES:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


print(f"\nInput QC-PASS cases: {len(df):,}")
print(
    "Unique subjects: "
    f"{df['subjectid'].nunique(dropna=True):,}"
)
print(f"Locked primary features: {len(LOCKED_FEATURES)}")
print("Outcome variables: NOT USED")


# ============================================================
# 8. DUPLICATE CASE CHECK
# ============================================================

duplicate_case_mask = df["caseid"].duplicated(keep=False)

n_duplicate_rows = int(duplicate_case_mask.sum())
n_duplicate_caseids = int(
    df.loc[duplicate_case_mask, "caseid"].nunique()
)

print("\n" + "=" * 92)
print("CASE ID INTEGRITY")
print("=" * 92)
print(f"Duplicate case rows: {n_duplicate_rows:,}")
print(f"Duplicate caseids: {n_duplicate_caseids:,}")

if n_duplicate_rows > 0:
    duplicate_path = (
        RESULT_DIR
        / "06_duplicate_caseids.csv"
    )

    df.loc[duplicate_case_mask].to_csv(
        duplicate_path,
        index=False,
        encoding="utf-8-sig"
    )

    raise ValueError(
        "Duplicate caseid가 발견되었습니다. "
        f"{duplicate_path}를 확인하세요."
    )


# ============================================================
# 9. FEATURE AUDIT
# ============================================================

print("\n" + "=" * 92)
print("LOCKED FEATURE AUDIT")
print("=" * 92)

feature_audit_rows = []
extreme_flag_rows = []

for feature in LOCKED_FEATURES:
    values = df[feature].copy()

    n_total = len(values)
    n_missing = int(values.isna().sum())
    n_inf = int(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).sum())

    finite_mask = (
        values.notna()
        & np.isfinite(values)
    )

    finite_values = values.loc[finite_mask].astype(float)

    if len(finite_values) == 0:
        stats = {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "q1": np.nan,
            "median": np.nan,
            "q3": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "iqr": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "n_unique": 0,
        }
        lower = np.nan
        upper = np.nan
        n_extreme = 0
    else:
        q1 = float(finite_values.quantile(0.25))
        q3 = float(finite_values.quantile(0.75))
        lower = float(finite_values.quantile(EXTREME_LOWER_Q))
        upper = float(finite_values.quantile(EXTREME_UPPER_Q))

        stats = {
            "mean": float(finite_values.mean()),
            "std": float(finite_values.std(ddof=1)),
            "min": float(finite_values.min()),
            "p01": lower,
            "p05": float(finite_values.quantile(0.05)),
            "q1": q1,
            "median": float(finite_values.median()),
            "q3": q3,
            "p95": float(finite_values.quantile(0.95)),
            "p99": upper,
            "max": float(finite_values.max()),
            "iqr": float(q3 - q1),
            "skew": float(finite_values.skew()),
            "kurtosis": float(finite_values.kurtosis()),
            "n_unique": int(finite_values.nunique()),
        }

        extreme_mask = (
            finite_mask
            & (
                (df[feature] < lower)
                | (df[feature] > upper)
            )
        )

        n_extreme = int(extreme_mask.sum())

        for idx in df.index[extreme_mask]:
            extreme_flag_rows.append({
                "caseid": df.at[idx, "caseid"],
                "subjectid": df.at[idx, "subjectid"],
                "feature": feature,
                "value": df.at[idx, feature],
                "p01": lower,
                "p99": upper,
                "extreme_direction": (
                    "low"
                    if df.at[idx, feature] < lower
                    else "high"
                ),
            })

    near_zero_variance = int(
        (
            pd.notna(stats["std"])
            and stats["std"] < NEAR_ZERO_STD_THRESHOLD
        )
        or stats["n_unique"] <= 1
    )

    feature_audit_rows.append({
        "feature": feature,
        "n_total": n_total,
        "n_finite": int(finite_mask.sum()),
        "n_missing": n_missing,
        "missing_pct": n_missing / n_total * 100,
        "n_inf": n_inf,
        "inf_pct": n_inf / n_total * 100,
        "n_unique": stats["n_unique"],
        "mean": stats["mean"],
        "std": stats["std"],
        "min": stats["min"],
        "p01": stats["p01"],
        "p05": stats["p05"],
        "q1": stats["q1"],
        "median": stats["median"],
        "q3": stats["q3"],
        "p95": stats["p95"],
        "p99": stats["p99"],
        "max": stats["max"],
        "iqr": stats["iqr"],
        "skew": stats["skew"],
        "kurtosis": stats["kurtosis"],
        "n_extreme_outside_p01_p99": n_extreme,
        "near_zero_variance": near_zero_variance,
    })

    print(
        f"{feature:<38} "
        f"missing={n_missing:>4} | "
        f"inf={n_inf:>3} | "
        f"median={stats['median']:>8.4f} | "
        f"IQR={stats['iqr']:>8.4f} | "
        f"unique={stats['n_unique']:>4}"
    )

feature_audit_df = pd.DataFrame(feature_audit_rows)
extreme_flag_df = pd.DataFrame(extreme_flag_rows)

feature_audit_df.to_csv(
    FEATURE_AUDIT_PATH,
    index=False,
    encoding="utf-8-sig"
)

extreme_flag_df.to_csv(
    OUTLIER_FLAG_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. COMPLETE / FINITE CASE MASK
# ============================================================

feature_matrix = df[LOCKED_FEATURES].copy()

finite_matrix = np.isfinite(
    feature_matrix.to_numpy(dtype=float)
)

complete_finite_mask = (
    feature_matrix.notna().all(axis=1)
    & finite_matrix.all(axis=1)
)

valid_subject_mask = (
    df["subjectid"].notna()
    & np.isfinite(df["subjectid"])
)

valid_case_mask = (
    df["caseid"].notna()
    & np.isfinite(df["caseid"])
)

final_mask = (
    complete_finite_mask
    & valid_subject_mask
    & valid_case_mask
)


# ============================================================
# 11. SPEARMAN CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 92)
print("PAIRWISE SPEARMAN CORRELATION")
print("=" * 92)

corr_df = (
    df.loc[final_mask, LOCKED_FEATURES]
    .corr(method="spearman")
)

corr_df.to_csv(
    CORR_MATRIX_PATH,
    encoding="utf-8-sig"
)

corr_pair_rows = []

for i in range(len(LOCKED_FEATURES)):
    for j in range(i + 1, len(LOCKED_FEATURES)):
        feature_1 = LOCKED_FEATURES[i]
        feature_2 = LOCKED_FEATURES[j]

        rho = float(
            corr_df.loc[feature_1, feature_2]
        )

        abs_rho = abs(rho)

        if abs_rho >= VERY_STRONG_CORR_THRESHOLD:
            level = "VERY_STRONG"
        elif abs_rho >= STRONG_CORR_THRESHOLD:
            level = "STRONG"
        elif abs_rho >= REVIEW_CORR_THRESHOLD:
            level = "REVIEW"
        else:
            level = "OK"

        corr_pair_rows.append({
            "feature_1": feature_1,
            "feature_2": feature_2,
            "spearman_rho": rho,
            "abs_rho": abs_rho,
            "flag_level": level,
        })

corr_pairs_df = pd.DataFrame(corr_pair_rows)

corr_pairs_df = (
    corr_pairs_df
    .sort_values(
        "abs_rho",
        ascending=False
    )
    .reset_index(drop=True)
)

corr_pairs_df.to_csv(
    CORR_PAIRS_PATH,
    index=False,
    encoding="utf-8-sig"
)

high_corr_df = corr_pairs_df.loc[
    corr_pairs_df["abs_rho"] >= REVIEW_CORR_THRESHOLD
].copy()

high_corr_df.to_csv(
    HIGH_CORR_PATH,
    index=False,
    encoding="utf-8-sig"
)

if len(high_corr_df) == 0:
    print("\nNo feature pairs with |rho| >= 0.70")
else:
    print("\nPairs with |rho| >= 0.70:")
    print(
        high_corr_df.to_string(
            index=False,
            formatters={
                "spearman_rho": "{:.4f}".format,
                "abs_rho": "{:.4f}".format,
            }
        )
    )

max_abs_rho = (
    float(corr_pairs_df["abs_rho"].max())
    if len(corr_pairs_df) > 0
    else np.nan
)

n_review_pairs = int(
    (corr_pairs_df["abs_rho"] >= REVIEW_CORR_THRESHOLD).sum()
)

n_strong_pairs = int(
    (corr_pairs_df["abs_rho"] >= STRONG_CORR_THRESHOLD).sum()
)

n_very_strong_pairs = int(
    (corr_pairs_df["abs_rho"] >= VERY_STRONG_CORR_THRESHOLD).sum()
)

print(f"\nMaximum |rho| = {max_abs_rho:.4f}")
print(f"|rho| >= 0.70 pairs = {n_review_pairs}")
print(f"|rho| >= 0.85 pairs = {n_strong_pairs}")
print(f"|rho| >= 0.90 pairs = {n_very_strong_pairs}")


# ============================================================
# 12. CORRELATION HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(12, 10))

image = ax.imshow(
    corr_df.to_numpy(),
    vmin=-1,
    vmax=1,
    aspect="auto"
)

ax.set_xticks(
    np.arange(len(LOCKED_FEATURES))
)
ax.set_yticks(
    np.arange(len(LOCKED_FEATURES))
)

short_labels = [
    "Lag median",
    "Lag IQR",
    "Xcorr median",
    "Xcorr IQR",
    "Xcorr lag",
    "Amplitude rho",
    "HR-MAP med",
    "HR-MAP IQR",
    "MAP-ETCO2 med",
    "MAP-ETCO2 IQR",
    "HR-ETCO2 med",
    "HR-ETCO2 IQR",
]

ax.set_xticklabels(
    short_labels,
    rotation=60,
    ha="right"
)
ax.set_yticklabels(short_labels)

for i in range(len(LOCKED_FEATURES)):
    for j in range(len(LOCKED_FEATURES)):
        value = corr_df.iloc[i, j]
        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=7
        )

ax.set_title(
    "Spearman Correlation of Locked Coupling Features"
)

fig.colorbar(
    image,
    ax=ax,
    label="Spearman rho"
)

plt.tight_layout()
plt.savefig(
    HEATMAP_PATH,
    dpi=220,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 13. INDIVIDUAL FEATURE DISTRIBUTION FIGURES
# ============================================================

for feature in LOCKED_FEATURES:
    values = (
        df.loc[final_mask, feature]
        .dropna()
        .astype(float)
    )

    if len(values) == 0:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=50)
    ax.axvline(
        values.median(),
        linestyle="--",
        linewidth=1.5,
        label=f"Median = {values.median():.4f}"
    )
    ax.set_title(feature)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()

    safe_name = feature.replace("/", "_")

    plt.savefig(
        FIGURE_DIR / f"distribution_{safe_name}.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 14. SUBJECT / REPEATED-CASE STRUCTURE
# ============================================================

print("\n" + "=" * 92)
print("SUBJECT / REPEATED-CASE STRUCTURE")
print("=" * 92)

subject_case_counts = (
    df.loc[final_mask]
    .groupby("subjectid")["caseid"]
    .nunique()
    .sort_values(ascending=False)
)

subject_structure_df = (
    subject_case_counts
    .rename("n_cases")
    .reset_index()
)

subject_structure_df["repeated_subject"] = (
    subject_structure_df["n_cases"] > 1
).astype(int)

subject_structure_df.to_csv(
    SUBJECT_STRUCTURE_PATH,
    index=False,
    encoding="utf-8-sig"
)

n_final_cases = int(final_mask.sum())
n_final_subjects = int(
    df.loc[final_mask, "subjectid"].nunique()
)
n_repeated_subjects = int(
    (subject_case_counts > 1).sum()
)
n_single_case_subjects = int(
    (subject_case_counts == 1).sum()
)
max_cases_per_subject = int(
    subject_case_counts.max()
) if len(subject_case_counts) > 0 else 0

repeat_case_rows = int(
    n_final_cases - n_final_subjects
)

print(f"Final finite cases: {n_final_cases:,}")
print(f"Final unique subjects: {n_final_subjects:,}")
print(f"Single-case subjects: {n_single_case_subjects:,}")
print(f"Repeated-case subjects: {n_repeated_subjects:,}")
print(f"Extra repeated case rows: {repeat_case_rows:,}")
print(f"Maximum cases per subject: {max_cases_per_subject}")
print("Future ML split requirement: GROUP BY subjectid")


# ============================================================
# 15. FREEZE FINAL ANALYSIS COHORT
# ============================================================

final_df = (
    df.loc[final_mask]
    .copy()
    .sort_values(
        ["subjectid", "caseid"]
    )
    .reset_index(drop=True)
)

final_df["analysis_case_order"] = np.arange(
    1,
    len(final_df) + 1
)

final_df["subject_group_required"] = 1
final_df["locked_feature_set_version"] = "STEP06_v1_12features"

final_df.to_csv(
    FINAL_COHORT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. FINAL AUDIT DECISION
# ============================================================

n_features_missing = int(
    (feature_audit_df["n_missing"] > 0).sum()
)

n_features_inf = int(
    (feature_audit_df["n_inf"] > 0).sum()
)

n_near_zero_features = int(
    feature_audit_df["near_zero_variance"].sum()
)

removed_for_nonfinite_or_id = int(
    len(df) - n_final_cases
)

print("\n" + "=" * 92)
print("STEP 06 AUDIT SUMMARY")
print("=" * 92)

print(f"\nInput QC-PASS cases: {len(df):,}")
print(f"Final frozen cases: {n_final_cases:,}")
print(f"Final frozen subjects: {n_final_subjects:,}")
print(
    "Removed for missing/non-finite feature or ID: "
    f"{removed_for_nonfinite_or_id:,}"
)
print(f"Features with missing values: {n_features_missing}")
print(f"Features with inf values: {n_features_inf}")
print(f"Near-zero variance features: {n_near_zero_features}")
print(f"Maximum absolute Spearman rho: {max_abs_rho:.4f}")
print(f"Correlation review pairs |rho|>=0.70: {n_review_pairs}")
print(f"Strong redundancy pairs |rho|>=0.85: {n_strong_pairs}")
print(f"Very strong pairs |rho|>=0.90: {n_very_strong_pairs}")


if (
    n_final_cases >= 2500
    and n_features_missing == 0
    and n_features_inf == 0
    and n_near_zero_features == 0
    and n_strong_pairs == 0
):
    decision = "VERY GOOD"
    reason = (
        "Large complete cohort with no strong feature redundancy. "
        "Keep all 12 locked primary features."
    )

elif (
    n_final_cases >= 2000
    and n_features_inf == 0
    and n_near_zero_features == 0
    and n_very_strong_pairs == 0
):
    decision = "GOOD / REVIEW CORRELATION"
    reason = (
        "Cohort is sufficient. Review flagged correlated feature pairs "
        "before outcome modeling."
    )

else:
    decision = "REVIEW REQUIRED"
    reason = (
        "Feature integrity, redundancy, or cohort size requires review "
        "before outcomes are examined."
    )

print(f"\nDecision: {decision}")
print(f"Reason: {reason}")


# ============================================================
# 17. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame([
    {
        "input_qc_pass_cases": len(df),
        "input_unique_subjects": int(
            df["subjectid"].nunique(dropna=True)
        ),
        "final_frozen_cases": n_final_cases,
        "final_frozen_subjects": n_final_subjects,
        "removed_nonfinite_or_missing_id": removed_for_nonfinite_or_id,
        "n_locked_features": len(LOCKED_FEATURES),
        "features_with_missing": n_features_missing,
        "features_with_inf": n_features_inf,
        "near_zero_variance_features": n_near_zero_features,
        "max_abs_spearman_rho": max_abs_rho,
        "corr_pairs_abs_rho_ge_070": n_review_pairs,
        "corr_pairs_abs_rho_ge_085": n_strong_pairs,
        "corr_pairs_abs_rho_ge_090": n_very_strong_pairs,
        "single_case_subjects": n_single_case_subjects,
        "repeated_case_subjects": n_repeated_subjects,
        "extra_repeated_case_rows": repeat_case_rows,
        "max_cases_per_subject": max_cases_per_subject,
        "decision": decision,
        "reason": reason,
    }
])

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. FINISH
# ============================================================

print("\n" + "=" * 92)
print("STEP 06 COMPLETE")
print("=" * 92)

print("\nGenerated files:")
print(FEATURE_AUDIT_PATH)
print(CORR_MATRIX_PATH)
print(CORR_PAIRS_PATH)
print(HIGH_CORR_PATH)
print(SUBJECT_STRUCTURE_PATH)
print(OUTLIER_FLAG_PATH)
print(FINAL_COHORT_PATH)
print(SUMMARY_PATH)
print(HEATMAP_PATH)

print("\nIMPORTANT:")
print(
    "Do NOT merge or inspect postoperative outcomes until this audit "
    "has been reviewed."
)
print(
    "If |rho| >= 0.85 exists, decide redundancy handling using only "
    "signal-processing meaning before outcome modeling."
)
print(
    "Repeated cases are retained; all future CV/train-test splitting "
    "must be grouped by subjectid."
)

print("\nNEXT STEP:")
print(
    "STEP 07 = after reviewing STEP 06, generate conventional vital "
    "features from the same first 30-min window, then freeze model "
    "feature sets before outcomes are examined."
)