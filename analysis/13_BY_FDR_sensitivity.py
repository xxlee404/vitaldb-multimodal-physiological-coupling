from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# ============================================================
# BSPC REVISION
# STEP 13 - BY-FDR Sensitivity Analysis
# NaN-safe version
# ============================================================

print("=" * 92)
print("BSPC REVISION - STEP 13")
print("BY-FDR SENSITIVITY ANALYSIS")
print("=" * 92)

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_DIR / "results"

INPUT_PATH = RESULT_DIR / "12B_adjusted_context_omnibus_tests.csv"

OUTPUT_PATH = RESULT_DIR / "13_revision_BY_FDR_sensitivity.csv"
SUMMARY_PATH = RESULT_DIR / "13_revision_BY_FDR_summary.csv"
MAJOR_PATH = RESULT_DIR / "13_revision_BY_FDR_major6.csv"
MISSING_PATH = RESULT_DIR / "13_revision_missing_pvalue_tests.csv"

print("\nPROJECT_DIR:")
print(PROJECT_DIR)

print("\nINPUT:")
print(INPUT_PATH)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"입력 파일을 찾을 수 없습니다:\n{INPUT_PATH}"
    )

# ============================================================
# 2. LOAD STEP12B RESULTS
# ============================================================

df = pd.read_csv(INPUT_PATH)
df.columns = (
    df.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

required_columns = [
    "coupling_feature",
    "domain",
    "context_variable",
    "p_value_cluster_robust",
    "delta_adjusted_R2",
    "q_value_FDR",
    "FDR_lt_0_05",
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"필수 column 없음: {col}")

print(f"\nTotal rows loaded: {len(df)}")

if len(df) != 144:
    print(
        f"WARNING: Expected 144 rows, but found {len(df)}."
    )

# ============================================================
# 3. CHECK RAW P-VALUES
# ============================================================

df["p_value_cluster_robust"] = pd.to_numeric(
    df["p_value_cluster_robust"],
    errors="coerce"
)

valid_mask = df["p_value_cluster_robust"].notna()
missing_mask = ~valid_mask

n_valid = int(valid_mask.sum())
n_missing = int(missing_mask.sum())

print("\n" + "=" * 92)
print("RAW P-VALUE CHECK")
print("=" * 92)

print(f"Total planned tests : {len(df)}")
print(f"Valid p-values      : {n_valid}")
print(f"Missing p-values    : {n_missing}")

# ============================================================
# 4. PRINT / SAVE MISSING TESTS
# ============================================================

missing_df = df.loc[
    missing_mask,
    [
        "coupling_feature",
        "domain",
        "context_variable",
        "context_type",
        "n_cases",
        "n_subjects",
        "wald_chi2",
        "wald_df",
        "p_value_cluster_robust",
        "delta_adjusted_R2",
        "q_value_FDR",
        "FDR_lt_0_05",
    ]
].copy()

print("\n" + "=" * 92)
print("TESTS WITH MISSING RAW P-VALUES")
print("=" * 92)

if len(missing_df) > 0:
    print(missing_df.to_string(index=False))

    missing_df.to_csv(
        MISSING_PATH,
        index=False,
        encoding="utf-8-sig"
    )
else:
    print("No missing p-values.")

# ============================================================
# 5. PREPARE OUTPUT COLUMNS
# ============================================================

df["q_BH_recalculated"] = np.nan
df["BH_q_lt_0_05_recalculated"] = False

df["q_BY"] = np.nan
df["BY_q_lt_0_05"] = False
df["BY_q_lt_0_01"] = False

# ============================================================
# 6. BH-FDR ON VALID P-VALUES ONLY
# ============================================================

valid_pvals = df.loc[
    valid_mask,
    "p_value_cluster_robust"
].to_numpy()

reject_bh, q_bh, _, _ = multipletests(
    valid_pvals,
    alpha=0.05,
    method="fdr_bh"
)

df.loc[
    valid_mask,
    "q_BH_recalculated"
] = q_bh

df.loc[
    valid_mask,
    "BH_q_lt_0_05_recalculated"
] = reject_bh

# ============================================================
# 7. COMPARE WITH ORIGINAL BH RESULTS
# ============================================================

reported_q = pd.to_numeric(
    df["q_value_FDR"],
    errors="coerce"
)

comparison_mask = (
    valid_mask
    & reported_q.notna()
)

if comparison_mask.sum() > 0:
    max_q_diff = np.nanmax(
        np.abs(
            reported_q.loc[comparison_mask]
            - df.loc[
                comparison_mask,
                "q_BH_recalculated"
            ]
        )
    )
else:
    max_q_diff = np.nan

reported_bh_n = int(
    df["FDR_lt_0_05"]
    .fillna(False)
    .astype(bool)
    .sum()
)

recalculated_bh_n = int(
    df["BH_q_lt_0_05_recalculated"].sum()
)

print("\n" + "=" * 92)
print("BH-FDR REPRODUCIBILITY CHECK")
print("=" * 92)

print(f"Reported BH significant : {reported_bh_n}")
print(f"Recalculated BH sig.    : {recalculated_bh_n}")
print(f"Valid tests used        : {n_valid}")

if np.isfinite(max_q_diff):
    print(f"Maximum q-value diff    : {max_q_diff:.3e}")

if (
    reported_bh_n == recalculated_bh_n
    and np.isfinite(max_q_diff)
    and max_q_diff < 1e-10
):
    print(
        "\nCHECK: Original BH-FDR procedure reproduced exactly."
    )
else:
    print(
        "\nATTENTION: Original BH values were not reproduced exactly."
    )

# ============================================================
# 8. BY-FDR ON SAME VALID TESTS
# ============================================================

reject_by, q_by, _, _ = multipletests(
    valid_pvals,
    alpha=0.05,
    method="fdr_by"
)

df.loc[
    valid_mask,
    "q_BY"
] = q_by

df.loc[
    valid_mask,
    "BY_q_lt_0_05"
] = reject_by

df.loc[
    valid_mask,
    "BY_q_lt_0_01"
] = q_by < 0.01

by_n = int(
    df["BY_q_lt_0_05"].sum()
)

by_01_n = int(
    df["BY_q_lt_0_01"].sum()
)

# ============================================================
# 9. RETENTION
# ============================================================

df["significant_BH_and_BY"] = (
    df["BH_q_lt_0_05_recalculated"]
    & df["BY_q_lt_0_05"]
)

retained_n = int(
    df["significant_BH_and_BY"].sum()
)

retention_rate = (
    retained_n / recalculated_bh_n * 100
    if recalculated_bh_n > 0
    else np.nan
)

print("\n" + "=" * 92)
print("BY-FDR SENSITIVITY RESULT")
print("=" * 92)

print(f"Planned tests               : {len(df)}")
print(f"Evaluable tests             : {n_valid}")
print(f"BH-FDR q<0.05              : {recalculated_bh_n}")
print(f"BY-FDR q<0.05              : {by_n}")
print(f"BY-FDR q<0.01              : {by_01_n}")
print(
    f"BH associations retained BY: "
    f"{retained_n}/{recalculated_bh_n} "
    f"({retention_rate:.1f}%)"
)

# ============================================================
# 10. SIX MAJOR ASSOCIATIONS
# ============================================================

major6 = df.loc[
    pd.to_numeric(
        df["delta_adjusted_R2"],
        errors="coerce"
    ) >= 0.02
].copy()

major6 = major6.sort_values(
    "delta_adjusted_R2",
    ascending=False
)

print("\n" + "=" * 92)
print("MAJOR ASSOCIATIONS (Δadjusted R² >= 0.02)")
print("=" * 92)

show_cols = [
    "coupling_feature",
    "domain",
    "context_variable",
    "delta_adjusted_R2",
    "p_value_cluster_robust",
    "q_BH_recalculated",
    "q_BY",
    "BY_q_lt_0_05",
]

print(
    major6[show_cols]
    .to_string(index=False)
)

major_n = len(major6)
major_by_n = int(
    major6["BY_q_lt_0_05"].sum()
)

print(
    f"\nMajor associations retained under BY-FDR: "
    f"{major_by_n}/{major_n}"
)

# ============================================================
# 11. DOMAIN SUMMARY
# ============================================================

domain_summary = (
    df.groupby("domain")
    .agg(
        planned_tests=("coupling_feature", "size"),
        valid_tests=(
            "p_value_cluster_robust",
            lambda x: x.notna().sum()
        ),
        BH_significant=(
            "BH_q_lt_0_05_recalculated",
            "sum"
        ),
        BY_significant=(
            "BY_q_lt_0_05",
            "sum"
        ),
    )
    .reset_index()
)

domain_summary[
    "BY_retention_percent_of_BH"
] = np.where(
    domain_summary["BH_significant"] > 0,
    domain_summary["BY_significant"]
    / domain_summary["BH_significant"]
    * 100,
    np.nan
)

print("\n" + "=" * 92)
print("DOMAIN-SPECIFIC RESULTS")
print("=" * 92)

print(
    domain_summary.to_string(index=False)
)

# ============================================================
# 12. SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

major6.to_csv(
    MAJOR_PATH,
    index=False,
    encoding="utf-8-sig"
)

summary_df = pd.DataFrame([{
    "planned_tests": len(df),
    "valid_pvalue_tests": n_valid,
    "missing_pvalue_tests": n_missing,
    "reported_BH_significant_q05": reported_bh_n,
    "recalculated_BH_significant_q05": recalculated_bh_n,
    "BY_significant_q05": by_n,
    "BY_significant_q01": by_01_n,
    "BH_significant_retained_BY": retained_n,
    "BY_retention_percent_of_BH": retention_rate,
    "major_association_count": major_n,
    "major_associations_retained_BY": major_by_n,
    "max_BH_q_difference": max_q_diff,
}])

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 92)
print("SAVED FILES")
print("=" * 92)

print(OUTPUT_PATH)
print(SUMMARY_PATH)
print(MAJOR_PATH)

if len(missing_df) > 0:
    print(MISSING_PATH)

print("\nSTEP 13 COMPLETE")