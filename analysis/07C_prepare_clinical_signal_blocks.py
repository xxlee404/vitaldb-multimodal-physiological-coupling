# -*- coding: utf-8 -*-
"""
VitalDB Multimodal Physiological Coupling Study
STEP 07C - Clinical Covariate Recovery + Pre-Outcome Signal Block Freeze

Purpose
-------
1) Preserve all 20 conventional features for sensitivity analysis.
2) Freeze a reduced 14-feature conventional PRIMARY baseline using ONLY
   outcome-independent redundancy + signal-processing meaning.
3) Keep all 12 locked coupling features.
4) Recover candidate clinical covariates by caseid WITHOUT loading outcome columns.
5) Audit clinical missingness / category counts.
6) Do NOT inspect postoperative outcomes.

Primary conventional reduction rule (fixed before outcomes)
-----------------------------------------------------------
HR:    mean, std, min, max
MAP:   mean, std, min, max
SpO2:  mean, min
ETCO2: mean, std, min, max

Why:
- range = max - min, so all range variables are deterministically derived and removed.
- SpO2_max shows a strong ceiling effect (almost always 100).
- SpO2_std is strongly redundant with SpO2_mean/min in STEP 07B.
- SpO2_mean + SpO2_min retain overall oxygenation level + desaturation information.

The original 20-feature conventional block is retained as a prespecified
sensitivity-analysis block.
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PATHS
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data_processed"
RAW_DIR = PROJECT_DIR / "data_raw"
RESULT_DIR = PROJECT_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "07_analysis_features_without_outcomes.csv"

OUT_DATA = PROCESSED_DIR / "07C_preoutcome_analysis_dataset.csv"
OUT_CLINICAL_AUDIT = RESULT_DIR / "07C_clinical_covariate_audit.csv"
OUT_SOURCE_AUDIT = RESULT_DIR / "07C_clinical_source_search.csv"
OUT_SPEC_JSON = RESULT_DIR / "07C_frozen_signal_blocks_and_clinical_candidates.json"
OUT_SPEC_TXT = RESULT_DIR / "07C_frozen_signal_blocks_and_clinical_candidates.txt"

# =============================================================================
# 2. LOCKED SIGNAL FEATURES
# =============================================================================
CONVENTIONAL_ALL20 = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
    "map_mean", "map_std", "map_min", "map_max", "map_range",
    "spo2_mean", "spo2_std", "spo2_min", "spo2_max", "spo2_range",
    "etco2_mean", "etco2_std", "etco2_min", "etco2_max", "etco2_range",
]

CONVENTIONAL_PRIMARY14 = [
    "hr_mean", "hr_std", "hr_min", "hr_max",
    "map_mean", "map_std", "map_min", "map_max",
    "spo2_mean", "spo2_min",
    "etco2_mean", "etco2_std", "etco2_min", "etco2_max",
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

# Candidate clinical variables only.
# IMPORTANT: outcome columns are intentionally NOT included here.
CLINICAL_CANDIDATES = [
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
    "approach",
    "position",
    "preop_htn",
    "preop_dm",
]

# Explicitly forbidden outcome / post-exposure variables.
FORBIDDEN_COLUMNS = {
    "icu_days",
    "death_inhosp",
    "death",
    "mortality",
    "adm",
    "dis",
    "los",
    "length_of_stay",
    "postop_los",
    "intraop_ebl",
    "intraop_uo",
    "intraop_rbc",
    "intraop_ffp",
    "intraop_crystalloid",
    "intraop_colloid",
    "intraop_ppf",
    "intraop_mdz",
    "intraop_ftn",
    "intraop_rocu",
    "intraop_vecu",
    "intraop_eph",
    "intraop_phe",
    "intraop_epi",
    "intraop_ca",
}

# =============================================================================
# 3. LOAD CURRENT PRE-OUTCOME FEATURE DATASET
# =============================================================================
print("=" * 100)
print("VitalDB Multimodal Physiological Coupling Study")
print("STEP 07C - Clinical Covariate Recovery + Pre-Outcome Signal Block Freeze")
print("=" * 100)

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"입력 파일이 없습니다:\n{INPUT_PATH}")

base = pd.read_csv(INPUT_PATH)
base.columns = base.columns.str.replace("\ufeff", "", regex=False).str.strip()

for col in ["caseid", "subjectid"]:
    if col not in base.columns:
        raise ValueError(f"필수 컬럼 없음: {col}")

base["caseid"] = pd.to_numeric(base["caseid"], errors="coerce")
base = base.dropna(subset=["caseid"]).copy()
base["caseid"] = base["caseid"].astype(int)

missing_signal = [
    c for c in (CONVENTIONAL_ALL20 + COUPLING12)
    if c not in base.columns
]
if missing_signal:
    raise ValueError(
        "필수 signal feature가 없습니다:\n" + "\n".join(missing_signal)
    )

print(f"\nBase cases: {len(base):,}")
print(f"Unique subjects: {base['subjectid'].nunique():,}")
print("Outcome variables loaded: NO")

# =============================================================================
# 4. REPORT FROZEN SIGNAL BLOCKS
# =============================================================================
print("\n" + "=" * 100)
print("FROZEN SIGNAL FEATURE BLOCKS")
print("=" * 100)

print("\nPRIMARY conventional baseline (14):")
for i, f in enumerate(CONVENTIONAL_PRIMARY14, 1):
    print(f"{i:02d}. {f}")

removed_primary = [f for f in CONVENTIONAL_ALL20 if f not in CONVENTIONAL_PRIMARY14]

print("\nRemoved from PRIMARY conventional baseline (still retained for sensitivity):")
for f in removed_primary:
    reason = ""
    if f.endswith("_range"):
        reason = "range = max - min (derived/redundant)"
    elif f == "spo2_std":
        reason = "strong redundancy with SpO2 mean/min"
    elif f == "spo2_max":
        reason = "strong ceiling effect / very low information"
    print(f"- {f}: {reason}")

print("\nCoupling primary block remains all 12 features.")
print("Sensitivity conventional block remains all original 20 features.")

# =============================================================================
# 5. FIND A LOCAL CLINICAL SOURCE WITHOUT READING OUTCOME VALUES
# =============================================================================
candidate_paths = [
    PROCESSED_DIR / "02_numeric_quality_eligible_cases.csv",
    PROCESSED_DIR / "01_primary_candidate_cases.csv",
    RAW_DIR / "cases.csv",
    PROJECT_DIR / "cases.csv",
]

# Also consider other CSVs under data_processed, but only inspect HEADERS first.
for p in sorted(PROCESSED_DIR.glob("*.csv")):
    if p not in candidate_paths and p != INPUT_PATH:
        candidate_paths.append(p)

source_rows = []
best_source = None
best_present = []

for path in candidate_paths:
    if not path.exists():
        continue

    try:
        header = pd.read_csv(path, nrows=0)
        cols = [str(c).replace("\ufeff", "").strip() for c in header.columns]

        if "caseid" not in cols:
            continue

        present = [c for c in CLINICAL_CANDIDATES if c in cols]

        source_rows.append({
            "path": str(path),
            "clinical_candidate_count": len(present),
            "clinical_candidates_present": ", ".join(present),
        })

        if len(present) > len(best_present):
            best_source = path
            best_present = present

    except Exception as e:
        source_rows.append({
            "path": str(path),
            "clinical_candidate_count": -1,
            "clinical_candidates_present": f"HEADER ERROR: {e}",
        })

source_audit = pd.DataFrame(source_rows)
source_audit.to_csv(OUT_SOURCE_AUDIT, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("CLINICAL SOURCE SEARCH")
print("=" * 100)

if len(source_audit):
    print(source_audit.to_string(index=False))
else:
    print("No candidate local CSV sources found.")

if best_source is None or len(best_present) == 0:
    print(
        "\nWARNING: Local clinical source with candidate covariates was not found.\n"
        "No outcome data were loaded. The script will save the signal dataset unchanged.\n"
        "If needed, clinical metadata can be recovered in the next step from the official VitalDB cases table."
    )
    clinical_df = pd.DataFrame({"caseid": base["caseid"].unique()})
    best_source_text = "NONE"
else:
    print(f"\nSelected clinical source: {best_source}")
    print(f"Clinical variables available: {', '.join(best_present)}")

    # Critical safeguard:
    # Load ONLY caseid + allowlisted clinical columns.
    # Postoperative outcomes and intraoperative treatment variables are never read.
    usecols = ["caseid"] + best_present

    assert not (set(usecols) & FORBIDDEN_COLUMNS), (
        "Safety check failed: forbidden outcome/post-exposure column requested."
    )

    clinical_df = pd.read_csv(best_source, usecols=usecols)
    clinical_df.columns = clinical_df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    clinical_df["caseid"] = pd.to_numeric(clinical_df["caseid"], errors="coerce")
    clinical_df = clinical_df.dropna(subset=["caseid"]).copy()
    clinical_df["caseid"] = clinical_df["caseid"].astype(int)

    # If a source unexpectedly has duplicate case rows, preserve one row per case.
    clinical_df = clinical_df.drop_duplicates(subset=["caseid"], keep="first")

    best_source_text = str(best_source)

# =============================================================================
# 6. MERGE ONLY ALLOWLISTED CLINICAL COVARIATES
# =============================================================================
merge_cols = ["caseid"] + [c for c in CLINICAL_CANDIDATES if c in clinical_df.columns]

preout = base.merge(
    clinical_df[merge_cols],
    on="caseid",
    how="left",
    validate="one_to_one",
)

# Assert forbidden columns were not newly introduced.
forbidden_present = sorted(set(preout.columns) & FORBIDDEN_COLUMNS)
if forbidden_present:
    # Some may have existed already in base from an earlier stage.
    # Remove them defensively from this dedicated pre-outcome file.
    preout = preout.drop(columns=forbidden_present, errors="ignore")

# =============================================================================
# 7. CLINICAL AUDIT
# =============================================================================
print("\n" + "=" * 100)
print("CLINICAL COVARIATE AUDIT")
print("=" * 100)

audit_rows = []

for col in CLINICAL_CANDIDATES:
    present = col in preout.columns

    if not present:
        row = {
            "candidate": col,
            "present": False,
            "missing_n": np.nan,
            "missing_pct": np.nan,
            "unique_n": np.nan,
            "dtype": "",
        }
    else:
        s = preout[col]
        row = {
            "candidate": col,
            "present": True,
            "missing_n": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100.0),
            "unique_n": int(s.nunique(dropna=True)),
            "dtype": str(s.dtype),
        }

    audit_rows.append(row)

clinical_audit = pd.DataFrame(audit_rows)
clinical_audit.to_csv(OUT_CLINICAL_AUDIT, index=False, encoding="utf-8-sig")
print(clinical_audit.to_string(index=False))

# =============================================================================
# 8. SAVE PRE-OUTCOME DATASET
# =============================================================================
preout.to_csv(OUT_DATA, index=False, encoding="utf-8-sig")

# =============================================================================
# 9. SAVE PRE-OUTCOME SPECIFICATION
# =============================================================================
specification = {
    "stage": "STEP 07C",
    "outcomes_inspected": False,
    "clinical_source": best_source_text,
    "primary_conventional_feature_count": len(CONVENTIONAL_PRIMARY14),
    "primary_conventional_features": CONVENTIONAL_PRIMARY14,
    "sensitivity_conventional_feature_count": len(CONVENTIONAL_ALL20),
    "sensitivity_conventional_features": CONVENTIONAL_ALL20,
    "coupling_feature_count": len(COUPLING12),
    "coupling_features": COUPLING12,
    "conventional_reduction_reason": {
        "all_range_features": "Removed from primary baseline because range=max-min and is deterministically derived from extrema.",
        "spo2_std": "Removed from primary baseline because STEP 07B showed very strong redundancy with SpO2 level/extreme features.",
        "spo2_max": "Removed from primary baseline because of ceiling effect and very low information content.",
        "sensitivity_analysis": "All original 20 conventional features remain prespecified for sensitivity analysis."
    },
    "future_model_structure": {
        "Model_A": "clinical covariates only; final clinical list to be frozen after this missingness/category audit",
        "Model_B": "Model A + 14 primary conventional signal features",
        "Model_C": "Model B + all 12 locked coupling features",
        "Sensitivity_B20": "Model A + all 20 original conventional features",
        "Sensitivity_C20": "Sensitivity_B20 + all 12 locked coupling features",
    },
    "future_cv": {
        "splitter": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 2026,
        "group": "subjectid",
    },
    "preprocessing": {
        "fit_inside_training_fold_only": True,
        "numeric_imputation": "median",
        "categorical_encoding": "one-hot",
        "logistic_scaling": "StandardScaler",
    },
}

OUT_SPEC_JSON.write_text(
    json.dumps(specification, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

txt = f"""
VitalDB Multimodal Physiological Coupling Study
STEP 07C - PRE-OUTCOME SIGNAL BLOCK FREEZE
============================================================

OUTCOMES INSPECTED: NO

PRIMARY CONVENTIONAL BLOCK: 14 FEATURES
---------------------------------------
{chr(10).join(CONVENTIONAL_PRIMARY14)}

SENSITIVITY CONVENTIONAL BLOCK: 20 FEATURES
-------------------------------------------
All original STEP 07 conventional features are retained.

COUPLING BLOCK: 12 FEATURES
---------------------------
All 12 STEP 04/05/06 locked coupling features remain unchanged.

RATIONALE
---------
1) All range features were removed from the PRIMARY baseline because
   range = max - min and STEP 07B showed substantial deterministic redundancy.

2) SpO2_std was removed because it was very strongly redundant with
   SpO2_mean / SpO2_min.

3) SpO2_max was removed because it shows a strong ceiling effect
   and carries little discriminatory information.

4) All 20 conventional features remain prespecified for sensitivity analysis.

FUTURE MODEL STRUCTURE
----------------------
Model A: clinical covariates only
Model B: Model A + 14 conventional primary features
Model C: Model B + 12 coupling features

Sensitivity:
B20 = Model A + all 20 conventional features
C20 = B20 + all 12 coupling features

CROSS-VALIDATION
----------------
StratifiedGroupKFold, 5 folds
group = subjectid
shuffle = True
random_state = 2026

All preprocessing is fit within training folds only.
"""

OUT_SPEC_TXT.write_text(txt.strip() + "\n", encoding="utf-8")

# =============================================================================
# 10. SUMMARY
# =============================================================================
available_clinical = clinical_audit.loc[
    clinical_audit["present"] == True, "candidate"
].tolist()

print("\n" + "=" * 100)
print("STEP 07C SUMMARY")
print("=" * 100)

print(f"\nCases preserved: {len(preout):,}")
print(f"Unique subjects: {preout['subjectid'].nunique():,}")
print(f"Primary conventional features: {len(CONVENTIONAL_PRIMARY14)}")
print(f"Sensitivity conventional features: {len(CONVENTIONAL_ALL20)}")
print(f"Coupling features: {len(COUPLING12)}")
print(f"Clinical source: {best_source_text}")
print(f"Clinical candidate variables recovered: {len(available_clinical)}")
print(f"Recovered: {', '.join(available_clinical) if available_clinical else 'NONE'}")
print("Outcome variables loaded/inspected: NO")

print("\nDecision:")
if len(available_clinical) >= 5:
    print("GOOD - Review clinical missingness/category counts and freeze Model A.")
elif len(available_clinical) > 0:
    print("REVIEW - Some clinical variables recovered; may need official cases table for a fuller baseline.")
else:
    print("REVIEW - Clinical covariates were not found locally; recover them before outcome modeling.")

print("\nGenerated files:")
for p in [
    OUT_DATA,
    OUT_CLINICAL_AUDIT,
    OUT_SOURCE_AUDIT,
    OUT_SPEC_JSON,
    OUT_SPEC_TXT,
]:
    print(p)

print(
    """
NEXT:
Send these console sections:

1) CLINICAL SOURCE SEARCH
2) CLINICAL COVARIATE AUDIT
3) STEP 07C SUMMARY

Then freeze the exact Model A clinical covariates BEFORE any outcome is merged.
"""
)