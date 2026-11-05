# ============================================================
# VitalDB Multimodal Physiological Coupling Study
# STEP 07 - FIXED VERSION
# Conventional Vital-Sign Feature Extraction
#
# IMPORTANT FIX
# ------------------------------------------------------------
# STEP 05/06 final cohort stores the locked first-30-min start time as
# `window_start_sec`, not `opstart_sec`.
# This version automatically resolves the analysis-start column in this order:
#   1) window_start_sec
#   2) opstart_sec
#   3) opstart
#
# Outcome variables are NOT used.
# ============================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data_raw"
PROCESSED_DIR = PROJECT_DIR / "data_processed"
RESULT_DIR = PROJECT_DIR / "results"
CACHE_DIR = RAW_DIR / "numeric_tracks"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "06_final_analysis_cohort_locked.csv"
FEATURE_PATH = PROCESSED_DIR / "07_conventional_vital_features.csv"
COMBINED_PATH = PROCESSED_DIR / "07_analysis_features_without_outcomes.csv"
SUMMARY_PATH = RESULT_DIR / "07_conventional_vital_feature_summary.csv"
MISSING_PATH = RESULT_DIR / "07_conventional_vital_missingness.csv"
ERROR_PATH = RESULT_DIR / "07_conventional_vital_errors.csv"

# ============================================================
# 2. LOCKED SETTINGS
# ============================================================
ANALYSIS_WINDOW_SEC = 30 * 60
INTERPOLATION_LIMIT_SEC = 10

SIGNAL_CONFIG = {
    "HR":   {"minimum": 20.0, "maximum": 250.0},
    "MAP":  {"minimum": 20.0, "maximum": 200.0},
    "SPO2": {"minimum": 50.0, "maximum": 100.0},
    "ETCO2":{"minimum": 5.0,  "maximum": 80.0},
}

STATISTICS = ["mean", "std", "min", "max", "range"]
LOCKED_CONVENTIONAL_FEATURES = [
    f"{signal.lower()}_{stat}"
    for signal in SIGNAL_CONFIG
    for stat in STATISTICS
]

# ============================================================
# 3. LOAD FINAL FROZEN COHORT
# ============================================================
print("=" * 92)
print("VitalDB Multimodal Physiological Coupling Study")
print("STEP 07 - Conventional Vital-Sign Feature Extraction [FIXED]")
print("=" * 92)

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"입력 파일이 없습니다:\n{INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

if "caseid" not in df.columns:
    raise ValueError("06_final_analysis_cohort_locked.csv에 caseid가 없습니다.")

df["caseid"] = pd.to_numeric(df["caseid"], errors="coerce")
df = df.dropna(subset=["caseid"]).copy()
df["caseid"] = df["caseid"].astype(int)

# ============================================================
# 4. RESOLVE ANALYSIS START COLUMN
# ============================================================
START_COLUMN_CANDIDATES = [
    "window_start_sec",  # STEP05/06 actual column
    "opstart_sec",
    "opstart",
]

ANALYSIS_START_COLUMN = None
for col in START_COLUMN_CANDIDATES:
    if col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() > 0:
            ANALYSIS_START_COLUMN = col
            df[col] = numeric
            break

if ANALYSIS_START_COLUMN is None:
    raise ValueError(
        "분석 시작시간 column을 찾을 수 없습니다. "
        "window_start_sec / opstart_sec / opstart 중 하나가 필요합니다."
    )

print(f"\nFinal frozen cohort: {len(df):,} cases")
if "subjectid" in df.columns:
    print(f"Unique subjects: {df['subjectid'].nunique():,}")
print(f"Analysis window: first {ANALYSIS_WINDOW_SEC // 60} min")
print(f"Analysis start column detected: {ANALYSIS_START_COLUMN}")
print(f"Locked conventional features: {len(LOCKED_CONVENTIONAL_FEATURES)}")
print("Outcome variables: NOT USED")

print("\n" + "=" * 92)
print("LOCKED CONVENTIONAL FEATURES")
print("=" * 92)
for i, feature in enumerate(LOCKED_CONVENTIONAL_FEATURES, start=1):
    print(f"{i:02d}. {feature}")

# ============================================================
# 5. CACHE HELPERS
# ============================================================
def get_cache_path(caseid, signal_name):
    return (
        CACHE_DIR
        / signal_name
        / f"case_{caseid:05d}_{signal_name}.csv.gz"
    )


def load_cached_signal(caseid, signal_name):
    path = get_cache_path(caseid, signal_name)
    if not path.exists():
        raise FileNotFoundError(f"cache 없음: {path}")

    temp = pd.read_csv(path)
    if "time_sec" not in temp.columns or "value" not in temp.columns:
        raise ValueError(f"{signal_name}: time_sec/value column 없음")

    temp["time_sec"] = pd.to_numeric(temp["time_sec"], errors="coerce")
    temp["value"] = pd.to_numeric(temp["value"], errors="coerce")
    temp = (
        temp.dropna(subset=["time_sec"])
        .sort_values("time_sec")
        .reset_index(drop=True)
    )
    return temp

# ============================================================
# 6. PREPARE LOCKED FIRST 30 MIN AT 1 Hz
# ============================================================
def prepare_first30_signal(raw_df, start_sec, minimum, maximum):
    end_sec = start_sec + ANALYSIS_WINDOW_SEC

    temp = raw_df.loc[
        (raw_df["time_sec"] >= start_sec)
        & (raw_df["time_sec"] < end_sec)
    ].copy()

    temp.loc[
        (temp["value"] < minimum) | (temp["value"] > maximum),
        "value"
    ] = np.nan

    start_integer = int(np.floor(start_sec))
    full_index = np.arange(
        start_integer,
        start_integer + ANALYSIS_WINDOW_SEC,
        dtype=int,
    )

    if len(temp) == 0:
        return pd.Series(index=full_index, dtype=float)

    temp["second"] = np.floor(temp["time_sec"]).astype(int)
    per_second = temp.groupby("second")["value"].median()
    series = per_second.reindex(full_index)

    # Same short-gap rule used previously: <=10 samples/seconds
    series = series.interpolate(
        method="linear",
        limit=INTERPOLATION_LIMIT_SEC,
        limit_area="inside",
    )

    return series

# ============================================================
# 7. STATISTICS
# ============================================================
def calculate_statistics(series, signal_name):
    values = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    prefix = signal_name.lower()
    coverage = float(len(values) / ANALYSIS_WINDOW_SEC)

    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_coverage": 0.0,
        }

    vmin = float(values.min())
    vmax = float(values.max())

    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=1)),
        f"{prefix}_min": vmin,
        f"{prefix}_max": vmax,
        f"{prefix}_range": float(vmax - vmin),
        f"{prefix}_coverage": coverage,
    }

# ============================================================
# 8. ONE CASE
# ============================================================
def analyze_case(row):
    caseid = int(row["caseid"])
    start_sec = pd.to_numeric(row[ANALYSIS_START_COLUMN], errors="coerce")

    if pd.isna(start_sec):
        raise ValueError(f"{ANALYSIS_START_COLUMN} missing")

    start_sec = float(start_sec)
    if start_sec < 0:
        raise ValueError("analysis start < 0")

    result = {
        "caseid": caseid,
        "analysis_start_sec": start_sec,
        "analysis_end_sec": start_sec + ANALYSIS_WINDOW_SEC,
    }

    for signal_name, config in SIGNAL_CONFIG.items():
        raw = load_cached_signal(caseid, signal_name)
        series = prepare_first30_signal(
            raw,
            start_sec,
            config["minimum"],
            config["maximum"],
        )
        result.update(calculate_statistics(series, signal_name))

    return result

# ============================================================
# 9. RUN
# ============================================================
print("\n" + "=" * 92)
print("CONVENTIONAL FEATURE EXTRACTION START")
print("=" * 92)

result_rows = []
error_rows = []
total = len(df)

for index, (_, row) in enumerate(df.iterrows(), start=1):
    caseid = int(row["caseid"])
    try:
        result_rows.append(analyze_case(row))
    except Exception as error:
        error_rows.append({"caseid": caseid, "error": str(error)})

    if index % 100 == 0 or index == total:
        print(
            f"{index:,} / {total:,} | "
            f"success={len(result_rows):,} | "
            f"errors={len(error_rows):,}"
        )

feature_df = pd.DataFrame(result_rows)
error_df = pd.DataFrame(error_rows)

if len(feature_df) == 0:
    if len(error_df) > 0:
        print("\nFIRST 10 ERRORS:")
        print(error_df.head(10).to_string(index=False))
    raise RuntimeError("추출된 conventional feature가 없습니다.")

# ============================================================
# 10. ADD SUBJECT ID
# ============================================================
id_columns = ["caseid"]
if "subjectid" in df.columns:
    id_columns.append("subjectid")

feature_df = feature_df.merge(
    df[id_columns],
    on="caseid",
    how="left",
    validate="one_to_one",
)

first_columns = ["caseid"]
if "subjectid" in feature_df.columns:
    first_columns.append("subjectid")

coverage_columns = [f"{signal.lower()}_coverage" for signal in SIGNAL_CONFIG]
feature_df = feature_df[
    first_columns
    + ["analysis_start_sec", "analysis_end_sec"]
    + LOCKED_CONVENTIONAL_FEATURES
    + coverage_columns
].copy()

# ============================================================
# 11. MISSING / INF AUDIT
# ============================================================
print("\n" + "=" * 92)
print("CONVENTIONAL FEATURE AUDIT")
print("=" * 92)

missing_rows = []
for feature in LOCKED_CONVENTIONAL_FEATURES:
    values = pd.to_numeric(feature_df[feature], errors="coerce")
    missing_n = int(values.isna().sum())
    inf_n = int(np.isinf(values).sum())
    missing_rows.append({
        "feature": feature,
        "missing_n": missing_n,
        "missing_pct": missing_n / len(feature_df) * 100,
        "inf_n": inf_n,
    })
    print(f"{feature:<20} missing={missing_n:>4} | inf={inf_n:>3}")

missing_df = pd.DataFrame(missing_rows)

# ============================================================
# 12. DISTRIBUTIONS
# ============================================================
print("\n" + "=" * 92)
print("CONVENTIONAL FEATURE DISTRIBUTIONS")
print("=" * 92)

summary_rows = []
for feature in LOCKED_CONVENTIONAL_FEATURES:
    values = (
        pd.to_numeric(feature_df[feature], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(values) == 0:
        continue

    summary_rows.append({
        "feature": feature,
        "N": len(values),
        "mean": values.mean(),
        "std": values.std(),
        "median": values.median(),
        "q1": values.quantile(0.25),
        "q3": values.quantile(0.75),
        "min": values.min(),
        "max": values.max(),
    })

    print(
        f"{feature:<20} N={len(values):,} | "
        f"median={values.median():.3f} | "
        f"IQR={values.quantile(0.25):.3f} to {values.quantile(0.75):.3f}"
    )

summary_df = pd.DataFrame(summary_rows)

# ============================================================
# 13. MERGE WITH LOCKED COUPLING FEATURES (NO OUTCOME)
# ============================================================
merge_keys = ["caseid"]
if "subjectid" in df.columns and "subjectid" in feature_df.columns:
    merge_keys.append("subjectid")

combined_df = df.merge(
    feature_df,
    on=merge_keys,
    how="inner",
    validate="one_to_one",
)

# ============================================================
# 14. SAVE
# ============================================================
feature_df.to_csv(FEATURE_PATH, index=False, encoding="utf-8-sig")
combined_df.to_csv(COMBINED_PATH, index=False, encoding="utf-8-sig")
summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
missing_df.to_csv(MISSING_PATH, index=False, encoding="utf-8-sig")
error_df.to_csv(ERROR_PATH, index=False, encoding="utf-8-sig")

# ============================================================
# 15. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 92)
print("STEP 07 SUMMARY")
print("=" * 92)

print(f"\nInput frozen cases: {len(df):,}")
print(f"Successfully extracted: {len(feature_df):,}")
print(f"Errors: {len(error_df):,}")
if "subjectid" in feature_df.columns:
    print(f"Subjects: {feature_df['subjectid'].nunique():,}")

features_with_missing = int((missing_df["missing_n"] > 0).sum())
features_with_inf = int((missing_df["inf_n"] > 0).sum())

print(f"\nLocked conventional features: {len(LOCKED_CONVENTIONAL_FEATURES)}")
print(f"Features with missing values: {features_with_missing}")
print(f"Features with inf values: {features_with_inf}")
print("Outcome variables inspected: NO")

# Coverage summary
print("\nSignal coverage in first 30 min:")
for signal in SIGNAL_CONFIG:
    col = f"{signal.lower()}_coverage"
    vals = feature_df[col].dropna()
    print(
        f"{signal:<6} median={vals.median()*100:.1f}% | "
        f"IQR={vals.quantile(0.25)*100:.1f}% to {vals.quantile(0.75)*100:.1f}%"
    )

print("\n" + "=" * 92)
print("STEP 07 DECISION")
print("=" * 92)

if len(feature_df) >= 2700 and features_with_inf == 0:
    decision = "VERY GOOD"
    reason = "Conventional vital features were successfully extracted for nearly the entire frozen cohort."
elif len(feature_df) >= 2500:
    decision = "GOOD"
    reason = "Adequate conventional feature cohort."
else:
    decision = "REVIEW"
    reason = "Unexpected loss during conventional feature extraction."

print(f"\nDecision: {decision}")
print(f"Reason: {reason}")

print("\n" + "=" * 92)
print("STEP 07 COMPLETE")
print("=" * 92)
print("\nGenerated files:")
print(FEATURE_PATH)
print(COMBINED_PATH)
print(SUMMARY_PATH)
print(MISSING_PATH)
print(ERROR_PATH)

print("""
NEXT STEP:

STEP 07B = outcome-independent conventional-feature audit.

1. Check redundancy among 20 conventional vital features.
2. Check overlap between conventional and 12 coupling features.
3. Freeze Model A / B / C feature sets.
4. Freeze preprocessing and subject-grouped CV.
5. Only then merge postoperative outcomes.
""")