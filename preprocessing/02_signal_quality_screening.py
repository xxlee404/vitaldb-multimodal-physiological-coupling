# ============================================================
# VitalDB Multimodal Physiological Coupling Study
# STEP 02
#
# Numeric Signal Quality & Common-Overlap Screening
#
# ------------------------------------------------------------
# 대상
# STEP 01:
# Adult + ECG + PLETH + ART + ART_MBP + ETCO2
#
# STEP 02에서는 500-Hz waveform을 아직 받지 않고
# 다음 numeric signal의 실제 수술 중 품질을 먼저 확인
#
# - Solar8000/HR
# - Solar8000/ART_MBP
# - Solar8000/PLETH_SPO2
# - Solar8000/ETCO2
#
# ------------------------------------------------------------
# Artifact bounds
#
# HR      : 20–250 bpm
# MAP     : 20–200 mmHg
# SpO2    : 50–100 %
# ETCO2   : 5–80 mmHg
#
# ------------------------------------------------------------
# Short gap interpolation
#
# <= 10 sec
#
# ------------------------------------------------------------
# Initial signal-quality criteria
#
# Operation duration       >= 30 min
# Each signal coverage     >= 50%
# Common valid duration    >= 30 min
# Common coverage          >= 40%
# Longest common segment   >= 10 min
#
# 위 기준은 임상 cutoff가 아니라
# waveform 분석 feasibility screening용 기준
# ============================================================


from pathlib import Path
from io import BytesIO

import gzip
import time
import warnings

import numpy as np
import pandas as pd
import requests


warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_DIR / "data_raw"

PROCESSED_DIR = PROJECT_DIR / "data_processed"

RESULT_DIR = PROJECT_DIR / "results"

CACHE_DIR = RAW_DIR / "numeric_tracks"


for folder in [
    RAW_DIR,
    PROCESSED_DIR,
    RESULT_DIR,
    CACHE_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 2. INPUT FILES
# ============================================================

CASES_PATH = (
    RAW_DIR
    / "vitaldb_cases.csv"
)

TRACKS_PATH = (
    RAW_DIR
    / "vitaldb_tracks.csv"
)

CANDIDATE_PATH = (
    PROCESSED_DIR
    / "01_full_multimodal_candidate_cases.csv"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

QUALITY_PATH = (
    RESULT_DIR
    / "02_numeric_signal_quality.csv"
)

ELIGIBLE_PATH = (
    PROCESSED_DIR
    / "02_numeric_quality_eligible_cases.csv"
)

EXCLUDED_PATH = (
    RESULT_DIR
    / "02_numeric_quality_excluded_cases.csv"
)

ERROR_PATH = (
    RESULT_DIR
    / "02_numeric_signal_errors.csv"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "02_numeric_signal_quality_summary.csv"
)

CHECKPOINT_PATH = (
    RESULT_DIR
    / "02_numeric_quality_checkpoint.csv"
)

REASON_COUNT_PATH = (
    RESULT_DIR
    / "02_exclusion_reason_counts.csv"
)


# ============================================================
# 4. SIGNAL SETTINGS
# ============================================================

TRACK_CONFIG = {

    "HR": {
        "tname": "Solar8000/HR",
        "min": 20.0,
        "max": 250.0,
    },

    "MAP": {
        "tname": "Solar8000/ART_MBP",
        "min": 20.0,
        "max": 200.0,
    },

    "SPO2": {
        "tname": "Solar8000/PLETH_SPO2",
        "min": 50.0,
        "max": 100.0,
    },

    "ETCO2": {
        "tname": "Solar8000/ETCO2",
        "min": 5.0,
        "max": 80.0,
    },
}


# ============================================================
# 5. QUALITY CRITERIA
# ============================================================

INTERPOLATION_LIMIT_SEC = 10

MIN_OPERATION_DURATION_MIN = 30.0

MIN_SIGNAL_COVERAGE = 0.50

MIN_COMMON_DURATION_MIN = 30.0

MIN_COMMON_COVERAGE = 0.40

MIN_LONGEST_COMMON_RUN_MIN = 10.0


# ============================================================
# 6. NETWORK SETTINGS
# ============================================================

REQUEST_TIMEOUT = 120

MAX_RETRIES = 3

RETRY_WAIT_SEC = 2


# ============================================================
# 7. START
# ============================================================

print("=" * 90)
print("VitalDB Multimodal Physiological Coupling Study")
print("STEP 02 - Numeric Signal Quality & Common Overlap")
print("=" * 90)


for path in [
    CASES_PATH,
    TRACKS_PATH,
    CANDIDATE_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"필요한 파일이 없습니다:\n{path}"
        )


# ============================================================
# 8. LOAD
# ============================================================

cases = pd.read_csv(
    CASES_PATH
)

tracks = pd.read_csv(
    TRACKS_PATH
)

candidate = pd.read_csv(
    CANDIDATE_PATH
)


for dataframe in [
    cases,
    tracks,
    candidate,
]:

    dataframe.columns = (
        dataframe.columns
        .str.replace(
            "\ufeff",
            "",
            regex=False
        )
        .str.strip()
    )


# ============================================================
# 9. CASE ID CLEANING
# ============================================================

for dataframe in [
    cases,
    tracks,
    candidate,
]:

    if "caseid" not in dataframe.columns:

        raise ValueError(
            "caseid column이 없습니다."
        )

    dataframe["caseid"] = pd.to_numeric(
        dataframe["caseid"],
        errors="coerce"
    )

    dataframe.dropna(
        subset=["caseid"],
        inplace=True
    )

    dataframe["caseid"] = (
        dataframe["caseid"]
        .astype(int)
    )


if "tname" not in tracks.columns:

    raise ValueError(
        "vitaldb_tracks.csv에 tname column이 없습니다."
    )


if "tid" not in tracks.columns:

    raise ValueError(
        "vitaldb_tracks.csv에 tid column이 없습니다."
    )


tracks["tname"] = (
    tracks["tname"]
    .astype(str)
    .str.strip()
)


print(
    f"\nSTEP 01 full multimodal candidate: "
    f"{len(candidate):,} cases"
)


if "subjectid" in candidate.columns:

    print(
        f"Subjects: "
        f"{candidate['subjectid'].nunique():,}"
    )


# ============================================================
# 10. TIME COLUMNS
# ============================================================

for col in [
    "opstart",
    "opend",
    "caseend",
]:

    if col in candidate.columns:

        candidate[col] = pd.to_numeric(
            candidate[col],
            errors="coerce"
        )


# ============================================================
# 11. TID LOOKUP
# ============================================================

print("\n")
print("=" * 90)
print("NUMERIC TRACK TID AVAILABILITY")
print("=" * 90)


tid_lookup = {}


for signal_name, config in TRACK_CONFIG.items():

    tname = config["tname"]

    subset = tracks.loc[
        tracks["tname"] == tname,
        [
            "caseid",
            "tid"
        ]
    ].copy()

    subset["tid"] = (
        subset["tid"]
        .astype(str)
        .str.strip()
    )

    tid_lookup[signal_name] = {

        int(row["caseid"]):
            str(row["tid"])

        for _, row in subset.iterrows()
    }

    print(
        f"{signal_name:<10} "
        f"{tname:<30} "
        f"{len(tid_lookup[signal_name]):,} cases"
    )


# ============================================================
# 12. REQUEST SESSION
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent":
            "VitalDB-Multimodal-Research/1.0"
    }
)


# ============================================================
# 13. DOWNLOAD TRACK
# ============================================================

def download_track_dataframe(
    tid
):

    url = (
        f"https://api.vitaldb.net/{tid}"
    )

    last_error = None


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            content = response.content


            # ------------------------------------------------
            # gzip 압축 여부 확인
            # ------------------------------------------------

            if (
                len(content) >= 2
                and
                content[:2] == b"\x1f\x8b"
            ):

                content = gzip.decompress(
                    content
                )


            # ------------------------------------------------
            # VitalDB track CSV
            # header 유무와 관계없이 robust하게 처리
            # ------------------------------------------------

            temp = pd.read_csv(
                BytesIO(content),
                header=None
            )


            if temp.shape[1] < 2:

                raise ValueError(
                    "Track column 수가 2개 미만입니다."
                )


            temp = (
                temp.iloc[
                    :,
                    :2
                ]
                .copy()
            )


            temp.columns = [
                "time_sec",
                "value"
            ]


            temp["time_sec"] = pd.to_numeric(
                temp["time_sec"],
                errors="coerce"
            )


            temp["value"] = pd.to_numeric(
                temp["value"],
                errors="coerce"
            )


            # header row가 들어왔다면
            # time이 숫자가 아니므로 여기서 자동 제거
            temp = (
                temp
                .dropna(
                    subset=[
                        "time_sec"
                    ]
                )
                .sort_values(
                    "time_sec"
                )
                .reset_index(
                    drop=True
                )
            )


            if len(temp) == 0:

                raise ValueError(
                    "유효한 track data가 없습니다."
                )


            return temp


        except Exception as error:

            last_error = error


            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_WAIT_SEC
                    * attempt
                )


    raise RuntimeError(
        f"TID {tid} 다운로드 실패: "
        f"{last_error}"
    )


# ============================================================
# 14. CACHE PATH
# ============================================================

def get_cache_path(
    caseid,
    signal_name
):

    signal_folder = (
        CACHE_DIR
        / signal_name
    )


    signal_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    return (
        signal_folder
        /
        (
            f"case_{caseid:05d}_"
            f"{signal_name}.csv.gz"
        )
    )


# ============================================================
# 15. LOAD SIGNAL
# ============================================================

def load_signal(
    caseid,
    signal_name
):

    cache_path = get_cache_path(
        caseid,
        signal_name
    )


    # --------------------------------------------------------
    # 이미 다운로드했다면 cache 사용
    # --------------------------------------------------------

    if cache_path.exists():

        cached = pd.read_csv(
            cache_path
        )


        if (
            "time_sec"
            in cached.columns
            and
            "value"
            in cached.columns
        ):

            return cached


    # --------------------------------------------------------
    # TID 확인
    # --------------------------------------------------------

    tid = (
        tid_lookup[
            signal_name
        ]
        .get(
            caseid
        )
    )


    if tid is None:

        raise ValueError(
            f"{signal_name} TID 없음"
        )


    temp = download_track_dataframe(
        tid
    )


    temp.to_csv(
        cache_path,
        index=False,
        compression="gzip"
    )


    return temp


# ============================================================
# 16. SHORT GAP INTERPOLATION
# ============================================================

def interpolate_short_gaps(
    series,
    max_gap=10
):

    original = (
        series
        .astype(float)
        .copy()
    )

    output = (
        original.copy()
    )


    is_missing = (
        original.isna()
    )


    if not is_missing.any():

        return output


    # --------------------------------------------------------
    # 전체 interpolation 후보
    # --------------------------------------------------------

    full_interp = (
        original
        .interpolate(
            method="linear",
            limit_area="inside"
        )
    )


    # --------------------------------------------------------
    # 연속 결측 run 번호
    # --------------------------------------------------------

    run_id = (
        is_missing
        .ne(
            is_missing.shift()
        )
        .cumsum()
    )


    missing_indices = (
        is_missing[
            is_missing
        ]
        .index
    )


    if len(
        missing_indices
    ) == 0:

        return output


    missing_run_ids = (
        run_id.loc[
            missing_indices
        ]
    )


    # --------------------------------------------------------
    # 각 결측 run 검사
    # --------------------------------------------------------

    for run_number in (
        missing_run_ids.unique()
    ):

        indices = (
            missing_run_ids[
                missing_run_ids
                == run_number
            ]
            .index
            .tolist()
        )


        gap_length = len(
            indices
        )


        if gap_length <= max_gap:

            output.loc[
                indices
            ] = full_interp.loc[
                indices
            ]


    return output


# ============================================================
# 17. PREPARE SIGNAL TO 1 Hz
# ============================================================

def prepare_signal_1hz(
    raw_df,
    opstart,
    opend,
    minimum,
    maximum
):

    temp = raw_df.copy()


    temp["time_sec"] = pd.to_numeric(
        temp["time_sec"],
        errors="coerce"
    )


    temp["value"] = pd.to_numeric(
        temp["value"],
        errors="coerce"
    )


    temp = temp.dropna(
        subset=[
            "time_sec"
        ]
    )


    # --------------------------------------------------------
    # Operation window
    # --------------------------------------------------------

    temp = temp.loc[
        (
            temp["time_sec"]
            >= opstart
        )
        &
        (
            temp["time_sec"]
            <= opend
        )
    ].copy()


    # --------------------------------------------------------
    # Artifact rejection
    # --------------------------------------------------------

    temp.loc[
        (
            temp["value"]
            < minimum
        )
        |
        (
            temp["value"]
            > maximum
        ),
        "value"
    ] = np.nan


    # --------------------------------------------------------
    # Full 1-second grid
    # --------------------------------------------------------

    start_second = int(
        np.floor(
            opstart
        )
    )


    end_second = int(
        np.floor(
            opend
        )
    )


    if end_second < start_second:

        raise ValueError(
            "operation time이 올바르지 않습니다."
        )


    full_index = np.arange(
        start_second,
        end_second + 1,
        dtype=int
    )


    # --------------------------------------------------------
    # 해당 수술구간에 track data 없음
    # --------------------------------------------------------

    if len(temp) == 0:

        series = pd.Series(
            index=full_index,
            dtype=float
        )


    else:

        temp["second"] = np.floor(
            temp["time_sec"]
        ).astype(int)


        # 같은 초에 여러 값이 있으면 median
        second_values = (
            temp
            .groupby(
                "second"
            )["value"]
            .median()
        )


        series = (
            second_values
            .reindex(
                full_index
            )
        )


    # --------------------------------------------------------
    # interpolation 전 coverage
    # --------------------------------------------------------

    raw_valid = (
        series.notna()
    )


    # --------------------------------------------------------
    # 10초 이하 gap만 interpolation
    # --------------------------------------------------------

    interpolated = (
        interpolate_short_gaps(
            series,
            INTERPOLATION_LIMIT_SEC
        )
    )


    return (
        interpolated,
        raw_valid
    )


# ============================================================
# 18. LONGEST TRUE RUN
# ============================================================

def longest_true_run(
    boolean_array
):

    arr = np.asarray(
        boolean_array,
        dtype=bool
    )


    maximum = 0

    current = 0


    for value in arr:

        if value:

            current += 1

            if current > maximum:

                maximum = current

        else:

            current = 0


    return maximum


# ============================================================
# 19. COUNT VALID BLOCKS
# ============================================================

def count_valid_blocks(
    boolean_array,
    minimum_length_sec
):

    arr = np.asarray(
        boolean_array,
        dtype=bool
    )


    count = 0

    current = 0


    for value in arr:

        if value:

            current += 1

        else:

            if (
                current
                >= minimum_length_sec
            ):

                count += 1


            current = 0


    if (
        current
        >= minimum_length_sec
    ):

        count += 1


    return count


# ============================================================
# 20. GET OPERATION WINDOW
# ============================================================

def get_operation_window(
    row
):

    opstart = pd.to_numeric(
        row.get(
            "opstart",
            np.nan
        ),
        errors="coerce"
    )


    opend = pd.to_numeric(
        row.get(
            "opend",
            np.nan
        ),
        errors="coerce"
    )


    # --------------------------------------------------------
    # 우선 opstart-opend 사용
    # --------------------------------------------------------

    if (
        pd.notna(
            opstart
        )
        and
        pd.notna(
            opend
        )
        and
        opend > opstart
    ):

        return (
            float(opstart),
            float(opend),
            "opstart_opend"
        )


    # --------------------------------------------------------
    # fallback = 0 ~ caseend
    # --------------------------------------------------------

    caseend = pd.to_numeric(
        row.get(
            "caseend",
            np.nan
        ),
        errors="coerce"
    )


    if (
        pd.notna(
            caseend
        )
        and
        caseend > 0
    ):

        return (
            0.0,
            float(caseend),
            "zero_caseend"
        )


    raise ValueError(
        "유효한 operation time 없음"
    )


# ============================================================
# 21. ANALYZE ONE CASE
# ============================================================

def analyze_case(
    row
):

    caseid = int(
        row["caseid"]
    )


    (
        opstart,
        opend,
        operation_window_source

    ) = get_operation_window(
        row
    )


    operation_duration_sec = (
        opend
        - opstart
    )


    operation_duration_min = (
        operation_duration_sec
        / 60.0
    )


    if operation_duration_sec <= 0:

        raise ValueError(
            "operation duration <= 0"
        )


    # ========================================================
    # LOAD SIGNALS
    # ========================================================

    signal_series = {}

    raw_valid_masks = {}


    for (
        signal_name,
        config

    ) in TRACK_CONFIG.items():


        raw = load_signal(
            caseid,
            signal_name
        )


        (
            prepared,
            raw_valid

        ) = prepare_signal_1hz(

            raw_df=raw,

            opstart=opstart,

            opend=opend,

            minimum=config["min"],

            maximum=config["max"]
        )


        signal_series[
            signal_name
        ] = prepared


        raw_valid_masks[
            signal_name
        ] = raw_valid


    # ========================================================
    # COMBINE
    # ========================================================

    signal_df = pd.DataFrame(
        signal_series
    )


    n_seconds = len(
        signal_df
    )


    if n_seconds == 0:

        raise ValueError(
            "operation signal length = 0"
        )


    # ========================================================
    # BASIC RESULT
    # ========================================================

    metrics = {

        "caseid":
            caseid,

        "operation_window_source":
            operation_window_source,

        "opstart_sec":
            opstart,

        "opend_sec":
            opend,

        "operation_duration_min":
            operation_duration_min,
    }


    coverage_list = []


    # ========================================================
    # INDIVIDUAL SIGNAL COVERAGE
    # ========================================================

    for signal_name in (
        TRACK_CONFIG.keys()
    ):

        raw_mask = (
            raw_valid_masks[
                signal_name
            ]
            .reindex(
                signal_df.index,
                fill_value=False
            )
        )


        raw_coverage = (
            raw_mask.sum()
            / n_seconds
        )


        valid_coverage = (
            signal_df[
                signal_name
            ]
            .notna()
            .mean()
        )


        valid_minutes = (
            signal_df[
                signal_name
            ]
            .notna()
            .sum()
            / 60.0
        )


        metrics[
            f"{signal_name.lower()}_raw_coverage"
        ] = float(
            raw_coverage
        )


        metrics[
            f"{signal_name.lower()}_valid_coverage"
        ] = float(
            valid_coverage
        )


        metrics[
            f"{signal_name.lower()}_valid_minutes"
        ] = float(
            valid_minutes
        )


        coverage_list.append(
            valid_coverage
        )


    # ========================================================
    # COMMON VALID OVERLAP
    # ========================================================

    common_valid = (

        signal_df[
            list(
                TRACK_CONFIG.keys()
            )
        ]

        .notna()

        .all(
            axis=1
        )
    )


    common_seconds = int(
        common_valid.sum()
    )


    common_minutes = (
        common_seconds
        / 60.0
    )


    common_coverage = (
        common_seconds
        / n_seconds
    )


    longest_common_sec = (
        longest_true_run(
            common_valid.to_numpy()
        )
    )


    longest_common_min = (
        longest_common_sec
        / 60.0
    )


    common_blocks_5min = (
        count_valid_blocks(
            common_valid.to_numpy(),
            5 * 60
        )
    )


    common_blocks_10min = (
        count_valid_blocks(
            common_valid.to_numpy(),
            10 * 60
        )
    )


    metrics[
        "common_valid_minutes"
    ] = float(
        common_minutes
    )


    metrics[
        "common_valid_coverage"
    ] = float(
        common_coverage
    )


    metrics[
        "longest_common_valid_min"
    ] = float(
        longest_common_min
    )


    metrics[
        "common_blocks_ge5min"
    ] = int(
        common_blocks_5min
    )


    metrics[
        "common_blocks_ge10min"
    ] = int(
        common_blocks_10min
    )


    metrics[
        "minimum_signal_coverage"
    ] = float(
        min(
            coverage_list
        )
    )


    # ========================================================
    # QUALITY PASS / FAIL
    # ========================================================

    reasons = []


    if (
        operation_duration_min
        < MIN_OPERATION_DURATION_MIN
    ):

        reasons.append(
            "operation_duration_lt30min"
        )


    for signal_name in (
        TRACK_CONFIG.keys()
    ):

        coverage = metrics[
            f"{signal_name.lower()}_valid_coverage"
        ]


        if (
            coverage
            < MIN_SIGNAL_COVERAGE
        ):

            reasons.append(
                f"{signal_name.lower()}_coverage_lt50pct"
            )


    if (
        common_minutes
        < MIN_COMMON_DURATION_MIN
    ):

        reasons.append(
            "common_duration_lt30min"
        )


    if (
        common_coverage
        < MIN_COMMON_COVERAGE
    ):

        reasons.append(
            "common_coverage_lt40pct"
        )


    if (
        longest_common_min
        < MIN_LONGEST_COMMON_RUN_MIN
    ):

        reasons.append(
            "longest_common_run_lt10min"
        )


    passed = (
        len(
            reasons
        )
        == 0
    )


    metrics[
        "quality_pass"
    ] = int(
        passed
    )


    metrics[
        "exclusion_reason"
    ] = (

        ";".join(
            reasons
        )

        if reasons

        else ""
    )


    return metrics


# ============================================================
# 22. SCREEN ALL CASES
# ============================================================

print("\n")
print("=" * 90)
print("NUMERIC SIGNAL QUALITY SCREENING START")
print("=" * 90)


result_rows = []

error_rows = []


total = len(
    candidate
)


for index, (_, row) in enumerate(
    candidate.iterrows(),
    start=1
):

    caseid = int(
        row["caseid"]
    )


    try:

        result = analyze_case(
            row
        )

        result_rows.append(
            result
        )


    except Exception as error:

        error_rows.append(
            {
                "caseid":
                    caseid,

                "error":
                    str(
                        error
                    ),
            }
        )


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        index % 25 == 0
        or
        index == total
    ):

        pass_n = sum(

            int(
                item[
                    "quality_pass"
                ]
            )

            for item in result_rows
        )


        print(

            f"{index:,} / {total:,} | "

            f"success="
            f"{len(result_rows):,} | "

            f"pass="
            f"{pass_n:,} | "

            f"error="
            f"{len(error_rows):,}"
        )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if (
        index % 100 == 0
        or
        index == total
    ):

        pd.DataFrame(
            result_rows
        ).to_csv(

            CHECKPOINT_PATH,

            index=False,

            encoding="utf-8-sig"
        )


# ============================================================
# 23. RESULT DATAFRAMES
# ============================================================

quality_df = pd.DataFrame(
    result_rows
)


error_df = pd.DataFrame(
    error_rows
)


if len(
    quality_df
) == 0:

    raise RuntimeError(
        "분석 성공 case가 없습니다."
    )


# ============================================================
# 24. MERGE CLINICAL INFO
# ============================================================

merge_columns = [

    col

    for col in [

        "caseid",
        "subjectid",
        "age",
        "sex",
        "asa",
        "department",
        "optype",
        "approach",
        "icu_days",
        "death_inhosp",
        "los_postop",
        "los_postop_days",

    ]

    if col
    in candidate.columns
]


quality_df = quality_df.merge(

    candidate[
        merge_columns
    ],

    on="caseid",

    how="left",

    validate="one_to_one"
)


# ============================================================
# 25. ELIGIBLE / EXCLUDED
# ============================================================

eligible_df = quality_df.loc[

    quality_df[
        "quality_pass"
    ]
    == 1

].copy()


excluded_df = quality_df.loc[

    quality_df[
        "quality_pass"
    ]
    == 0

].copy()


# ============================================================
# 26. SCREENING SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("STEP 02 SCREENING SUMMARY")
print("=" * 90)


print(
    f"\nInitial candidate: "
    f"{total:,}"
)


print(
    f"Successfully evaluated: "
    f"{len(quality_df):,}"
)


print(
    f"Errors: "
    f"{len(error_df):,}"
)


print(
    f"\nQuality eligible: "
    f"{len(eligible_df):,}"
)


print(
    f"Excluded by quality criteria: "
    f"{len(excluded_df):,}"
)


if (
    "subjectid"
    in eligible_df.columns
):

    print(
        f"Eligible unique subjects: "
        f"{eligible_df['subjectid'].nunique():,}"
    )


# ============================================================
# 27. SIGNAL COVERAGE
# ============================================================

print("\n")
print("=" * 90)
print("SIGNAL COVERAGE")
print("=" * 90)


for signal_name in (
    TRACK_CONFIG.keys()
):

    col = (
        f"{signal_name.lower()}_valid_coverage"
    )


    values = (
        quality_df[
            col
        ]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .dropna()
    )


    print(
        f"\n{signal_name}"
    )


    if len(
        values
    ) > 0:

        print(
            f"Median coverage = "
            f"{values.median()*100:.1f}%"
        )


        print(
            f"IQR = "
            f"{values.quantile(0.25)*100:.1f}% - "
            f"{values.quantile(0.75)*100:.1f}%"
        )


    else:

        print(
            "No valid coverage values"
        )


# ============================================================
# 28. COMMON OVERLAP
# ============================================================

print("\n")
print("=" * 90)
print("COMMON OVERLAP")
print("=" * 90)


for col in [

    "common_valid_minutes",

    "common_valid_coverage",

    "longest_common_valid_min",

]:

    values = (
        quality_df[
            col
        ]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .dropna()
    )


    print(
        f"\n{col}"
    )


    if len(
        values
    ) == 0:

        print(
            "No valid values"
        )

        continue


    if (
        col
        == "common_valid_coverage"
    ):

        print(
            f"Median = "
            f"{values.median()*100:.1f}%"
        )


        print(
            f"IQR = "
            f"{values.quantile(0.25)*100:.1f}% - "
            f"{values.quantile(0.75)*100:.1f}%"
        )


    else:

        print(
            f"Median = "
            f"{values.median():.3f}"
        )


        print(
            f"IQR = "
            f"{values.quantile(0.25):.3f} - "
            f"{values.quantile(0.75):.3f}"
        )


# ============================================================
# 29. EXCLUSION REASONS
# ============================================================

print("\n")
print("=" * 90)
print("EXCLUSION REASONS")
print("=" * 90)


reason_counts = {}


for text in (
    excluded_df[
        "exclusion_reason"
    ]
    .fillna("")
):

    for reason in (
        str(text)
        .split(";")
    ):

        reason = (
            reason.strip()
        )


        if not reason:

            continue


        reason_counts[
            reason
        ] = (

            reason_counts.get(
                reason,
                0
            )
            + 1
        )


reason_df = pd.DataFrame(

    [

        {
            "reason":
                reason,

            "n_cases":
                n_cases,
        }

        for (
            reason,
            n_cases
        )
        in reason_counts.items()
    ]
)


if len(
    reason_df
) > 0:

    reason_df = (
        reason_df
        .sort_values(
            "n_cases",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "\n"
        + reason_df.to_string(
            index=False
        )
    )


else:

    print(
        "\nNo quality exclusions"
    )


# ============================================================
# 30. OUTCOME CHECK
# ============================================================

if (
    "icu_days"
    in eligible_df.columns
):

    icu = pd.to_numeric(

        eligible_df[
            "icu_days"
        ],

        errors="coerce"
    )


    valid_icu = (
        icu.notna()
    )


    icu_positive = int(

        (
            icu[
                valid_icu
            ]
            > 0
        )
        .sum()
    )


    icu_n = int(
        valid_icu.sum()
    )


    print("\n")
    print("=" * 90)
    print("OUTCOME CHECK IN ELIGIBLE COHORT")
    print("=" * 90)


    print(
        f"\nICU utilization N: "
        f"{icu_n:,}"
    )


    if icu_n > 0:

        print(

            f"ICU > 0 days: "
            f"{icu_positive:,} "
            f"({icu_positive / icu_n * 100:.1f}%)"
        )


# ============================================================
# 31. SAVE
# ============================================================

quality_df.to_csv(

    QUALITY_PATH,

    index=False,

    encoding="utf-8-sig"
)


eligible_df.to_csv(

    ELIGIBLE_PATH,

    index=False,

    encoding="utf-8-sig"
)


excluded_df.to_csv(

    EXCLUDED_PATH,

    index=False,

    encoding="utf-8-sig"
)


error_df.to_csv(

    ERROR_PATH,

    index=False,

    encoding="utf-8-sig"
)


reason_df.to_csv(

    REASON_COUNT_PATH,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 32. SUMMARY CSV
# ============================================================

summary_data = {

    "initial_candidate_cases":
        total,

    "evaluated_cases":
        len(
            quality_df
        ),

    "error_cases":
        len(
            error_df
        ),

    "quality_eligible_cases":
        len(
            eligible_df
        ),

    "quality_excluded_cases":
        len(
            excluded_df
        ),

    "eligible_subjects":

        (
            eligible_df[
                "subjectid"
            ]
            .nunique()

            if (
                "subjectid"
                in eligible_df.columns
            )

            else np.nan
        ),

    "median_common_valid_min":

        quality_df[
            "common_valid_minutes"
        ]
        .median(),

    "median_common_coverage":

        quality_df[
            "common_valid_coverage"
        ]
        .median(),

    "median_longest_common_min":

        quality_df[
            "longest_common_valid_min"
        ]
        .median(),
}


summary_df = pd.DataFrame(
    [
        summary_data
    ]
)


summary_df.to_csv(

    SUMMARY_PATH,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 33. FINAL FEASIBILITY
# ============================================================

eligible_n = len(
    eligible_df
)


print("\n")
print("=" * 90)
print("FINAL STEP 02 FEASIBILITY")
print("=" * 90)


if (
    eligible_n
    >= 2500
):

    decision = (
        "VERY GOOD"
    )


    explanation = (
        "충분한 usable multimodal cohort. "
        "본 연구를 그대로 진행."
    )


elif (
    eligible_n
    >= 1800
):

    decision = (
        "GOOD"
    )


    explanation = (
        "signal-quality exclusion 후에도 충분한 표본. "
        "연구 진행 가능."
    )


elif (
    eligible_n
    >= 1200
):

    decision = (
        "ACCEPTABLE"
    )


    explanation = (
        "진행 가능하지만 다음 waveform 단계에서 "
        "추가 표본 감소를 주의해야 함."
    )


else:

    decision = (
        "REVIEW"
    )


    explanation = (
        "실제 usable cohort 감소가 큼. "
        "신호 조합 또는 연구설계 재검토 필요."
    )


print(
    f"\nEligible cases: "
    f"{eligible_n:,}"
)


if (
    "subjectid"
    in eligible_df.columns
):

    print(
        f"Eligible subjects: "
        f"{eligible_df['subjectid'].nunique():,}"
    )


print(
    f"\nDecision: "
    f"{decision}"
)


print(
    f"Reason: "
    f"{explanation}"
)


# ============================================================
# 34. FINISH
# ============================================================

print("\n")
print("=" * 90)
print("STEP 02 완료")
print("=" * 90)


print(
    "\n생성 파일:"
)


print(
    QUALITY_PATH
)


print(
    ELIGIBLE_PATH
)


print(
    EXCLUDED_PATH
)


print(
    ERROR_PATH
)


print(
    REASON_COUNT_PATH
)


print(
    SUMMARY_PATH
)


print("\n")
print("=" * 90)
print("NEXT STEP")
print("=" * 90)


print(
    """
STEP 03에서는 quality-eligible cohort 전체 waveform을
바로 다운로드하지 않습니다.

먼저 소수 대표 case에서 ECG / PLETH / ART waveform pilot을 수행합니다.

확인 예정:

1. ECG waveform sampling 확인
2. ECG R-peak 검출
3. PLETH pulse peak / foot 검출
4. ART systolic pulse 검출
5. ECG -> PLETH timing
6. ECG -> ART timing
7. PLETH <-> ART amplitude relationship
8. cross-correlation / lag 계산

pilot이 정상적으로 동작한 후 전체 cohort로 확장합니다.
"""
)