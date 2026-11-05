# ============================================================
# VitalDB Multimodal Physiological Coupling Study
# STEP 01
#
# Check Multimodal Biosignal Track Availability
#
# Target journal:
# Biomedical Signal Processing and Control
#
# ============================================================
#
# 연구 후보:
#
# Multimodal Physiological Coupling and Decoupling
# During Surgery Using ECG, PPG, Arterial Pressure
# and Capnography
#
# ------------------------------------------------------------
# 주요 신호
#
# ECG waveform
#   SNUADC/ECG_II
#
# PPG waveform
#   SNUADC/PLETH
#
# ART waveform
#   SNUADC/ART
#
# Numeric MAP
#   Solar8000/ART_MBP
#
# HR
#   Solar8000/HR
#
# SpO2
#   Solar8000/PLETH_SPO2
#
# ETCO2
#   Solar8000/ETCO2
#
#
# STEP 01 목적
# ------------------------------------------------------------
# 1. 각 signal track 보유 case 수 확인
# 2. 성인 case 수 확인
# 3. ECG + PPG + ETCO2 조합 확인
# 4. ECG + PPG + ART + ETCO2 조합 확인
# 5. ECG + PPG + ART + MAP + ETCO2 조합 확인
# 6. subject 수 확인
# 7. 연구 진행 가능성 판단
#
# 아직 실제 signal quality는 검사하지 않음.
# STEP 02에서 신호 유효시간과 품질을 확인.
# ============================================================


from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import requests


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]


RAW_DIR = (
    PROJECT_DIR
    / "data_raw"
)


PROCESSED_DIR = (
    PROJECT_DIR
    / "data_processed"
)


RESULT_DIR = (
    PROJECT_DIR
    / "results"
)


FIGURE_DIR = (
    PROJECT_DIR
    / "figures"
)


CODE_DIR = (
    PROJECT_DIR
    / "code"
)


for folder in [

    RAW_DIR,
    PROCESSED_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    CODE_DIR,

]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 2. VitalDB API
# ============================================================

CASES_URL = (
    "https://api.vitaldb.net/cases"
)


TRACKS_URL = (
    "https://api.vitaldb.net/trks"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

CASES_PATH = (
    RAW_DIR
    / "vitaldb_cases.csv"
)


TRACKS_PATH = (
    RAW_DIR
    / "vitaldb_tracks.csv"
)


AVAILABILITY_PATH = (
    RESULT_DIR
    / "01_multimodal_track_availability.csv"
)


COMBINATION_PATH = (
    RESULT_DIR
    / "01_multimodal_combination_counts.csv"
)


ALL_CASE_FLAGS_PATH = (
    PROCESSED_DIR
    / "01_multimodal_case_track_flags.csv"
)


PRIMARY_CANDIDATE_PATH = (
    PROCESSED_DIR
    / "01_primary_multimodal_candidate_cases.csv"
)


FULL_CANDIDATE_PATH = (
    PROCESSED_DIR
    / "01_full_multimodal_candidate_cases.csv"
)


ART_CANDIDATE_PATH = (
    PROCESSED_DIR
    / "01_art_multimodal_candidate_cases.csv"
)


TRACK_SEARCH_PATH = (
    RESULT_DIR
    / "01_signal_related_track_names.csv"
)


# ============================================================
# 4. TRACKS OF INTEREST
# ============================================================

TRACKS_OF_INTEREST = [

    # --------------------------------------------------------
    # ECG
    # --------------------------------------------------------

    "SNUADC/ECG_II",


    # --------------------------------------------------------
    # PPG waveform
    # --------------------------------------------------------

    "SNUADC/PLETH",


    # --------------------------------------------------------
    # Arterial waveform
    # --------------------------------------------------------

    "SNUADC/ART",


    # --------------------------------------------------------
    # Numeric arterial pressure
    # --------------------------------------------------------

    "Solar8000/ART_MBP",

    "Solar8000/ART_SBP",

    "Solar8000/ART_DBP",


    # --------------------------------------------------------
    # Heart rate
    # --------------------------------------------------------

    "Solar8000/HR",


    # --------------------------------------------------------
    # SpO2
    # --------------------------------------------------------

    "Solar8000/PLETH_SPO2",


    # --------------------------------------------------------
    # Capnography
    # --------------------------------------------------------

    "Solar8000/ETCO2",
]


# ============================================================
# 5. DOWNLOAD FUNCTION
# ============================================================

def download_csv(
    url,
    save_path
):

    print(
        f"\n다운로드 중:\n{url}"
    )


    response = requests.get(

        url,

        timeout=120
    )


    response.raise_for_status()


    dataframe = pd.read_csv(

        BytesIO(
            response.content
        )
    )


    dataframe.columns = (

        dataframe.columns

        .str.replace(
            "\ufeff",
            "",
            regex=False
        )

        .str.strip()
    )


    dataframe.to_csv(

        save_path,

        index=False,

        encoding="utf-8-sig"
    )


    print(
        f"저장 완료: {save_path}"
    )


    print(
        f"Rows: {len(dataframe):,}"
    )


    return dataframe


# ============================================================
# 6. DOWNLOAD OR LOAD CASE METADATA
# ============================================================

print("=" * 90)

print(
    "VitalDB Multimodal Physiological Coupling Study"
)

print(
    "STEP 01 - Multimodal Track Availability Screening"
)

print("=" * 90)


try:

    cases = download_csv(

        CASES_URL,

        CASES_PATH
    )


except Exception as error:

    print(
        "\n/cases 다운로드 실패"
    )

    print(
        f"ERROR: {error}"
    )


    if CASES_PATH.exists():

        print(
            "기존 local vitaldb_cases.csv 사용"
        )


        cases = pd.read_csv(
            CASES_PATH
        )


    else:

        raise


# ============================================================
# 7. DOWNLOAD OR LOAD TRACK METADATA
# ============================================================

try:

    tracks = download_csv(

        TRACKS_URL,

        TRACKS_PATH
    )


except Exception as error:

    print(
        "\n/trks 다운로드 실패"
    )


    print(
        f"ERROR: {error}"
    )


    if TRACKS_PATH.exists():

        print(
            "기존 local vitaldb_tracks.csv 사용"
        )


        tracks = pd.read_csv(
            TRACKS_PATH
        )


    else:

        raise


# ============================================================
# 8. COLUMN CLEANING
# ============================================================

cases.columns = (

    cases.columns

    .str.replace(
        "\ufeff",
        "",
        regex=False
    )

    .str.strip()
)


tracks.columns = (

    tracks.columns

    .str.replace(
        "\ufeff",
        "",
        regex=False
    )

    .str.strip()
)


# ============================================================
# 9. REQUIRED COLUMN CHECK
# ============================================================

if "caseid" not in cases.columns:

    raise ValueError(
        "VitalDB /cases 데이터에 caseid가 없습니다."
    )


for required in [

    "caseid",
    "tname",
    "tid",

]:

    if required not in tracks.columns:

        raise ValueError(
            f"VitalDB /trks 데이터에 {required}가 없습니다."
        )


# ============================================================
# 10. CLEAN IDs
# ============================================================

cases["caseid"] = pd.to_numeric(

    cases["caseid"],

    errors="coerce"
)


tracks["caseid"] = pd.to_numeric(

    tracks["caseid"],

    errors="coerce"
)


cases = (

    cases

    .dropna(
        subset=["caseid"]
    )

    .copy()
)


tracks = (

    tracks

    .dropna(
        subset=["caseid"]
    )

    .copy()
)


cases["caseid"] = (

    cases["caseid"]

    .astype(int)
)


tracks["caseid"] = (

    tracks["caseid"]

    .astype(int)
)


tracks["tname"] = (

    tracks["tname"]

    .astype(str)

    .str.strip()
)


# ============================================================
# 11. BASIC METADATA SUMMARY
# ============================================================

print("\n")
print("=" * 90)

print(
    "VITALDB METADATA"
)

print("=" * 90)


TOTAL_CASES = (
    cases[
        "caseid"
    ]
    .nunique()
)


print(
    f"\n전체 cases: {TOTAL_CASES:,}"
)


if (
    "subjectid"
    in cases.columns
):

    print(
        f"고유 subjects: "
        f"{cases['subjectid'].nunique():,}"
    )


print(
    f"전체 track rows: "
    f"{len(tracks):,}"
)


print(
    f"고유 track names: "
    f"{tracks['tname'].nunique():,}"
)


# ============================================================
# 12. SIGNAL-RELATED TRACK NAME SEARCH
#
# 우리가 생각한 이름과 실제 VitalDB metadata를
# 다시 확인하기 위한 단계
# ============================================================

signal_search_mask = (

    tracks[
        "tname"
    ]

    .str.upper()

    .str.contains(

        "ECG|PLETH|ART|ETCO2|CO2|SPO2",

        regex=True,

        na=False
    )
)


signal_track_names = (

    tracks.loc[
        signal_search_mask,
        "tname"
    ]

    .value_counts()

    .rename_axis(
        "tname"
    )

    .reset_index(
        name="track_rows"
    )
)


signal_case_counts = (

    tracks[

        tracks[
            "tname"
        ]
        .isin(
            signal_track_names[
                "tname"
            ]
        )

    ]

    .groupby(
        "tname"
    )[
        "caseid"
    ]

    .nunique()
)


signal_track_names[
    "unique_cases"
] = (

    signal_track_names[
        "tname"
    ]

    .map(
        signal_case_counts
    )
)


signal_track_names.to_csv(

    TRACK_SEARCH_PATH,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 13. CASE SET HELPER
# ============================================================

def cases_with_track(
    track_name
):

    return set(

        tracks.loc[

            tracks[
                "tname"
            ]
            == track_name,

            "caseid"

        ]

        .astype(int)

        .unique()
    )


# ============================================================
# 14. INDIVIDUAL TRACK AVAILABILITY
# ============================================================

print("\n")
print("=" * 90)

print(
    "INDIVIDUAL TRACK AVAILABILITY"
)

print("=" * 90)


track_case_sets = {}

availability_rows = []


for track_name in (
    TRACKS_OF_INTEREST
):


    case_set = cases_with_track(
        track_name
    )


    track_case_sets[
        track_name
    ] = case_set


    n_cases = len(
        case_set
    )


    percent = (

        n_cases
        / TOTAL_CASES
        * 100
    )


    availability_rows.append({

        "track_name":
            track_name,

        "n_cases":
            n_cases,

        "percent_total_cases":
            percent
    })


    print(

        f"{track_name:<30} "

        f"{n_cases:>5,} / "

        f"{TOTAL_CASES:,} "

        f"({percent:5.1f}%)"
    )


availability_df = pd.DataFrame(
    availability_rows
)


availability_df.to_csv(

    AVAILABILITY_PATH,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 15. SIGNAL SETS
# ============================================================

ECG = (
    track_case_sets[
        "SNUADC/ECG_II"
    ]
)


PLETH = (
    track_case_sets[
        "SNUADC/PLETH"
    ]
)


ART = (
    track_case_sets[
        "SNUADC/ART"
    ]
)


ART_MBP = (
    track_case_sets[
        "Solar8000/ART_MBP"
    ]
)


ART_SBP = (
    track_case_sets[
        "Solar8000/ART_SBP"
    ]
)


ART_DBP = (
    track_case_sets[
        "Solar8000/ART_DBP"
    ]
)


HR = (
    track_case_sets[
        "Solar8000/HR"
    ]
)


SPO2 = (
    track_case_sets[
        "Solar8000/PLETH_SPO2"
    ]
)


ETCO2 = (
    track_case_sets[
        "Solar8000/ETCO2"
    ]
)


# ============================================================
# 16. ADULT CASES
# ============================================================

all_case_ids = set(

    cases[
        "caseid"
    ]
    .astype(int)
)


if (
    "age"
    in cases.columns
):

    age = pd.to_numeric(

        cases["age"],

        errors="coerce"
    )


    adult_cases = set(

        cases.loc[

            age >= 18,

            "caseid"

        ]

        .astype(int)
    )


else:

    print(
        "\nWARNING: age column 없음."
    )

    print(
        "전체 case를 adult 후보로 사용합니다."
    )


    adult_cases = (
        all_case_ids.copy()
    )


print("\n")
print("=" * 90)

print(
    "ADULT COHORT"
)

print("=" * 90)


print(
    f"\n성인 case 수: "
    f"{len(adult_cases):,}"
)


# ============================================================
# 17. BASIC MULTIMODAL COMBINATIONS
# ============================================================

combination_sets = {


    # ========================================================
    # TWO SIGNAL
    # ========================================================

    "ECG + PLETH":

        ECG
        & PLETH,


    "ECG + ART":

        ECG
        & ART,


    "PLETH + ART":

        PLETH
        & ART,


    "ART + ETCO2":

        ART
        & ETCO2,


    # ========================================================
    # THREE SIGNAL
    # ========================================================

    "ECG + PLETH + ETCO2":

        ECG
        & PLETH
        & ETCO2,


    "ECG + PLETH + ART":

        ECG
        & PLETH
        & ART,


    "ECG + ART + ETCO2":

        ECG
        & ART
        & ETCO2,


    "PLETH + ART + ETCO2":

        PLETH
        & ART
        & ETCO2,


    # ========================================================
    # FOUR SIGNAL
    # ========================================================

    "ECG + PLETH + ART + ETCO2":

        ECG
        & PLETH
        & ART
        & ETCO2,


    # ========================================================
    # Add numeric MAP
    # ========================================================

    "ECG + PLETH + ART + ART_MBP":

        ECG
        & PLETH
        & ART
        & ART_MBP,


    "ECG + PLETH + ART + ART_MBP + ETCO2":

        ECG
        & PLETH
        & ART
        & ART_MBP
        & ETCO2,


    # ========================================================
    # Numeric multimodal
    # ========================================================

    "HR + ART_MBP + SPO2 + ETCO2":

        HR
        & ART_MBP
        & SPO2
        & ETCO2,


    # ========================================================
    # Adult combinations
    # ========================================================

    "Adult + ECG + PLETH":

        adult_cases
        & ECG
        & PLETH,


    "Adult + ECG + PLETH + ETCO2":

        adult_cases
        & ECG
        & PLETH
        & ETCO2,


    "Adult + ECG + PLETH + ART":

        adult_cases
        & ECG
        & PLETH
        & ART,


    "Adult + ECG + ART + ETCO2":

        adult_cases
        & ECG
        & ART
        & ETCO2,


    "Adult + PLETH + ART + ETCO2":

        adult_cases
        & PLETH
        & ART
        & ETCO2,


    "Adult + ECG + PLETH + ART + ETCO2":

        adult_cases
        & ECG
        & PLETH
        & ART
        & ETCO2,


    "Adult + ECG + PLETH + ART + ART_MBP":

        adult_cases
        & ECG
        & PLETH
        & ART
        & ART_MBP,


    "Adult + ECG + PLETH + ART + ART_MBP + ETCO2":

        adult_cases
        & ECG
        & PLETH
        & ART
        & ART_MBP
        & ETCO2,


    "Adult + HR + ART_MBP + SPO2 + ETCO2":

        adult_cases
        & HR
        & ART_MBP
        & SPO2
        & ETCO2,
}


# ============================================================
# 18. PRINT COMBINATION COUNTS
# ============================================================

print("\n")
print("=" * 90)

print(
    "MULTIMODAL TRACK COMBINATION COUNTS"
)

print("=" * 90)


combination_rows = []


for name, case_set in (
    combination_sets.items()
):


    n_cases = len(
        case_set
    )


    percent = (

        n_cases
        / TOTAL_CASES
        * 100
    )


    # --------------------------------------------------------
    # Subject count
    # --------------------------------------------------------

    if (
        "subjectid"
        in cases.columns
    ):

        n_subjects = (

            cases.loc[

                cases[
                    "caseid"
                ]
                .isin(
                    case_set
                ),

                "subjectid"

            ]

            .nunique()
        )


    else:

        n_subjects = np.nan


    combination_rows.append({

        "combination":
            name,

        "n_cases":
            n_cases,

        "n_subjects":
            n_subjects,

        "percent_total_cases":
            percent,
    })


    print(

        f"{name:<58} "

        f"{n_cases:>5,} cases | "

        f"{n_subjects:>5,} subjects"
    )


combination_df = pd.DataFrame(
    combination_rows
)


combination_df.to_csv(

    COMBINATION_PATH,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 19. CASE-LEVEL FLAGS
# ============================================================

candidate = (
    cases.copy()
)


TRACK_FLAGS = {

    "has_ecg_ii":
        ECG,

    "has_pleth_wave":
        PLETH,

    "has_art_wave":
        ART,

    "has_art_mbp":
        ART_MBP,

    "has_art_sbp":
        ART_SBP,

    "has_art_dbp":
        ART_DBP,

    "has_hr":
        HR,

    "has_spo2":
        SPO2,

    "has_etco2":
        ETCO2,
}


for flag_name, case_set in (
    TRACK_FLAGS.items()
):

    candidate[
        flag_name
    ] = (

        candidate[
            "caseid"
        ]

        .isin(
            case_set
        )

        .astype(int)
    )


# ============================================================
# 20. ADULT FLAG
# ============================================================

if (
    "age"
    in candidate.columns
):

    candidate[
        "adult"
    ] = (

        pd.to_numeric(

            candidate[
                "age"
            ],

            errors="coerce"
        )

        >= 18

    ).astype(int)


else:

    candidate[
        "adult"
    ] = 1


# ============================================================
# 21. CANDIDATE DEFINITIONS
# ============================================================

# ------------------------------------------------------------
# Candidate 1
#
# ECG + PPG + ETCO2
#
# ART 없이 사용할 수 있는 큰 cohort
# ------------------------------------------------------------

candidate[
    "candidate_ecg_pleth_etco2"
] = (

    (
        candidate[
            "adult"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_ecg_ii"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_pleth_wave"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_etco2"
        ]
        == 1
    )

).astype(int)


# ------------------------------------------------------------
# Candidate 2
#
# ECG + PPG + ART + ETCO2
#
# Signal coupling 연구의 핵심 후보
# ------------------------------------------------------------

candidate[
    "candidate_ecg_pleth_art_etco2"
] = (

    (
        candidate[
            "adult"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_ecg_ii"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_pleth_wave"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_art_wave"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_etco2"
        ]
        == 1
    )

).astype(int)


# ------------------------------------------------------------
# Candidate 3
#
# ECG + PPG + ART + numeric MAP + ETCO2
#
# 가장 완전한 cohort
# ------------------------------------------------------------

candidate[
    "candidate_full_multimodal"
] = (

    (
        candidate[
            "candidate_ecg_pleth_art_etco2"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_art_mbp"
        ]
        == 1
    )

).astype(int)


# ------------------------------------------------------------
# Candidate 4
#
# Numeric physiology baseline comparison
# ------------------------------------------------------------

candidate[
    "candidate_numeric_multimodal"
] = (

    (
        candidate[
            "adult"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_hr"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_art_mbp"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_spo2"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_etco2"
        ]
        == 1
    )

).astype(int)


# ============================================================
# 22. SAVE ALL CASE FLAGS
# ============================================================

candidate.to_csv(

    ALL_CASE_FLAGS_PATH,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 23. PRIMARY MULTIMODAL CANDIDATE
#
# Adult + ECG + PLETH + ART + ETCO2
# ============================================================

primary_candidate = candidate[

    candidate[
        "candidate_ecg_pleth_art_etco2"
    ]
    == 1

].copy()


primary_candidate.to_csv(

    PRIMARY_CANDIDATE_PATH,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 24. FULL MULTIMODAL CANDIDATE
#
# Adult + ECG + PLETH + ART + MAP + ETCO2
# ============================================================

full_candidate = candidate[

    candidate[
        "candidate_full_multimodal"
    ]
    == 1

].copy()


full_candidate.to_csv(

    FULL_CANDIDATE_PATH,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 25. ART MULTIMODAL CANDIDATE
#
# Adult + ECG + PLETH + ART
# ============================================================

art_candidate = candidate[

    (
        candidate[
            "adult"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_ecg_ii"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_pleth_wave"
        ]
        == 1
    )

    &

    (
        candidate[
            "has_art_wave"
        ]
        == 1
    )

].copy()


art_candidate.to_csv(

    ART_CANDIDATE_PATH,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 26. MAIN SCREENING SUMMARY
# ============================================================

print("\n")
print("=" * 90)

print(
    "MAIN MULTIMODAL SCREENING"
)

print("=" * 90)


def get_subject_count(
    dataframe
):

    if (
        "subjectid"
        in dataframe.columns
    ):

        return dataframe[
            "subjectid"
        ].nunique()

    return np.nan


# ------------------------------------------------------------
# Counts
# ------------------------------------------------------------

ecg_pleth_etco2_n = int(

    candidate[
        "candidate_ecg_pleth_etco2"
    ]
    .sum()
)


ecg_pleth_art_n = (
    len(
        art_candidate
    )
)


primary_n = (
    len(
        primary_candidate
    )
)


full_n = (
    len(
        full_candidate
    )
)


numeric_n = int(

    candidate[
        "candidate_numeric_multimodal"
    ]
    .sum()
)


print(
    "\nAdult + ECG + PLETH + ETCO2"
)

print(
    f"Cases    : "
    f"{ecg_pleth_etco2_n:,}"
)


temp = candidate[

    candidate[
        "candidate_ecg_pleth_etco2"
    ]
    == 1

]


print(
    f"Subjects : "
    f"{get_subject_count(temp):,}"
)


print(
    "\nAdult + ECG + PLETH + ART"
)

print(
    f"Cases    : "
    f"{ecg_pleth_art_n:,}"
)

print(
    f"Subjects : "
    f"{get_subject_count(art_candidate):,}"
)


print(
    "\nAdult + ECG + PLETH + ART + ETCO2"
)

print(
    f"Cases    : "
    f"{primary_n:,}"
)

print(
    f"Subjects : "
    f"{get_subject_count(primary_candidate):,}"
)


print(
    "\nAdult + ECG + PLETH + ART + ART_MBP + ETCO2"
)

print(
    f"Cases    : "
    f"{full_n:,}"
)

print(
    f"Subjects : "
    f"{get_subject_count(full_candidate):,}"
)


numeric_candidate = candidate[

    candidate[
        "candidate_numeric_multimodal"
    ]
    == 1

]


print(
    "\nAdult + HR + ART_MBP + SpO2 + ETCO2"
)

print(
    f"Cases    : "
    f"{numeric_n:,}"
)

print(
    f"Subjects : "
    f"{get_subject_count(numeric_candidate):,}"
)


# ============================================================
# 27. CLINICAL OUTCOME AVAILABILITY
#
# candidate cohort에서 사용할 수 있는 outcome도 미리 확인
# ============================================================

print("\n")
print("=" * 90)

print(
    "OUTCOME AVAILABILITY IN PRIMARY COHORT"
)

print("=" * 90)


OUTCOME_COLUMNS = [

    "icu_days",

    "los_postop",

    "death_inhosp",

    "death_30d",

    "death_1yr",
]


for col in OUTCOME_COLUMNS:

    if (
        col
        in primary_candidate.columns
    ):

        values = primary_candidate[
            col
        ]


        nonmissing = int(
            values.notna().sum()
        )


        print(

            f"{col:<20} "

            f"nonmissing="
            f"{nonmissing:,}"
        )


# ============================================================
# 28. ICU UTILIZATION DISTRIBUTION
# ============================================================

if (
    "icu_days"
    in primary_candidate.columns
):


    icu_days = pd.to_numeric(

        primary_candidate[
            "icu_days"
        ],

        errors="coerce"
    )


    valid_icu = (
        icu_days.notna()
    )


    icu_positive = int(

        (
            icu_days[
                valid_icu
            ]
            > 0
        )
        .sum()
    )


    icu_total = int(
        valid_icu.sum()
    )


    print("\n")
    print(
        "Primary cohort postoperative ICU utilization"
    )


    print(
        f"N = {icu_total:,}"
    )


    if (
        icu_total > 0
    ):

        print(

            f"ICU > 0 days = "
            f"{icu_positive:,} "
            f"({icu_positive / icu_total * 100:.1f}%)"
        )


# ============================================================
# 29. POSTOPERATIVE LOS DISTRIBUTION
# ============================================================

los_column = None


for possible_col in [

    "los_postop",

    "los_postop_days",

]:

    if (
        possible_col
        in primary_candidate.columns
    ):

        los_column = (
            possible_col
        )

        break


if (
    los_column
    is not None
):


    los = pd.to_numeric(

        primary_candidate[
            los_column
        ],

        errors="coerce"
    )


    los = (
        los
        .dropna()
    )


    if len(
        los
    ) > 0:

        print("\n")

        print(
            "Primary cohort postoperative LOS"
        )


        print(
            f"N = {len(los):,}"
        )


        print(
            f"Median = "
            f"{los.median():.3f}"
        )


        print(
            f"Q3 = "
            f"{los.quantile(0.75):.3f}"
        )


# ============================================================
# 30. FEASIBILITY DECISION
#
# Primary = ECG + PPG + ART + ETCO2
# ============================================================

print("\n")
print("=" * 90)

print(
    "FEASIBILITY SCREENING"
)

print("=" * 90)


if (
    primary_n
    >= 2500
):

    decision = (
        "VERY GOOD"
    )


    explanation = (
        "충분히 큰 multimodal waveform cohort. "
        "연구 진행을 강하게 권장."
    )


elif (
    primary_n
    >= 1500
):

    decision = (
        "GOOD"
    )


    explanation = (
        "BSPC 수준 multimodal signal 분석을 "
        "진행하기에 충분한 후보 표본."
    )


elif (
    primary_n
    >= 1000
):

    decision = (
        "ACCEPTABLE"
    )


    explanation = (
        "진행 가능하나 STEP 02 신호 품질검사 후 "
        "최종 표본 수를 다시 확인해야 함."
    )


elif (
    primary_n
    >= 500
):

    decision = (
        "REVIEW"
    )


    explanation = (
        "가능성은 있으나 signal-quality exclusion 후 "
        "표본 감소 위험이 큼."
    )


else:

    decision = (
        "LOW"
    )


    explanation = (
        "multimodal waveform 연구에는 표본이 부족할 가능성이 높음."
    )


print(
    f"\nPrimary candidate:"
)

print(
    "Adult + ECG + PLETH + ART + ETCO2"
)


print(
    f"\nCases: "
    f"{primary_n:,}"
)


print(
    f"Subjects: "
    f"{get_subject_count(primary_candidate):,}"
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
# 31. SECONDARY DECISION
#
# ETCO2 때문에 표본이 많이 감소하는지도 확인
# ============================================================

print("\n")
print("=" * 90)

print(
    "ETCO2 SAMPLE LOSS CHECK"
)

print("=" * 90)


if (
    ecg_pleth_art_n
    > 0
):


    retained_fraction = (

        primary_n
        /
        ecg_pleth_art_n
    )


    lost_n = (

        ecg_pleth_art_n
        -
        primary_n
    )


    print(

        f"\nECG + PLETH + ART cases: "
        f"{ecg_pleth_art_n:,}"
    )


    print(

        f"+ ETCO2 cases: "
        f"{primary_n:,}"
    )


    print(

        f"ETCO2 추가로 제외되는 case: "
        f"{lost_n:,}"
    )


    print(

        f"Retained: "
        f"{retained_fraction * 100:.1f}%"
    )


    if (
        retained_fraction
        < 0.70
    ):

        print(
            "\n주의: ETCO2 추가 시 표본 감소가 큼."
        )


        print(
            "Primary analysis를 ECG + PLETH + ART로 두고 "
            "ETCO2를 secondary analysis로 사용하는 방안 검토."
        )


    else:

        print(
            "\nETCO2 추가에 따른 표본 감소는 허용 가능한 수준."
        )


# ============================================================
# 32. SAVE FINAL SCREENING SUMMARY
# ============================================================

screening_summary = pd.DataFrame({

    "cohort": [

        "Adult_ECG_PLETH_ETCO2",

        "Adult_ECG_PLETH_ART",

        "Adult_ECG_PLETH_ART_ETCO2",

        "Adult_ECG_PLETH_ART_ARTMBP_ETCO2",

        "Adult_HR_ARTMBP_SPO2_ETCO2",
    ],

    "n_cases": [

        ecg_pleth_etco2_n,

        ecg_pleth_art_n,

        primary_n,

        full_n,

        numeric_n,
    ],

    "n_subjects": [

        get_subject_count(
            temp
        ),

        get_subject_count(
            art_candidate
        ),

        get_subject_count(
            primary_candidate
        ),

        get_subject_count(
            full_candidate
        ),

        get_subject_count(
            numeric_candidate
        ),
    ],
})


SCREENING_SUMMARY_PATH = (

    RESULT_DIR
    / "01_multimodal_screening_summary.csv"
)


screening_summary.to_csv(

    SCREENING_SUMMARY_PATH,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 33. FINISH
# ============================================================

print("\n")
print("=" * 90)

print(
    "STEP 01 완료"
)

print("=" * 90)


print(
    "\n생성 파일:"
)


print(
    CASES_PATH
)


print(
    TRACKS_PATH
)


print(
    AVAILABILITY_PATH
)


print(
    COMBINATION_PATH
)


print(
    TRACK_SEARCH_PATH
)


print(
    ALL_CASE_FLAGS_PATH
)


print(
    PRIMARY_CANDIDATE_PATH
)


print(
    FULL_CANDIDATE_PATH
)


print(
    ART_CANDIDATE_PATH
)


print(
    SCREENING_SUMMARY_PATH
)


print("\n")
print("=" * 90)

print(
    "NEXT STEP"
)

print("=" * 90)


print(
    """
STEP 02에서는 Primary candidate를 대상으로 실제 waveform의
유효성을 검사합니다.

검사 예정:

1. ECG waveform 실제 유효 duration
2. PLETH waveform 실제 유효 duration
3. ART waveform 실제 유효 duration
4. ETCO2 numeric 실제 유효 duration
5. 수술 중 공통 overlap duration
6. artifact / missing signal 비율
7. 최종 analyzable cohort 수

아직 coupling feature 추출은 하지 않습니다.

먼저 충분한 신호 품질을 가진 환자가 실제로 몇 명인지 확인한 뒤
연구 주제를 최종 확정합니다.
"""
)