# ============================================================
# VitalDB Multimodal Physiological Coupling Study
# STEP 03
#
# Waveform Coupling Pilot
#
# ------------------------------------------------------------
# 목적
#
# STEP 02 quality-eligible cohort 중 대표 5 case를 선택하여
#
# 1. ECG waveform 로딩 확인
# 2. PLETH waveform 로딩 확인
# 3. ART waveform 로딩 확인
# 4. ECG R-peak 검출
# 5. PLETH systolic peak 검출
# 6. ART systolic peak 검출
# 7. ECG -> ART timing 계산
# 8. ECG -> PLETH timing 계산
# 9. ART -> PLETH timing 계산
# 10. ART-PLETH cross-correlation / lag 계산
#
# ------------------------------------------------------------
# IMPORTANT
#
# 원본 waveform은 500 Hz이지만 pilot에서는 100 Hz로
# resampling하여 처리량을 줄임.
#
# 각 case에서 수술 중 가장 quality가 좋은 10분 구간을 자동 선택.
#
# 아직 outcome prediction은 수행하지 않음.
# ============================================================


from pathlib import Path
import warnings
import traceback

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.signal import (
    butter,
    sosfiltfilt,
    find_peaks,
    correlate,
    correlation_lags,
)

import vitaldb


warnings.filterwarnings("ignore")


# ============================================================
# 1. PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
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
    / "03_waveform_pilot"
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


INPUT_PATH = (
    PROCESSED_DIR
    / "02_numeric_quality_eligible_cases.csv"
)


# ============================================================
# 2. OUTPUT
# ============================================================

SELECTED_CASES_PATH = (
    RESULT_DIR
    / "03_pilot_selected_cases.csv"
)

PILOT_RESULT_PATH = (
    RESULT_DIR
    / "03_waveform_coupling_pilot_results.csv"
)

PILOT_ERROR_PATH = (
    RESULT_DIR
    / "03_waveform_coupling_pilot_errors.csv"
)


# ============================================================
# 3. SETTINGS
# ============================================================

# ------------------------------------------------------------
# Pilot cases
# ------------------------------------------------------------

N_PILOT_CASES = 5


# ------------------------------------------------------------
# Resampled waveform frequency
# ------------------------------------------------------------

FS = 100

INTERVAL = (
    1.0
    / FS
)


# ------------------------------------------------------------
# Pilot window
# ------------------------------------------------------------

PILOT_WINDOW_SEC = (
    10
    * 60
)


# ------------------------------------------------------------
# Signal tracks
# ------------------------------------------------------------

TRACK_NAMES = [

    "SNUADC/ECG_II",

    "SNUADC/PLETH",

    "SNUADC/ART",
]


# ------------------------------------------------------------
# Minimum usable coverage
# ------------------------------------------------------------

MIN_WINDOW_COVERAGE = 0.80


# ============================================================
# 4. LOAD ELIGIBLE COHORT
# ============================================================

print("=" * 90)

print(
    "VitalDB Multimodal Physiological Coupling Study"
)

print(
    "STEP 03 - Waveform Coupling Pilot"
)

print("=" * 90)


if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"입력 파일 없음:\n{INPUT_PATH}"
    )


df = pd.read_csv(
    INPUT_PATH
)


df.columns = (
    df.columns
    .str.replace(
        "\ufeff",
        "",
        regex=False
    )
    .str.strip()
)


df["caseid"] = pd.to_numeric(
    df["caseid"],
    errors="coerce"
)


df = (
    df
    .dropna(
        subset=[
            "caseid"
        ]
    )
    .copy()
)


df["caseid"] = (
    df["caseid"]
    .astype(int)
)


print(
    f"\nEligible cases: "
    f"{len(df):,}"
)


if "subjectid" in df.columns:

    print(
        f"Eligible subjects: "
        f"{df['subjectid'].nunique():,}"
    )


# ============================================================
# 5. REPRESENTATIVE PILOT CASE SELECTION
#
# common_valid_minutes의 여러 분위수에 가까운 case 선택
#
# outcome을 이용해 선택하지 않음
# ============================================================

if (
    "common_valid_minutes"
    not in df.columns
):

    raise ValueError(
        "common_valid_minutes column 없음"
    )


df[
    "common_valid_minutes"
] = pd.to_numeric(

    df[
        "common_valid_minutes"
    ],

    errors="coerce"
)


selection_pool = (

    df
    .dropna(
        subset=[
            "common_valid_minutes"
        ]
    )
    .copy()
)


# ------------------------------------------------------------
# 10%, 30%, 50%, 70%, 90%
# ------------------------------------------------------------

quantiles = np.linspace(

    0.10,

    0.90,

    N_PILOT_CASES
)


selected_indices = []


for q in quantiles:

    target = (
        selection_pool[
            "common_valid_minutes"
        ]
        .quantile(
            q
        )
    )


    remaining = (
        selection_pool
        .drop(
            index=selected_indices,
            errors="ignore"
        )
    )


    distance = (

        remaining[
            "common_valid_minutes"
        ]
        -
        target

    ).abs()


    selected_index = (
        distance
        .idxmin()
    )


    selected_indices.append(
        selected_index
    )


pilot_cases = (

    selection_pool
    .loc[
        selected_indices
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


pilot_cases[
    "pilot_order"
] = np.arange(
    1,
    len(pilot_cases) + 1
)


pilot_cases.to_csv(

    SELECTED_CASES_PATH,

    index=False,

    encoding="utf-8-sig"
)


print("\n")
print("=" * 90)

print(
    "SELECTED PILOT CASES"
)

print("=" * 90)


for _, row in (
    pilot_cases.iterrows()
):

    print(

        f"Pilot {int(row['pilot_order'])} | "

        f"caseid="
        f"{int(row['caseid'])} | "

        f"common_valid="
        f"{row['common_valid_minutes']:.1f} min"
    )


# ============================================================
# 6. FILTER
# ============================================================

def bandpass_filter(
    signal,
    fs,
    lowcut,
    highcut,
    order=3
):

    signal = np.asarray(
        signal,
        dtype=float
    )


    nyquist = (
        fs
        / 2.0
    )


    low = (
        lowcut
        / nyquist
    )


    high = (
        highcut
        / nyquist
    )


    sos = butter(

        order,

        [
            low,
            high
        ],

        btype="bandpass",

        output="sos"
    )


    filtered = sosfiltfilt(

        sos,

        signal
    )


    return filtered


# ============================================================
# 7. INTERPOLATE WAVEFORM
# ============================================================

def interpolate_waveform(
    signal
):

    series = pd.Series(
        signal,
        dtype=float
    )


    original_missing = (
        series
        .isna()
        .mean()
    )


    # --------------------------------------------------------
    # Pilot window는 coverage가 높은 구간만 사용하므로
    # filtering을 위해 내부 결측 linear interpolation
    # --------------------------------------------------------

    series = (
        series
        .interpolate(
            method="linear",
            limit_direction="both"
        )
    )


    return (
        series.to_numpy(),
        original_missing
    )


# ============================================================
# 8. AUTO POLARITY
#
# ECG lead polarity가 뒤집힌 경우 자동 대응
# ============================================================

def auto_polarity(
    signal
):

    x = np.asarray(
        signal
    )


    upper = np.nanpercentile(
        x,
        99
    )


    lower = abs(
        np.nanpercentile(
            x,
            1
        )
    )


    if lower > upper:

        return (
            -x,
            -1
        )


    return (
        x,
        1
    )


# ============================================================
# 9. ECG R-PEAK DETECTION
# ============================================================

def detect_r_peaks(
    ecg,
    fs
):

    # --------------------------------------------------------
    # QRS 중심 band
    # --------------------------------------------------------

    filtered = bandpass_filter(

        ecg,

        fs,

        5.0,

        25.0,

        order=3
    )


    detection_signal, polarity = (
        auto_polarity(
            filtered
        )
    )


    iqr = (

        np.percentile(
            detection_signal,
            75
        )

        -

        np.percentile(
            detection_signal,
            25
        )
    )


    prominence = max(

        iqr * 1.5,

        np.std(
            detection_signal
        )
        * 0.35,

        1e-8
    )


    peaks, properties = (
        find_peaks(

            detection_signal,

            distance=int(
                0.30
                * fs
            ),

            prominence=
                prominence
        )
    )


    return (
        peaks,
        filtered,
        polarity
    )


# ============================================================
# 10. PULSE PEAK DETECTION
#
# PLETH / ART
# ============================================================

def detect_pulse_peaks(
    signal,
    fs,
    signal_type
):


    if (
        signal_type
        == "PLETH"
    ):

        highcut = 8.0


    else:

        highcut = 12.0


    filtered = bandpass_filter(

        signal,

        fs,

        0.5,

        highcut,

        order=3
    )


    detection_signal, polarity = (
        auto_polarity(
            filtered
        )
    )


    iqr = (

        np.percentile(
            detection_signal,
            75
        )

        -

        np.percentile(
            detection_signal,
            25
        )
    )


    prominence = max(

        iqr * 0.30,

        np.std(
            detection_signal
        )
        * 0.15,

        1e-8
    )


    peaks, properties = (
        find_peaks(

            detection_signal,

            distance=int(
                0.35
                * fs
            ),

            prominence=
                prominence
        )
    )


    return (
        peaks,
        filtered,
        polarity
    )


# ============================================================
# 11. HEART RATE FROM ECG
# ============================================================

def calculate_ecg_hr(
    r_peaks,
    fs
):

    if len(
        r_peaks
    ) < 2:

        return np.nan


    rr = (

        np.diff(
            r_peaks
        )

        /
        fs
    )


    # plausible RR
    rr = rr[

        (
            rr >= 0.30
        )

        &

        (
            rr <= 2.0
        )
    ]


    if len(
        rr
    ) == 0:

        return np.nan


    return float(

        60.0

        /
        np.median(
            rr
        )
    )


# ============================================================
# 12. PAIR SOURCE EVENTS TO NEXT TARGET EVENT
# ============================================================

def calculate_event_lags(
    source_peaks,
    target_peaks,
    fs,
    min_lag_sec,
    max_lag_sec
):


    source_times = (

        np.asarray(
            source_peaks
        )

        /
        fs
    )


    target_times = (

        np.asarray(
            target_peaks
        )

        /
        fs
    )


    lag_values = []


    target_index = 0


    for source_time in (
        source_times
    ):


        while (

            target_index
            < len(
                target_times
            )

            and

            target_times[
                target_index
            ]
            <
            source_time
            +
            min_lag_sec

        ):

            target_index += 1


        if (
            target_index
            >= len(
                target_times
            )
        ):

            break


        lag = (

            target_times[
                target_index
            ]

            -
            source_time
        )


        if (

            lag
            >= min_lag_sec

            and

            lag
            <= max_lag_sec

        ):

            lag_values.append(
                lag
            )


    return np.asarray(
        lag_values
    )


# ============================================================
# 13. DESCRIBE LAGS
# ============================================================

def describe_lags(
    lags
):


    if len(
        lags
    ) == 0:

        return {

            "n":
                0,

            "median":
                np.nan,

            "q1":
                np.nan,

            "q3":
                np.nan,

            "mean":
                np.nan,

            "std":
                np.nan,
        }


    return {

        "n":
            len(
                lags
            ),

        "median":
            float(
                np.median(
                    lags
                )
            ),

        "q1":
            float(
                np.percentile(
                    lags,
                    25
                )
            ),

        "q3":
            float(
                np.percentile(
                    lags,
                    75
                )
            ),

        "mean":
            float(
                np.mean(
                    lags
                )
            ),

        "std":
            float(
                np.std(
                    lags
                )
            ),
    }


# ============================================================
# 14. ART-PLETH CROSS CORRELATION
# ============================================================

def calculate_cross_correlation(
    art,
    pleth,
    fs,
    max_lag_sec=1.0
):


    art = np.asarray(
        art,
        dtype=float
    )


    pleth = np.asarray(
        pleth,
        dtype=float
    )


    # --------------------------------------------------------
    # z normalization
    # --------------------------------------------------------

    art_std = np.std(
        art
    )


    pleth_std = np.std(
        pleth
    )


    if (
        art_std == 0
        or
        pleth_std == 0
    ):

        return (
            np.nan,
            np.nan
        )


    art_z = (

        art
        -
        np.mean(
            art
        )

    ) / art_std


    pleth_z = (

        pleth
        -
        np.mean(
            pleth
        )

    ) / pleth_std


    corr = correlate(

        pleth_z,

        art_z,

        mode="full",

        method="fft"
    )


    lags = correlation_lags(

        len(
            pleth_z
        ),

        len(
            art_z
        ),

        mode="full"
    )


    max_lag_samples = int(

        max_lag_sec
        * fs
    )


    mask = (

        np.abs(
            lags
        )
        <= max_lag_samples

    )


    corr = (
        corr[
            mask
        ]
    )


    lags = (
        lags[
            mask
        ]
    )


    corr = (

        corr

        /
        len(
            art_z
        )
    )


    best_index = int(

        np.argmax(
            corr
        )
    )


    best_corr = float(

        corr[
            best_index
        ]
    )


    best_lag_sec = float(

        lags[
            best_index
        ]

        /
        fs
    )


    return (
        best_corr,
        best_lag_sec
    )


# ============================================================
# 15. BEST 10-MIN WINDOW
# ============================================================

def select_best_window(
    waveform,
    fs,
    window_sec
):


    common_valid = (

        np.isfinite(
            waveform[:, 0]
        )

        &

        np.isfinite(
            waveform[:, 1]
        )

        &

        np.isfinite(
            waveform[:, 2]
        )
    )


    total_samples = len(
        common_valid
    )


    total_seconds = int(

        total_samples
        // fs
    )


    if (
        total_seconds
        < window_sec
    ):

        return None


    usable_samples = (

        total_seconds
        * fs
    )


    second_coverage = (

        common_valid[
            :usable_samples
        ]

        .reshape(
            total_seconds,
            fs
        )

        .mean(
            axis=1
        )
    )


    # --------------------------------------------------------
    # moving mean
    # --------------------------------------------------------

    kernel = np.ones(
        window_sec,
        dtype=float
    )


    moving_sum = np.convolve(

        second_coverage,

        kernel,

        mode="valid"
    )


    moving_coverage = (

        moving_sum
        /
        window_sec
    )


    best_start_sec = int(

        np.argmax(
            moving_coverage
        )
    )


    best_coverage = float(

        moving_coverage[
            best_start_sec
        ]
    )


    start_sample = (

        best_start_sec
        * fs
    )


    end_sample = (

        start_sample

        +

        window_sec
        * fs
    )


    return {

        "start_sec":
            best_start_sec,

        "end_sec":
            best_start_sec
            +
            window_sec,

        "start_sample":
            start_sample,

        "end_sample":
            end_sample,

        "coverage":
            best_coverage,
    }


# ============================================================
# 16. LOAD ONE CASE
# ============================================================

def load_case_waveform(
    caseid,
    opstart,
    opend
):


    print(
        "  waveform 다운로드 중..."
    )


    values = vitaldb.load_case(

        caseid,

        TRACK_NAMES,

        interval=
            INTERVAL
    )


    values = np.asarray(
        values,
        dtype=float
    )


    if (
        values.ndim != 2

        or

        values.shape[1] != 3
    ):

        raise ValueError(

            f"Unexpected waveform shape: "
            f"{values.shape}"
        )


    # --------------------------------------------------------
    # operation window crop
    #
    # VitalDB case time axis 기준
    # --------------------------------------------------------

    start_index = max(

        0,

        int(
            np.floor(
                opstart
                * FS
            )
        )
    )


    end_index = min(

        len(
            values
        ),

        int(
            np.ceil(
                opend
                * FS
            )
        )
    )


    if (
        end_index
        <= start_index
    ):

        raise ValueError(
            "Operation waveform crop 실패"
        )


    operation_values = values[

        start_index:
        end_index,

        :

    ].copy()


    return operation_values


# ============================================================
# 17. PILOT RESULTS
# ============================================================

result_rows = []

error_rows = []


# ============================================================
# 18. RUN PILOT
# ============================================================

print("\n")
print("=" * 90)

print(
    "WAVEFORM PILOT START"
)

print("=" * 90)


for pilot_index, (
    _,
    row

) in enumerate(

    pilot_cases.iterrows(),

    start=1

):


    caseid = int(
        row[
            "caseid"
        ]
    )


    print("\n")
    print("-" * 90)

    print(
        f"PILOT {pilot_index}/{len(pilot_cases)} "
        f"| caseid={caseid}"
    )

    print("-" * 90)


    try:

        # ====================================================
        # OPERATION TIME
        # ====================================================

        opstart = pd.to_numeric(

            row.get(
                "opstart_sec",
                row.get(
                    "opstart",
                    np.nan
                )
            ),

            errors="coerce"
        )


        opend = pd.to_numeric(

            row.get(
                "opend_sec",
                row.get(
                    "opend",
                    np.nan
                )
            ),

            errors="coerce"
        )


        if (

            pd.isna(
                opstart
            )

            or

            pd.isna(
                opend
            )

            or

            opend
            <= opstart

        ):

            raise ValueError(
                "유효한 opstart/opend 없음"
            )


        print(
            f"Operation window: "
            f"{opstart/60:.1f} - "
            f"{opend/60:.1f} min"
        )


        # ====================================================
        # LOAD WAVEFORM
        # ====================================================

        waveform = load_case_waveform(

            caseid,

            float(
                opstart
            ),

            float(
                opend
            )
        )


        operation_min = (

            len(
                waveform
            )

            /
            FS

            /
            60
        )


        print(
            f"Loaded waveform: "
            f"{operation_min:.1f} min"
        )


        print(
            f"Shape: "
            f"{waveform.shape}"
        )


        # ====================================================
        # SELECT BEST 10 MIN
        # ====================================================

        window_info = select_best_window(

            waveform,

            FS,

            PILOT_WINDOW_SEC
        )


        if (
            window_info
            is None
        ):

            raise ValueError(
                "10분 waveform 확보 불가"
            )


        print(

            f"Best 10-min waveform coverage: "
            f"{window_info['coverage']*100:.1f}%"
        )


        if (
            window_info[
                "coverage"
            ]
            <
            MIN_WINDOW_COVERAGE
        ):

            raise ValueError(

                "Best waveform coverage "
                "<80%"
            )


        pilot_wave = waveform[

            window_info[
                "start_sample"
            ]:

            window_info[
                "end_sample"
            ],

            :

        ].copy()


        # ----------------------------------------------------
        # Tracks
        # ----------------------------------------------------

        ecg_raw = (
            pilot_wave[
                :,
                0
            ]
        )


        pleth_raw = (
            pilot_wave[
                :,
                1
            ]
        )


        art_raw = (
            pilot_wave[
                :,
                2
            ]
        )


        # ====================================================
        # MISSINGNESS
        # ====================================================

        ecg_missing = float(

            np.mean(
                ~np.isfinite(
                    ecg_raw
                )
            )
        )


        pleth_missing = float(

            np.mean(
                ~np.isfinite(
                    pleth_raw
                )
            )
        )


        art_missing = float(

            np.mean(
                ~np.isfinite(
                    art_raw
                )
            )
        )


        print(

            f"Missing | ECG="
            f"{ecg_missing*100:.2f}% | "

            f"PLETH="
            f"{pleth_missing*100:.2f}% | "

            f"ART="
            f"{art_missing*100:.2f}%"
        )


        # ====================================================
        # INTERPOLATE
        # ====================================================

        ecg, _ = interpolate_waveform(
            ecg_raw
        )


        pleth, _ = interpolate_waveform(
            pleth_raw
        )


        art, _ = interpolate_waveform(
            art_raw
        )


        # ====================================================
        # PEAK DETECTION
        # ====================================================

        (
            r_peaks,
            ecg_filtered,
            ecg_polarity

        ) = detect_r_peaks(

            ecg,

            FS
        )


        (
            pleth_peaks,
            pleth_filtered,
            pleth_polarity

        ) = detect_pulse_peaks(

            pleth,

            FS,

            "PLETH"
        )


        (
            art_peaks,
            art_filtered,
            art_polarity

        ) = detect_pulse_peaks(

            art,

            FS,

            "ART"
        )


        print(
            f"R-peaks: "
            f"{len(r_peaks)}"
        )


        print(
            f"PLETH peaks: "
            f"{len(pleth_peaks)}"
        )


        print(
            f"ART peaks: "
            f"{len(art_peaks)}"
        )


        # ====================================================
        # ECG HR
        # ====================================================

        ecg_hr = calculate_ecg_hr(

            r_peaks,

            FS
        )


        print(
            f"Estimated HR: "
            f"{ecg_hr:.1f} bpm"
        )


        # ====================================================
        # ECG -> ART
        #
        # R-wave -> arterial systolic peak
        # ====================================================

        ecg_art_lags = (
            calculate_event_lags(

                r_peaks,

                art_peaks,

                FS,

                min_lag_sec=0.05,

                max_lag_sec=0.60
            )
        )


        # ====================================================
        # ECG -> PLETH
        #
        # R-wave -> PPG systolic peak
        # ====================================================

        ecg_pleth_lags = (
            calculate_event_lags(

                r_peaks,

                pleth_peaks,

                FS,

                min_lag_sec=0.10,

                max_lag_sec=1.00
            )
        )


        # ====================================================
        # ART -> PLETH
        # ====================================================

        art_pleth_lags = (
            calculate_event_lags(

                art_peaks,

                pleth_peaks,

                FS,

                min_lag_sec=0.00,

                max_lag_sec=0.60
            )
        )


        ecg_art_stats = describe_lags(
            ecg_art_lags
        )


        ecg_pleth_stats = describe_lags(
            ecg_pleth_lags
        )


        art_pleth_stats = describe_lags(
            art_pleth_lags
        )


        print(
            "\nEvent timing:"
        )


        print(

            f"ECG -> ART median lag = "
            f"{ecg_art_stats['median']:.3f} s "
            f"(N={ecg_art_stats['n']})"
        )


        print(

            f"ECG -> PLETH median lag = "
            f"{ecg_pleth_stats['median']:.3f} s "
            f"(N={ecg_pleth_stats['n']})"
        )


        print(

            f"ART -> PLETH median lag = "
            f"{art_pleth_stats['median']:.3f} s "
            f"(N={art_pleth_stats['n']})"
        )


        # ====================================================
        # ART - PLETH CROSS CORRELATION
        # ====================================================

        art_pleth_corr, (
            art_pleth_xcorr_lag
        ) = calculate_cross_correlation(

            art_filtered,

            pleth_filtered,

            FS,

            max_lag_sec=1.0
        )


        print(
            "\nART-PLETH cross correlation:"
        )


        print(
            f"Max correlation = "
            f"{art_pleth_corr:.3f}"
        )


        print(
            f"Lag = "
            f"{art_pleth_xcorr_lag:.3f} s"
        )


        # ====================================================
        # SIMPLE ART-PPG AMPLITUDE COUPLING
        #
        # 5-second windows
        # ====================================================

        window_samples = (
            5
            * FS
        )


        art_amp_values = []

        pleth_amp_values = []


        for start in range(

            0,

            len(
                art
            )
            -
            window_samples
            +
            1,

            window_samples
        ):


            end = (
                start
                +
                window_samples
            )


            art_segment = art[
                start:end
            ]


            pleth_segment = pleth[
                start:end
            ]


            art_amp = (

                np.percentile(
                    art_segment,
                    95
                )

                -

                np.percentile(
                    art_segment,
                    5
                )
            )


            pleth_amp = (

                np.percentile(
                    pleth_segment,
                    95
                )

                -

                np.percentile(
                    pleth_segment,
                    5
                )
            )


            art_amp_values.append(
                art_amp
            )


            pleth_amp_values.append(
                pleth_amp
            )


        if (
            len(
                art_amp_values
            )
            >= 10
        ):

            amp_corr = float(

                pd.Series(
                    art_amp_values
                )
                .corr(

                    pd.Series(
                        pleth_amp_values
                    ),

                    method="spearman"
                )
            )


        else:

            amp_corr = np.nan


        print(

            f"ART-PPG amplitude "
            f"Spearman rho = "
            f"{amp_corr:.3f}"
        )


        # ====================================================
        # PILOT PASS
        # ====================================================

        # 10분이면 정상 HR에서 beat가 충분히 많아야 함
        enough_ecg = (
            len(
                r_peaks
            )
            >= 300
        )


        enough_art = (
            len(
                art_peaks
            )
            >= 300
        )


        enough_pleth = (
            len(
                pleth_peaks
            )
            >= 300
        )


        enough_pairs = (

            ecg_art_stats[
                "n"
            ]
            >= 200

            and

            ecg_pleth_stats[
                "n"
            ]
            >= 200
        )


        plausible_hr = (

            pd.notna(
                ecg_hr
            )

            and

            ecg_hr
            >= 30

            and

            ecg_hr
            <= 180
        )


        pilot_pass = (

            enough_ecg

            and

            enough_art

            and

            enough_pleth

            and

            enough_pairs

            and

            plausible_hr
        )


        # ====================================================
        # RESULT
        # ====================================================

        result_rows.append({

            "pilot_order":
                pilot_index,

            "caseid":
                caseid,

            "operation_duration_min":
                operation_min,

            "pilot_window_start_min":
                window_info[
                    "start_sec"
                ]
                / 60,

            "pilot_window_coverage":
                window_info[
                    "coverage"
                ],

            "ecg_missing_fraction":
                ecg_missing,

            "pleth_missing_fraction":
                pleth_missing,

            "art_missing_fraction":
                art_missing,

            "n_r_peaks":
                len(
                    r_peaks
                ),

            "n_pleth_peaks":
                len(
                    pleth_peaks
                ),

            "n_art_peaks":
                len(
                    art_peaks
                ),

            "ecg_estimated_hr_bpm":
                ecg_hr,

            "ecg_polarity":
                ecg_polarity,

            "pleth_polarity":
                pleth_polarity,

            "art_polarity":
                art_polarity,


            # --------------------------------------------
            # ECG -> ART
            # --------------------------------------------

            "ecg_art_pairs_n":
                ecg_art_stats[
                    "n"
                ],

            "ecg_art_lag_median_sec":
                ecg_art_stats[
                    "median"
                ],

            "ecg_art_lag_q1_sec":
                ecg_art_stats[
                    "q1"
                ],

            "ecg_art_lag_q3_sec":
                ecg_art_stats[
                    "q3"
                ],


            # --------------------------------------------
            # ECG -> PLETH
            # --------------------------------------------

            "ecg_pleth_pairs_n":
                ecg_pleth_stats[
                    "n"
                ],

            "ecg_pleth_lag_median_sec":
                ecg_pleth_stats[
                    "median"
                ],

            "ecg_pleth_lag_q1_sec":
                ecg_pleth_stats[
                    "q1"
                ],

            "ecg_pleth_lag_q3_sec":
                ecg_pleth_stats[
                    "q3"
                ],


            # --------------------------------------------
            # ART -> PLETH
            # --------------------------------------------

            "art_pleth_pairs_n":
                art_pleth_stats[
                    "n"
                ],

            "art_pleth_lag_median_sec":
                art_pleth_stats[
                    "median"
                ],

            "art_pleth_lag_q1_sec":
                art_pleth_stats[
                    "q1"
                ],

            "art_pleth_lag_q3_sec":
                art_pleth_stats[
                    "q3"
                ],


            # --------------------------------------------
            # coupling
            # --------------------------------------------

            "art_pleth_max_xcorr":
                art_pleth_corr,

            "art_pleth_xcorr_lag_sec":
                art_pleth_xcorr_lag,

            "art_pleth_amplitude_spearman":
                amp_corr,

            "pilot_pass":
                int(
                    pilot_pass
                )
        })


        # ====================================================
        # FIGURE
        #
        # first 20 seconds
        # ====================================================

        plot_sec = 20


        plot_samples = min(

            plot_sec
            * FS,

            len(
                ecg_filtered
            )
        )


        t = (

            np.arange(
                plot_samples
            )

            /
            FS
        )


        # ----------------------------------------------------
        # normalized for visualization
        # ----------------------------------------------------

        def zscore(
            x
        ):

            x = np.asarray(
                x,
                dtype=float
            )


            sd = np.std(
                x
            )


            if sd == 0:

                return (
                    x
                    -
                    np.mean(
                        x
                    )
                )


            return (

                x
                -
                np.mean(
                    x
                )

            ) / sd


        fig, axes = plt.subplots(

            3,

            1,

            figsize=(
                14,
                9
            ),

            sharex=True
        )


        # ----------------------------------------------------
        # ECG
        # ----------------------------------------------------

        axes[0].plot(

            t,

            zscore(
                ecg_filtered[
                    :plot_samples
                ]
            )
        )


        r_plot = r_peaks[

            r_peaks
            <
            plot_samples

        ]


        axes[0].scatter(

            r_plot
            /
            FS,

            zscore(
                ecg_filtered[
                    :plot_samples
                ]
            )[
                r_plot
            ],

            s=20
        )


        axes[0].set_ylabel(
            "ECG"
        )


        axes[0].set_title(

            f"Case {caseid} - "
            f"Waveform Peak Detection"
        )


        # ----------------------------------------------------
        # PLETH
        # ----------------------------------------------------

        axes[1].plot(

            t,

            zscore(
                pleth_filtered[
                    :plot_samples
                ]
            )
        )


        p_plot = pleth_peaks[

            pleth_peaks
            <
            plot_samples

        ]


        axes[1].scatter(

            p_plot
            /
            FS,

            zscore(
                pleth_filtered[
                    :plot_samples
                ]
            )[
                p_plot
            ],

            s=20
        )


        axes[1].set_ylabel(
            "PLETH"
        )


        # ----------------------------------------------------
        # ART
        # ----------------------------------------------------

        axes[2].plot(

            t,

            zscore(
                art_filtered[
                    :plot_samples
                ]
            )
        )


        a_plot = art_peaks[

            art_peaks
            <
            plot_samples

        ]


        axes[2].scatter(

            a_plot
            /
            FS,

            zscore(
                art_filtered[
                    :plot_samples
                ]
            )[
                a_plot
            ],

            s=20
        )


        axes[2].set_ylabel(
            "ART"
        )


        axes[2].set_xlabel(
            "Time (seconds)"
        )


        plt.tight_layout()


        figure_path = (

            FIGURE_DIR

            /

            (
                f"case_{caseid:05d}_"
                f"waveform_peaks.png"
            )
        )


        plt.savefig(

            figure_path,

            dpi=200,

            bbox_inches="tight"
        )


        plt.close()


        print(
            f"\nPilot PASS = "
            f"{pilot_pass}"
        )


        print(
            f"Figure saved: "
            f"{figure_path}"
        )


    except Exception as error:


        print(
            f"\nERROR: "
            f"{error}"
        )


        error_rows.append({

            "pilot_order":
                pilot_index,

            "caseid":
                caseid,

            "error":
                str(
                    error
                ),

            "traceback":
                traceback.format_exc()
        })


# ============================================================
# 19. SAVE
# ============================================================

result_df = pd.DataFrame(
    result_rows
)


error_df = pd.DataFrame(
    error_rows
)


result_df.to_csv(

    PILOT_RESULT_PATH,

    index=False,

    encoding="utf-8-sig"
)


error_df.to_csv(

    PILOT_ERROR_PATH,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 90)

print(
    "STEP 03 PILOT SUMMARY"
)

print("=" * 90)


print(
    f"\nSelected cases: "
    f"{len(pilot_cases)}"
)


print(
    f"Successfully analyzed: "
    f"{len(result_df)}"
)


print(
    f"Errors: "
    f"{len(error_df)}"
)


if (
    len(
        result_df
    )
    > 0
):


    pass_n = int(

        result_df[
            "pilot_pass"
        ]
        .sum()
    )


    print(
        f"\nPilot PASS: "
        f"{pass_n} / "
        f"{len(result_df)}"
    )


    # --------------------------------------------------------
    # Median values
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)

    print(
        "PILOT COUPLING RESULTS"
    )

    print("=" * 90)


    columns_to_summarize = [

        "ecg_estimated_hr_bpm",

        "ecg_art_lag_median_sec",

        "ecg_pleth_lag_median_sec",

        "art_pleth_lag_median_sec",

        "art_pleth_max_xcorr",

        "art_pleth_xcorr_lag_sec",

        "art_pleth_amplitude_spearman",
    ]


    for col in (
        columns_to_summarize
    ):


        values = (

            result_df[
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


        if len(
            values
        ) == 0:

            continue


        print(
            f"\n{col}"
        )


        print(
            f"Median = "
            f"{values.median():.4f}"
        )


        print(
            f"Range = "
            f"{values.min():.4f} - "
            f"{values.max():.4f}"
        )


# ============================================================
# 21. DECISION
# ============================================================

print("\n")
print("=" * 90)

print(
    "STEP 03 DECISION"
)

print("=" * 90)


if (
    len(
        result_df
    )
    == 0
):

    decision = (
        "FAIL"
    )


elif (
    result_df[
        "pilot_pass"
    ]
    .mean()
    >= 0.80
):

    decision = (
        "PASS"
    )


elif (
    result_df[
        "pilot_pass"
    ]
    .mean()
    >= 0.60
):

    decision = (
        "REVIEW"
    )


else:

    decision = (
        "FAIL"
    )


print(
    f"\nDecision: "
    f"{decision}"
)


if (
    decision
    == "PASS"
):

    print(
        """
Waveform peak detection 및 timing 추출이
대표 case에서 안정적으로 동작합니다.

다음 단계에서는 전체 cohort를 바로 처리하기 전에
최종 coupling feature 정의를 고정합니다.
"""
    )


elif (
    decision
    == "REVIEW"
):

    print(
        """
일부 waveform에서 peak detection 문제가 있습니다.

저장된 figure를 확인하여 ECG / PLETH / ART 중
어떤 신호에서 검출 오류가 발생하는지 확인해야 합니다.
"""
    )


else:

    print(
        """
현재 peak detection 방식으로는 전체 cohort 확장이 어렵습니다.

전체 waveform 분석 전에 preprocessing 알고리즘을
수정해야 합니다.
"""
    )


# ============================================================
# 22. FINISH
# ============================================================

print("\n")
print("=" * 90)

print(
    "STEP 03 완료"
)

print("=" * 90)


print(
    "\n생성 파일:"
)


print(
    SELECTED_CASES_PATH
)


print(
    PILOT_RESULT_PATH
)


print(
    PILOT_ERROR_PATH
)


print(
    f"\nFigures:"
)

print(
    FIGURE_DIR
)


print("\n다음 단계:")

print(
    """
STEP 04

Pilot PASS인 경우 최종 coupling feature set을
결과를 보기 전에 미리 고정합니다.

예정 후보:

- ECG -> ART median timing
- ECG -> PLETH median timing
- ART -> PLETH median timing

- beat-to-beat timing variability
- ART-PLETH cross-correlation
- ART-PLETH lag variability
- ART-PLETH amplitude coupling

- HR-MAP coupling
- MAP-ETCO2 coupling
- HR-ETCO2 coupling

그 후에만 전체 3,093 case 분석으로 확장합니다.
"""
)