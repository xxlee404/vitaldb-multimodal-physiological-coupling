# ============================================================
# VitalDB Multimodal Physiological Coupling Study
# STEP 03B
#
# Robust Peak Detection Validation
#
# 목적
# ------------------------------------------------------------
# 기존 STEP03에서 ECG R-peak 과검출 가능성이 확인됨.
#
# 이번 단계에서는:
#
# 1. 개선된 ECG QRS detector 사용
# 2. ART systolic peak 검출
# 3. PLETH systolic peak 검출
# 4. 각 신호에서 독립적으로 pulse rate 계산
# 5. Solar8000/HR numeric과 비교
# 6. ECG / ART / PLETH 간 rate consistency 평가
# 7. rate consistency가 확보된 경우에만 timing 계산
#
# PASS 기준
# ------------------------------------------------------------
# ECG HR vs Numeric HR:
#   absolute difference <= 10 bpm
#   또는 relative difference <= 15%
#
# ECG vs ART pulse rate:
#   difference <= 10 bpm
#
# ART vs PLETH:
#   difference <= 8 bpm
#
# 최소 4/5 pilot case 통과 시 전체 cohort 확장 가능
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
)

import vitaldb


warnings.filterwarnings("ignore")


# ============================================================
# 1. PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_DIR
    / "data_raw"
)

RESULT_DIR = (
    PROJECT_DIR
    / "results"
)

FIGURE_DIR = (
    PROJECT_DIR
    / "figures"
    / "03B_peak_validation"
)

NUMERIC_CACHE_DIR = (
    RAW_DIR
    / "numeric_tracks"
)


FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SELECTED_PATH = (
    RESULT_DIR
    / "03_pilot_selected_cases.csv"
)

OUTPUT_PATH = (
    RESULT_DIR
    / "03B_peak_validation_results.csv"
)

ERROR_PATH = (
    RESULT_DIR
    / "03B_peak_validation_errors.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

FS = 100

INTERVAL = (
    1.0 / FS
)

WINDOW_SEC = (
    10 * 60
)

TRACK_NAMES = [

    "SNUADC/ECG_II",

    "SNUADC/PLETH",

    "SNUADC/ART",
]


# ============================================================
# 3. LOAD PILOT CASES
# ============================================================

print("=" * 90)

print(
    "VitalDB Multimodal Physiological Coupling Study"
)

print(
    "STEP 03B - Robust Peak Detection Validation"
)

print("=" * 90)


if not SELECTED_PATH.exists():

    raise FileNotFoundError(
        f"파일 없음:\n{SELECTED_PATH}"
    )


pilot_df = pd.read_csv(
    SELECTED_PATH
)


pilot_df.columns = (
    pilot_df.columns
    .str.replace(
        "\ufeff",
        "",
        regex=False
    )
    .str.strip()
)


pilot_df["caseid"] = pd.to_numeric(
    pilot_df["caseid"],
    errors="coerce"
)


pilot_df = (
    pilot_df
    .dropna(
        subset=[
            "caseid"
        ]
    )
    .copy()
)


pilot_df["caseid"] = (
    pilot_df["caseid"]
    .astype(int)
)


print(
    f"\nPilot cases: "
    f"{len(pilot_df)}"
)


# ============================================================
# 4. FILTER
# ============================================================

def bandpass_filter(
    signal,
    fs,
    lowcut,
    highcut,
    order=3
):

    x = np.asarray(
        signal,
        dtype=float
    )


    nyquist = (
        fs / 2
    )


    sos = butter(

        order,

        [
            lowcut / nyquist,
            highcut / nyquist
        ],

        btype="bandpass",

        output="sos"
    )


    return sosfiltfilt(
        sos,
        x
    )


# ============================================================
# 5. INTERPOLATION
# ============================================================

def interpolate_signal(
    signal
):

    series = pd.Series(
        signal,
        dtype=float
    )


    missing_fraction = float(
        series.isna().mean()
    )


    series = (
        series
        .interpolate(
            method="linear",
            limit_direction="both"
        )
    )


    return (
        series.to_numpy(),
        missing_fraction
    )


# ============================================================
# 6. ROBUST ECG R-PEAK DETECTOR
#
# Pan-Tompkins-like approach
# ============================================================

def detect_ecg_rpeaks(
    ecg,
    fs
):

    # --------------------------------------------------------
    # QRS band
    # --------------------------------------------------------

    filtered = bandpass_filter(

        ecg,

        fs,

        5.0,

        20.0,

        order=3
    )


    # --------------------------------------------------------
    # Derivative
    # --------------------------------------------------------

    derivative = np.diff(
        filtered,
        prepend=filtered[0]
    )


    # --------------------------------------------------------
    # Square
    # --------------------------------------------------------

    squared = (
        derivative ** 2
    )


    # --------------------------------------------------------
    # Moving integration ~150 ms
    # --------------------------------------------------------

    integration_window = max(

        1,

        int(
            0.15 * fs
        )
    )


    kernel = (

        np.ones(
            integration_window
        )

        /
        integration_window
    )


    integrated = np.convolve(

        squared,

        kernel,

        mode="same"
    )


    # --------------------------------------------------------
    # Robust threshold
    # --------------------------------------------------------

    median_value = np.median(
        integrated
    )


    mad = np.median(

        np.abs(

            integrated

            -
            median_value
        )
    )


    threshold = (

        median_value

        +

        5.0
        * mad
    )


    if (
        threshold
        <= 0
    ):

        threshold = np.percentile(
            integrated,
            90
        )


    # --------------------------------------------------------
    # QRS candidates
    # --------------------------------------------------------

    candidates, _ = find_peaks(

        integrated,

        height=
            threshold,

        distance=int(
            0.28
            * fs
        )
    )


    # --------------------------------------------------------
    # 후보 주변 ±120 ms에서 실제 R 위치 찾기
    #
    # ECG polarity가 반대일 수도 있으므로
    # absolute amplitude 사용
    # --------------------------------------------------------

    search_radius = int(
        0.12
        * fs
    )


    r_peaks = []


    for candidate in candidates:

        start = max(
            0,
            candidate
            -
            search_radius
        )


        end = min(

            len(
                filtered
            ),

            candidate
            +
            search_radius
            +
            1
        )


        local = filtered[
            start:end
        ]


        if len(
            local
        ) == 0:

            continue


        local_index = int(

            np.argmax(
                np.abs(
                    local
                )
            )
        )


        peak_index = (

            start
            +
            local_index
        )


        r_peaks.append(
            peak_index
        )


    if len(
        r_peaks
    ) == 0:

        return (
            np.array(
                [],
                dtype=int
            ),
            filtered,
            integrated
        )


    r_peaks = np.asarray(

        sorted(
            set(
                r_peaks
            )
        ),

        dtype=int
    )


    # ========================================================
    # Refractory period
    #
    # 너무 가까운 peak가 2개면
    # abs ECG amplitude가 큰 것만 보존
    # ========================================================

    refractory_samples = int(
        0.30
        * fs
    )


    final_peaks = []


    for peak in r_peaks:

        if len(
            final_peaks
        ) == 0:

            final_peaks.append(
                peak
            )

            continue


        previous = (
            final_peaks[
                -1
            ]
        )


        if (
            peak
            -
            previous
            <
            refractory_samples
        ):

            if (

                abs(
                    filtered[
                        peak
                    ]
                )

                >

                abs(
                    filtered[
                        previous
                    ]
                )

            ):

                final_peaks[
                    -1
                ] = peak

        else:

            final_peaks.append(
                peak
            )


    return (

        np.asarray(
            final_peaks,
            dtype=int
        ),

        filtered,

        integrated
    )


# ============================================================
# 7. PULSE PEAK DETECTOR
#
# ART / PLETH
# ============================================================

def detect_pulse_peaks(
    signal,
    fs,
    signal_type
):


    if (
        signal_type
        == "ART"
    ):

        highcut = 12.0

    else:

        highcut = 8.0


    filtered = bandpass_filter(

        signal,

        fs,

        0.5,

        highcut,

        order=3
    )


    # --------------------------------------------------------
    # polarity 자동 결정
    # --------------------------------------------------------

    positive_strength = np.percentile(
        filtered,
        99
    )


    negative_strength = abs(
        np.percentile(
            filtered,
            1
        )
    )


    if (
        negative_strength
        >
        positive_strength
    ):

        detection = (
            -filtered
        )

    else:

        detection = (
            filtered
        )


    median = np.median(
        detection
    )


    mad = np.median(

        np.abs(

            detection

            -
            median
        )
    )


    prominence = max(

        1.5
        * mad,

        np.std(
            detection
        )
        * 0.10,

        1e-8
    )


    peaks, _ = find_peaks(

        detection,

        distance=int(
            0.35
            * fs
        ),

        prominence=
            prominence
    )


    return (
        peaks,
        filtered
    )


# ============================================================
# 8. RATE FROM PEAKS
# ============================================================

def rate_from_peaks(
    peaks,
    fs
):


    peaks = np.asarray(
        peaks
    )


    if len(
        peaks
    ) < 3:

        return np.nan


    intervals = (

        np.diff(
            peaks
        )

        /
        fs
    )


    # --------------------------------------------------------
    # plausible cardiac intervals
    # --------------------------------------------------------

    valid = intervals[

        (
            intervals
            >= 0.30
        )

        &

        (
            intervals
            <= 2.0
        )
    ]


    if len(
        valid
    ) < 2:

        return np.nan


    return float(

        60.0

        /
        np.median(
            valid
        )
    )


# ============================================================
# 9. COUNT RATE
#
# 10분 peak 개수를 이용한 보조 rate
# ============================================================

def count_rate(
    peaks,
    duration_sec
):


    if (
        duration_sec <= 0
    ):

        return np.nan


    return float(

        len(
            peaks
        )

        /
        duration_sec

        *
        60.0
    )


# ============================================================
# 10. BEST 10-MIN WINDOW
# ============================================================

def select_best_window(
    waveform,
    fs,
    window_sec
):


    valid = (

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


    total_sec = int(

        len(
            waveform
        )

        // fs
    )


    if (
        total_sec
        < window_sec
    ):

        return None


    valid = valid[
        :total_sec
        * fs
    ]


    sec_valid = (

        valid
        .reshape(
            total_sec,
            fs
        )
        .mean(
            axis=1
        )
    )


    moving = np.convolve(

        sec_valid,

        np.ones(
            window_sec
        ),

        mode="valid"
    )


    start_sec = int(

        np.argmax(
            moving
        )
    )


    coverage = float(

        moving[
            start_sec
        ]

        /
        window_sec
    )


    return {

        "start_sec":
            start_sec,

        "end_sec":
            start_sec
            +
            window_sec,

        "start_sample":
            start_sec
            * fs,

        "end_sample":

            (
                start_sec
                +
                window_sec
            )
            * fs,

        "coverage":
            coverage
    }


# ============================================================
# 11. NUMERIC HR CACHE
# ============================================================

def load_numeric_hr(
    caseid,
    absolute_start_sec,
    absolute_end_sec
):


    path = (

        NUMERIC_CACHE_DIR

        / "HR"

        /
        (
            f"case_{caseid:05d}_"
            f"HR.csv.gz"
        )
    )


    if not path.exists():

        return np.nan


    temp = pd.read_csv(
        path
    )


    if (
        "time_sec"
        not in temp.columns
        or
        "value"
        not in temp.columns
    ):

        return np.nan


    temp[
        "time_sec"
    ] = pd.to_numeric(

        temp[
            "time_sec"
        ],

        errors="coerce"
    )


    temp[
        "value"
    ] = pd.to_numeric(

        temp[
            "value"
        ],

        errors="coerce"
    )


    temp = temp.loc[

        (
            temp[
                "time_sec"
            ]
            >= absolute_start_sec
        )

        &

        (
            temp[
                "time_sec"
            ]
            <= absolute_end_sec
        )

        &

        (
            temp[
                "value"
            ]
            >= 20
        )

        &

        (
            temp[
                "value"
            ]
            <= 250
        )

    ]


    if len(
        temp
    ) == 0:

        return np.nan


    return float(

        temp[
            "value"
        ]
        .median()
    )


# ============================================================
# 12. EVENT LAGS
# ============================================================

def calculate_event_lags(
    source_peaks,
    target_peaks,
    fs,
    min_lag,
    max_lag
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


    lags = []


    for source in source_times:

        possible = target_times[

            (
                target_times
                >= source
                +
                min_lag
            )

            &

            (
                target_times
                <= source
                +
                max_lag
            )
        ]


        if len(
            possible
        ) > 0:

            lags.append(

                possible[0]
                -
                source
            )


    return np.asarray(
        lags
    )


# ============================================================
# 13. LOAD WAVEFORM
# ============================================================

def load_waveform(
    caseid,
    opstart,
    opend
):


    waveform = vitaldb.load_case(

        caseid,

        TRACK_NAMES,

        interval=
            INTERVAL
    )


    waveform = np.asarray(
        waveform,
        dtype=float
    )


    start = max(

        0,

        int(
            opstart
            * FS
        )
    )


    end = min(

        len(
            waveform
        ),

        int(
            opend
            * FS
        )
    )


    if (
        end <= start
    ):

        raise ValueError(
            "operation waveform crop 실패"
        )


    return waveform[
        start:end
    ]


# ============================================================
# 14. RESULTS
# ============================================================

result_rows = []

error_rows = []


# ============================================================
# 15. PILOT
# ============================================================

print("\n")
print("=" * 90)

print(
    "PEAK VALIDATION START"
)

print("=" * 90)


for pilot_no, (_, row) in enumerate(

    pilot_df.iterrows(),

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
        f"PILOT {pilot_no}/"
        f"{len(pilot_df)} "
        f"| caseid={caseid}"
    )

    print("-" * 90)


    try:

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
                "opstart/opend 오류"
            )


        # ====================================================
        # LOAD
        # ====================================================

        waveform = load_waveform(

            caseid,

            float(
                opstart
            ),

            float(
                opend
            )
        )


        best = select_best_window(

            waveform,

            FS,

            WINDOW_SEC
        )


        if (
            best is None
        ):

            raise ValueError(
                "10분 window 없음"
            )


        pilot = waveform[

            best[
                "start_sample"
            ]:

            best[
                "end_sample"
            ]

        ]


        ecg_raw = pilot[
            :,
            0
        ]

        pleth_raw = pilot[
            :,
            1
        ]

        art_raw = pilot[
            :,
            2
        ]


        (
            ecg,
            ecg_missing

        ) = interpolate_signal(
            ecg_raw
        )


        (
            pleth,
            pleth_missing

        ) = interpolate_signal(
            pleth_raw
        )


        (
            art,
            art_missing

        ) = interpolate_signal(
            art_raw
        )


        # ====================================================
        # DETECTION
        # ====================================================

        (
            r_peaks,
            ecg_filtered,
            qrs_envelope

        ) = detect_ecg_rpeaks(

            ecg,

            FS
        )


        (
            art_peaks,
            art_filtered

        ) = detect_pulse_peaks(

            art,

            FS,

            "ART"
        )


        (
            pleth_peaks,
            pleth_filtered

        ) = detect_pulse_peaks(

            pleth,

            FS,

            "PLETH"
        )


        # ====================================================
        # RATES
        # ====================================================

        ecg_hr = rate_from_peaks(

            r_peaks,

            FS
        )


        art_hr = rate_from_peaks(

            art_peaks,

            FS
        )


        pleth_hr = rate_from_peaks(

            pleth_peaks,

            FS
        )


        ecg_count_hr = count_rate(

            r_peaks,

            WINDOW_SEC
        )


        art_count_hr = count_rate(

            art_peaks,

            WINDOW_SEC
        )


        pleth_count_hr = count_rate(

            pleth_peaks,

            WINDOW_SEC
        )


        # ====================================================
        # NUMERIC HR
        # ====================================================

        absolute_window_start = (

            float(
                opstart
            )

            +
            best[
                "start_sec"
            ]
        )


        absolute_window_end = (

            absolute_window_start

            +
            WINDOW_SEC
        )


        numeric_hr = load_numeric_hr(

            caseid,

            absolute_window_start,

            absolute_window_end
        )


        # ====================================================
        # CONSISTENCY
        # ====================================================

        ecg_numeric_diff = (

            abs(
                ecg_hr
                -
                numeric_hr
            )

            if (
                pd.notna(
                    ecg_hr
                )

                and

                pd.notna(
                    numeric_hr
                )
            )

            else np.nan
        )


        ecg_art_diff = (

            abs(
                ecg_hr
                -
                art_hr
            )

            if (
                pd.notna(
                    ecg_hr
                )

                and

                pd.notna(
                    art_hr
                )
            )

            else np.nan
        )


        art_pleth_diff = (

            abs(
                art_hr
                -
                pleth_hr
            )

            if (
                pd.notna(
                    art_hr
                )

                and

                pd.notna(
                    pleth_hr
                )
            )

            else np.nan
        )


        # ----------------------------------------------------
        # Numeric HR consistency
        # ----------------------------------------------------

        numeric_consistent = False


        if (

            pd.notna(
                ecg_hr
            )

            and

            pd.notna(
                numeric_hr
            )

            and

            numeric_hr > 0

        ):

            relative_diff = (

                ecg_numeric_diff

                /
                numeric_hr
            )


            numeric_consistent = (

                ecg_numeric_diff
                <= 10

                or

                relative_diff
                <= 0.15
            )


        ecg_art_consistent = (

            pd.notna(
                ecg_art_diff
            )

            and

            ecg_art_diff
            <= 10
        )


        art_pleth_consistent = (

            pd.notna(
                art_pleth_diff
            )

            and

            art_pleth_diff
            <= 8
        )


        # ====================================================
        # LAGS
        # ====================================================

        ecg_art_lags = calculate_event_lags(

            r_peaks,

            art_peaks,

            FS,

            0.05,

            0.60
        )


        ecg_pleth_lags = calculate_event_lags(

            r_peaks,

            pleth_peaks,

            FS,

            0.10,

            1.00
        )


        art_pleth_lags = calculate_event_lags(

            art_peaks,

            pleth_peaks,

            FS,

            0.00,

            0.70
        )


        # ====================================================
        # FINAL PASS
        # ====================================================

        validation_pass = (

            numeric_consistent

            and

            ecg_art_consistent

            and

            art_pleth_consistent
        )


        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"Numeric HR        = "
            f"{numeric_hr:.1f}"
        )


        print(
            f"ECG HR            = "
            f"{ecg_hr:.1f}"
        )


        print(
            f"ART pulse rate    = "
            f"{art_hr:.1f}"
        )


        print(
            f"PLETH pulse rate  = "
            f"{pleth_hr:.1f}"
        )


        print(
            "\n10-min raw peak counts:"
        )


        print(
            f"ECG   = "
            f"{len(r_peaks)} "
            f"({ecg_count_hr:.1f}/min)"
        )


        print(
            f"ART   = "
            f"{len(art_peaks)} "
            f"({art_count_hr:.1f}/min)"
        )


        print(
            f"PLETH = "
            f"{len(pleth_peaks)} "
            f"({pleth_count_hr:.1f}/min)"
        )


        print(
            "\nRate differences:"
        )


        print(
            f"ECG vs Numeric = "
            f"{ecg_numeric_diff:.1f} bpm"
        )


        print(
            f"ECG vs ART     = "
            f"{ecg_art_diff:.1f} bpm"
        )


        print(
            f"ART vs PLETH   = "
            f"{art_pleth_diff:.1f} bpm"
        )


        print(
            "\nTiming:"
        )


        print(

            f"ECG -> ART = "
            f"{np.median(ecg_art_lags):.3f} sec "
            f"(N={len(ecg_art_lags)})"

            if len(
                ecg_art_lags
            ) > 0

            else
            "ECG -> ART = NA"
        )


        print(

            f"ECG -> PLETH = "
            f"{np.median(ecg_pleth_lags):.3f} sec "
            f"(N={len(ecg_pleth_lags)})"

            if len(
                ecg_pleth_lags
            ) > 0

            else
            "ECG -> PLETH = NA"
        )


        print(

            f"ART -> PLETH = "
            f"{np.median(art_pleth_lags):.3f} sec "
            f"(N={len(art_pleth_lags)})"

            if len(
                art_pleth_lags
            ) > 0

            else
            "ART -> PLETH = NA"
        )


        print(
            f"\nVALIDATION PASS = "
            f"{validation_pass}"
        )


        # ====================================================
        # SAVE RESULT
        # ====================================================

        result_rows.append({

            "pilot_no":
                pilot_no,

            "caseid":
                caseid,

            "numeric_hr":
                numeric_hr,

            "ecg_hr":
                ecg_hr,

            "art_hr":
                art_hr,

            "pleth_hr":
                pleth_hr,

            "ecg_peak_count":
                len(
                    r_peaks
                ),

            "art_peak_count":
                len(
                    art_peaks
                ),

            "pleth_peak_count":
                len(
                    pleth_peaks
                ),

            "ecg_count_rate":
                ecg_count_hr,

            "art_count_rate":
                art_count_hr,

            "pleth_count_rate":
                pleth_count_hr,

            "ecg_numeric_diff":
                ecg_numeric_diff,

            "ecg_art_diff":
                ecg_art_diff,

            "art_pleth_diff":
                art_pleth_diff,

            "numeric_consistent":
                int(
                    numeric_consistent
                ),

            "ecg_art_consistent":
                int(
                    ecg_art_consistent
                ),

            "art_pleth_consistent":
                int(
                    art_pleth_consistent
                ),

            "ecg_art_lag_median_sec":

                (
                    np.median(
                        ecg_art_lags
                    )

                    if len(
                        ecg_art_lags
                    ) > 0

                    else np.nan
                ),

            "ecg_pleth_lag_median_sec":

                (
                    np.median(
                        ecg_pleth_lags
                    )

                    if len(
                        ecg_pleth_lags
                    ) > 0

                    else np.nan
                ),

            "art_pleth_lag_median_sec":

                (
                    np.median(
                        art_pleth_lags
                    )

                    if len(
                        art_pleth_lags
                    ) > 0

                    else np.nan
                ),

            "validation_pass":
                int(
                    validation_pass
                ),
        })


        # ====================================================
        # FIGURE
        #
        # 첫 15초만 그림
        # ====================================================

        plot_sec = 15

        plot_n = (
            plot_sec
            * FS
        )


        time_axis = (

            np.arange(
                plot_n
            )

            /
            FS
        )


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

            time_axis,

            ecg_filtered[
                :plot_n
            ]
        )


        r_plot = r_peaks[

            r_peaks
            <
            plot_n

        ]


        axes[0].scatter(

            r_plot
            /
            FS,

            ecg_filtered[
                r_plot
            ],

            s=20
        )


        axes[0].set_ylabel(
            "ECG"
        )


        axes[0].set_title(

            f"Case {caseid} | "
            f"Peak Validation"
        )


        # ----------------------------------------------------
        # ART
        # ----------------------------------------------------

        axes[1].plot(

            time_axis,

            art_filtered[
                :plot_n
            ]
        )


        art_plot = art_peaks[

            art_peaks
            <
            plot_n

        ]


        axes[1].scatter(

            art_plot
            /
            FS,

            art_filtered[
                art_plot
            ],

            s=20
        )


        axes[1].set_ylabel(
            "ART"
        )


        # ----------------------------------------------------
        # PLETH
        # ----------------------------------------------------

        axes[2].plot(

            time_axis,

            pleth_filtered[
                :plot_n
            ]
        )


        pleth_plot = pleth_peaks[

            pleth_peaks
            <
            plot_n

        ]


        axes[2].scatter(

            pleth_plot
            /
            FS,

            pleth_filtered[
                pleth_plot
            ],

            s=20
        )


        axes[2].set_ylabel(
            "PLETH"
        )


        axes[2].set_xlabel(
            "Time (sec)"
        )


        plt.tight_layout()


        figure_path = (

            FIGURE_DIR

            /
            (
                f"case_{caseid:05d}_"
                f"validated_peaks.png"
            )
        )


        plt.savefig(

            figure_path,

            dpi=200,

            bbox_inches="tight"
        )


        plt.close()


    except Exception as error:

        print(
            f"\nERROR: {error}"
        )


        error_rows.append({

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
# 16. SAVE
# ============================================================

result_df = pd.DataFrame(
    result_rows
)


error_df = pd.DataFrame(
    error_rows
)


result_df.to_csv(

    OUTPUT_PATH,

    index=False,

    encoding="utf-8-sig"
)


error_df.to_csv(

    ERROR_PATH,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 17. SUMMARY
# ============================================================

print("\n")
print("=" * 90)

print(
    "STEP 03B VALIDATION SUMMARY"
)

print("=" * 90)


print(
    f"\nAnalyzed = "
    f"{len(result_df)}"
)


print(
    f"Errors = "
    f"{len(error_df)}"
)


if len(
    result_df
) > 0:

    passed = int(

        result_df[
            "validation_pass"
        ]
        .sum()
    )


    print(
        f"Validation PASS = "
        f"{passed} / "
        f"{len(result_df)}"
    )


    print("\nMedian rates:")


    for col in [

        "numeric_hr",
        "ecg_hr",
        "art_hr",
        "pleth_hr",

    ]:

        print(

            f"{col:<20} "
            f"{result_df[col].median():.2f} bpm"
        )


    print("\nMedian timing:")


    for col in [

        "ecg_art_lag_median_sec",

        "ecg_pleth_lag_median_sec",

        "art_pleth_lag_median_sec",

    ]:

        print(

            f"{col:<35} "
            f"{result_df[col].median():.3f} sec"
        )


# ============================================================
# 18. DECISION
# ============================================================

print("\n")
print("=" * 90)

print(
    "STEP 03B DECISION"
)

print("=" * 90)


if len(
    result_df
) == 0:

    decision = "FAIL"


elif (

    result_df[
        "validation_pass"
    ]
    .mean()

    >= 0.80

):

    decision = "PASS"


elif (

    result_df[
        "validation_pass"
    ]
    .mean()

    >= 0.60

):

    decision = "REVIEW"


else:

    decision = "FAIL"


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
ECG / ART / PLETH beat detection이 실제 numeric HR과
생리적으로 일치하는 수준으로 검증되었습니다.

다음 STEP 04에서 최종 coupling feature를 고정할 수 있습니다.
"""
    )


else:

    print(
        """
아직 waveform peak detection을 전체 cohort로
확장하면 안 됩니다.

저장된 peak figure와 rate 차이를 기준으로
검출 알고리즘을 추가 수정해야 합니다.
"""
    )


print("\n생성 파일:")

print(
    OUTPUT_PATH
)

print(
    ERROR_PATH
)

print(
    FIGURE_DIR
)