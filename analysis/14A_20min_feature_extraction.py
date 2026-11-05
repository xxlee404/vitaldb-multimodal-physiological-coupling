# -*- coding: utf-8 -*-

from pathlib import Path
import time
import traceback
import warnings

import numpy as np
import pandas as pd
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
# BSPC REVISION
# STEP 14A
# First-20-min Window Sensitivity Feature Extraction
# ============================================================

print("=" * 96)
print("BSPC REVISION - STEP 14A")
print("20-MIN WINDOW SENSITIVITY FEATURE EXTRACTION")
print("=" * 96)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_DIR / "data_raw"
PROCESSED_DIR = PROJECT_DIR / "data_processed"
RESULT_DIR = PROJECT_DIR / "results"

NUMERIC_CACHE_DIR = RAW_DIR / "numeric_tracks"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Primary 30-min frozen cohort
INPUT_PATH = (
    PROCESSED_DIR
    / "06_final_analysis_cohort_locked.csv"
)

# Revision outputs
CHECKPOINT_PATH = (
    RESULT_DIR
    / "14A_20min_feature_checkpoint.csv"
)

OUTPUT_ALL_PATH = (
    PROCESSED_DIR
    / "14A_20min_coupling_features_all.csv"
)

OUTPUT_QC_PASS_PATH = (
    PROCESSED_DIR
    / "14A_20min_coupling_features_qc_pass.csv"
)

ERROR_PATH = (
    RESULT_DIR
    / "14A_20min_feature_errors.csv"
)

QC_REASON_PATH = (
    RESULT_DIR
    / "14A_20min_qc_reason_counts.csv"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "14A_20min_feature_summary.csv"
)

DISTRIBUTION_PATH = (
    RESULT_DIR
    / "14A_20min_feature_distributions.csv"
)


# ============================================================
# 2. SENSITIVITY SETTINGS
# ============================================================

FS = 50
INTERVAL = 1.0 / FS

# Primary = 30 min
# Sensitivity = first 20 min
WINDOW_SEC = 20 * 60

TRACK_NAMES = [
    "SNUADC/PLETH",
    "SNUADC/ART",
]

# Same signal-quality definitions
MIN_WAVE_COVERAGE = 0.80
MAX_RATE_DIFF_BPM = 8.0

# Duration-scaled count thresholds
# Primary:
#   500 pairs / 30 min
#   15 xcorr windows / 30 min
#   6 numeric windows / max 11
#
# Sensitivity:
#   same approximate relative sufficiency
MIN_LAG_PAIRS = 334
MIN_XCORR_WINDOWS = 10
MIN_NUMERIC_WINDOWS = 4

# Same rolling correlation definition
ROLLING_WINDOW_SEC = 5 * 60
ROLLING_STEP_SEC = 150
MIN_NUMERIC_WINDOW_COVERAGE = 0.80

# VitalDB loading
MAX_RETRIES = 3
RETRY_WAIT_SEC = 3

# Progress saving
CHECKPOINT_EVERY = 10
PRINT_EVERY = 10


# ============================================================
# 3. LOCKED 12 COUPLING FEATURES
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
# 4. HELPERS
# ============================================================

def calc_iqr(values):

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(
        np.percentile(x, 75)
        - np.percentile(x, 25)
    )


def bandpass_filter(
    signal,
    fs,
    lowcut,
    highcut,
    order=3,
):

    x = np.asarray(
        signal,
        dtype=float,
    )

    nyquist = fs / 2.0

    sos = butter(
        order,
        [
            lowcut / nyquist,
            highcut / nyquist,
        ],
        btype="bandpass",
        output="sos",
    )

    return sosfiltfilt(
        sos,
        x,
    )


def interpolate_waveform(signal):

    series = pd.Series(
        signal,
        dtype=float,
    )

    missing_fraction = float(
        series.isna().mean()
    )

    # Same procedure as primary analysis
    series = series.interpolate(
        method="linear",
        limit_direction="both",
    )

    return (
        series.to_numpy(),
        missing_fraction,
    )


def detect_pulse_peaks(
    signal,
    fs,
    signal_type,
):

    highcut = (
        12.0
        if signal_type == "ART"
        else 8.0
    )

    filtered = bandpass_filter(
        signal,
        fs,
        0.5,
        highcut,
        order=3,
    )

    positive_strength = np.percentile(
        filtered,
        99,
    )

    negative_strength = abs(
        np.percentile(
            filtered,
            1,
        )
    )

    detection = (
        -filtered
        if negative_strength > positive_strength
        else filtered
    )

    median_value = np.median(
        detection
    )

    mad = np.median(
        np.abs(
            detection
            - median_value
        )
    )

    prominence = max(
        1.5 * mad,
        np.std(detection) * 0.10,
        1e-8,
    )

    peaks, _ = find_peaks(
        detection,
        distance=int(
            0.35 * fs
        ),
        prominence=prominence,
    )

    return (
        peaks,
        filtered,
    )


def rate_from_peaks(
    peaks,
    fs,
):

    peaks = np.asarray(
        peaks,
        dtype=int,
    )

    if len(peaks) < 3:
        return np.nan

    intervals = (
        np.diff(peaks)
        / fs
    )

    intervals = intervals[
        (intervals >= 0.30)
        & (intervals <= 2.0)
    ]

    if len(intervals) < 2:
        return np.nan

    return float(
        60.0
        / np.median(intervals)
    )


def pair_art_pleth(
    art_peaks,
    pleth_peaks,
    fs,
    min_lag=0.0,
    max_lag=0.8,
):

    art_peaks = np.asarray(
        art_peaks,
        dtype=int,
    )

    pleth_peaks = np.asarray(
        pleth_peaks,
        dtype=int,
    )

    pairs = []

    j = 0

    for art_peak in art_peaks:

        min_target = (
            art_peak
            + int(
                round(
                    min_lag * fs
                )
            )
        )

        max_target = (
            art_peak
            + int(
                round(
                    max_lag * fs
                )
            )
        )

        while (
            j < len(pleth_peaks)
            and pleth_peaks[j] < min_target
        ):
            j += 1

        if j >= len(pleth_peaks):
            break

        if (
            pleth_peaks[j]
            <= max_target
        ):

            pairs.append(
                (
                    art_peak,
                    pleth_peaks[j],
                )
            )

            j += 1

    return pairs


def local_pulse_amplitude(
    signal,
    peak_index,
    fs,
    pre_sec=0.6,
):

    start = max(
        0,
        int(
            peak_index
            - pre_sec * fs
        ),
    )

    if peak_index <= start:
        return np.nan

    segment = signal[
        start:peak_index + 1
    ]

    if len(segment) == 0:
        return np.nan

    baseline = np.nanmin(
        segment
    )

    return float(
        signal[peak_index]
        - baseline
    )


def positive_lag_xcorr(
    source,
    target,
    fs,
    max_lag_sec=0.8,
):

    source = np.asarray(
        source,
        dtype=float,
    )

    target = np.asarray(
        target,
        dtype=float,
    )

    if (
        np.std(source) == 0
        or np.std(target) == 0
    ):
        return (
            np.nan,
            np.nan,
        )

    source_z = (
        source
        - np.mean(source)
    ) / np.std(source)

    target_z = (
        target
        - np.mean(target)
    ) / np.std(target)

    corr = correlate(
        target_z,
        source_z,
        mode="full",
        method="fft",
    )

    lags = correlation_lags(
        len(target_z),
        len(source_z),
        mode="full",
    )

    max_lag_samples = int(
        max_lag_sec
        * fs
    )

    mask = (
        (lags >= 0)
        & (
            lags
            <= max_lag_samples
        )
    )

    corr = (
        corr[mask]
        / len(source_z)
    )

    lags = lags[mask]

    if len(corr) == 0:
        return (
            np.nan,
            np.nan,
        )

    best = int(
        np.argmax(corr)
    )

    return (
        float(corr[best]),
        float(
            lags[best]
            / fs
        ),
    )


def load_numeric_window(
    caseid,
    signal_name,
    start_sec,
    end_sec,
    min_value,
    max_value,
):

    path = (
        NUMERIC_CACHE_DIR
        / signal_name
        / (
            f"case_{caseid:05d}_"
            f"{signal_name}.csv.gz"
        )
    )

    if not path.exists():
        return None

    data = pd.read_csv(path)

    if (
        "time_sec"
        not in data.columns
        or "value"
        not in data.columns
    ):
        return None

    data["time_sec"] = pd.to_numeric(
        data["time_sec"],
        errors="coerce",
    )

    data["value"] = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["time_sec"]
    )

    data = data.loc[
        (
            data["time_sec"]
            >= start_sec
        )
        & (
            data["time_sec"]
            < end_sec
        )
    ].copy()

    data.loc[
        (
            data["value"]
            < min_value
        )
        | (
            data["value"]
            > max_value
        ),
        "value",
    ] = np.nan

    if len(data) > 0:

        data["second"] = np.floor(
            data["time_sec"]
        ).astype(int)

        per_second = (
            data
            .groupby("second")[
                "value"
            ]
            .median()
        )

    else:

        per_second = pd.Series(
            dtype=float
        )

    full_index = np.arange(
        int(
            np.floor(start_sec)
        ),
        int(
            np.floor(end_sec)
        ),
        dtype=int,
    )

    series = per_second.reindex(
        full_index
    )

    # Same <=10 sec interpolation rule
    series = series.interpolate(
        method="linear",
        limit=10,
        limit_area="inside",
    )

    return series.reset_index(
        drop=True
    )


def rolling_spearman(
    series_a,
    series_b,
):

    a = pd.Series(
        series_a,
        dtype=float,
    ).reset_index(
        drop=True
    )

    b = pd.Series(
        series_b,
        dtype=float,
    ).reset_index(
        drop=True
    )

    n = min(
        len(a),
        len(b),
    )

    correlations = []

    if n < ROLLING_WINDOW_SEC:
        return np.asarray(
            [],
            dtype=float,
        )

    for start in range(
        0,
        n - ROLLING_WINDOW_SEC + 1,
        ROLLING_STEP_SEC,
    ):

        end = (
            start
            + ROLLING_WINDOW_SEC
        )

        x = a.iloc[
            start:end
        ]

        y = b.iloc[
            start:end
        ]

        valid = (
            x.notna()
            & y.notna()
        )

        if (
            valid.mean()
            < MIN_NUMERIC_WINDOW_COVERAGE
        ):
            continue

        x_valid = x[valid]
        y_valid = y[valid]

        if (
            x_valid.nunique() < 3
            or y_valid.nunique() < 3
        ):
            continue

        rho = x_valid.corr(
            y_valid,
            method="spearman",
        )

        if pd.notna(rho):
            correlations.append(
                float(rho)
            )

    return np.asarray(
        correlations,
        dtype=float,
    )


# ============================================================
# 5. LOAD WAVEFORMS
# ============================================================

def load_waveforms_with_retry(
    caseid,
):

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            values = vitaldb.load_case(
                caseid,
                TRACK_NAMES,
                interval=INTERVAL,
            )

            values = np.asarray(
                values,
                dtype=float,
            )

            if (
                values.ndim != 2
                or values.shape[1] != 2
            ):
                raise ValueError(
                    "Unexpected waveform "
                    f"shape: {values.shape}"
                )

            return values

        except Exception as error:

            last_error = error

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_WAIT_SEC
                    * attempt
                )

    raise RuntimeError(
        "Waveform load failed "
        f"after {MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# 6. EXTRACT ONE CASE
# ============================================================

def extract_one_case(row):

    caseid = int(
        row["caseid"]
    )

    subjectid = row.get(
        "subjectid",
        np.nan,
    )

    start_sec = float(
        row["window_start_sec"]
    )

    end_sec = (
        start_sec
        + WINDOW_SEC
    )

    # --------------------------------------------------------
    # Waveforms
    # --------------------------------------------------------

    values = load_waveforms_with_retry(
        caseid
    )

    start_index = int(
        np.floor(
            start_sec * FS
        )
    )

    end_index = int(
        np.floor(
            end_sec * FS
        )
    )

    if start_index < 0:
        raise ValueError(
            "negative waveform start"
        )

    if end_index > len(values):
        raise ValueError(
            "waveform shorter than "
            "first 20-min window"
        )

    window = values[
        start_index:end_index,
        :,
    ]

    expected_samples = (
        WINDOW_SEC * FS
    )

    if (
        len(window)
        != expected_samples
    ):
        raise ValueError(
            "unexpected waveform length: "
            f"{len(window)} "
            f"!= {expected_samples}"
        )

    pleth_raw = window[:, 0]
    art_raw = window[:, 1]

    pleth, pleth_missing = (
        interpolate_waveform(
            pleth_raw
        )
    )

    art, art_missing = (
        interpolate_waveform(
            art_raw
        )
    )

    pleth_coverage = (
        1.0
        - pleth_missing
    )

    art_coverage = (
        1.0
        - art_missing
    )

    if not np.all(
        np.isfinite(pleth)
    ):
        raise ValueError(
            "PLETH contains "
            "non-finite values"
        )

    if not np.all(
        np.isfinite(art)
    ):
        raise ValueError(
            "ART contains "
            "non-finite values"
        )

    # --------------------------------------------------------
    # Peaks
    # --------------------------------------------------------

    art_peaks, art_filtered = (
        detect_pulse_peaks(
            art,
            FS,
            "ART",
        )
    )

    (
        pleth_peaks,
        pleth_filtered,
    ) = detect_pulse_peaks(
        pleth,
        FS,
        "PLETH",
    )

    art_rate = rate_from_peaks(
        art_peaks,
        FS,
    )

    pleth_rate = rate_from_peaks(
        pleth_peaks,
        FS,
    )

    if (
        np.isfinite(art_rate)
        and np.isfinite(
            pleth_rate
        )
    ):

        rate_diff = abs(
            art_rate
            - pleth_rate
        )

    else:

        rate_diff = np.nan

    # --------------------------------------------------------
    # ART -> PLETH pairing
    # --------------------------------------------------------

    pairs = pair_art_pleth(
        art_peaks,
        pleth_peaks,
        FS,
        min_lag=0.0,
        max_lag=0.8,
    )

    lag_values = np.asarray(
        [
            (p - a) / FS
            for a, p in pairs
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Beat amplitude coupling
    # --------------------------------------------------------

    art_amplitudes = []
    pleth_amplitudes = []

    for (
        art_peak,
        pleth_peak,
    ) in pairs:

        art_amp = (
            local_pulse_amplitude(
                art,
                art_peak,
                FS,
            )
        )

        pleth_amp = (
            local_pulse_amplitude(
                pleth,
                pleth_peak,
                FS,
            )
        )

        if (
            np.isfinite(art_amp)
            and np.isfinite(
                pleth_amp
            )
        ):

            art_amplitudes.append(
                art_amp
            )

            pleth_amplitudes.append(
                pleth_amp
            )

    if len(
        art_amplitudes
    ) >= 30:

        amplitude_rho = (
            pd.Series(
                art_amplitudes
            )
            .corr(
                pd.Series(
                    pleth_amplitudes
                ),
                method="spearman",
            )
        )

    else:

        amplitude_rho = np.nan

    # --------------------------------------------------------
    # 60-sec cross-correlation
    # --------------------------------------------------------

    xcorr_values = []
    xcorr_lag_values = []

    xcorr_window_samples = (
        60 * FS
    )

    for local_start in range(
        0,
        len(art_filtered)
        - xcorr_window_samples
        + 1,
        xcorr_window_samples,
    ):

        local_end = (
            local_start
            + xcorr_window_samples
        )

        (
            corr_value,
            corr_lag,
        ) = positive_lag_xcorr(
            art_filtered[
                local_start:local_end
            ],
            pleth_filtered[
                local_start:local_end
            ],
            FS,
            max_lag_sec=0.8,
        )

        if (
            np.isfinite(
                corr_value
            )
            and np.isfinite(
                corr_lag
            )
        ):

            xcorr_values.append(
                corr_value
            )

            xcorr_lag_values.append(
                corr_lag
            )

    xcorr_values = np.asarray(
        xcorr_values,
        dtype=float,
    )

    xcorr_lag_values = np.asarray(
        xcorr_lag_values,
        dtype=float,
    )

    # --------------------------------------------------------
    # Numeric signals
    # --------------------------------------------------------

    hr = load_numeric_window(
        caseid,
        "HR",
        start_sec,
        end_sec,
        20,
        250,
    )

    map_values = (
        load_numeric_window(
            caseid,
            "MAP",
            start_sec,
            end_sec,
            20,
            200,
        )
    )

    etco2 = load_numeric_window(
        caseid,
        "ETCO2",
        start_sec,
        end_sec,
        5,
        80,
    )

    if (
        hr is None
        or map_values is None
        or etco2 is None
    ):

        raise ValueError(
            "numeric cache missing"
        )

    hr_map_corr = (
        rolling_spearman(
            hr,
            map_values,
        )
    )

    map_etco2_corr = (
        rolling_spearman(
            map_values,
            etco2,
        )
    )

    hr_etco2_corr = (
        rolling_spearman(
            hr,
            etco2,
        )
    )

    # --------------------------------------------------------
    # 20-min sensitivity QC
    # --------------------------------------------------------

    qc_reasons = []

    if (
        art_coverage
        < MIN_WAVE_COVERAGE
    ):
        qc_reasons.append(
            "art_coverage_lt80pct"
        )

    if (
        pleth_coverage
        < MIN_WAVE_COVERAGE
    ):
        qc_reasons.append(
            "pleth_coverage_lt80pct"
        )

    if (
        not np.isfinite(
            rate_diff
        )
        or rate_diff
        > MAX_RATE_DIFF_BPM
    ):
        qc_reasons.append(
            "art_pleth_rate_diff_gt8bpm"
        )

    if (
        len(lag_values)
        < MIN_LAG_PAIRS
    ):
        qc_reasons.append(
            "lag_pairs_lt334"
        )

    if (
        len(xcorr_values)
        < MIN_XCORR_WINDOWS
    ):
        qc_reasons.append(
            "xcorr_windows_lt10"
        )

    if (
        len(hr_map_corr)
        < MIN_NUMERIC_WINDOWS
    ):
        qc_reasons.append(
            "hr_map_windows_lt4"
        )

    if (
        len(map_etco2_corr)
        < MIN_NUMERIC_WINDOWS
    ):
        qc_reasons.append(
            "map_etco2_windows_lt4"
        )

    if (
        len(hr_etco2_corr)
        < MIN_NUMERIC_WINDOWS
    ):
        qc_reasons.append(
            "hr_etco2_windows_lt4"
        )

    qc_pass = (
        len(qc_reasons)
        == 0
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    result = {
        "caseid":
            caseid,

        "subjectid":
            subjectid,

        "window_start_sec":
            start_sec,

        "window_end_sec":
            end_sec,

        "sensitivity_window_min":
            20,

        "art_coverage":
            art_coverage,

        "pleth_coverage":
            pleth_coverage,

        "art_rate_bpm":
            art_rate,

        "pleth_rate_bpm":
            pleth_rate,

        "art_pleth_rate_diff_bpm":
            rate_diff,

        "art_pleth_pairs_n":
            len(lag_values),

        "art_pleth_xcorr_windows_n":
            len(xcorr_values),

        "hr_map_windows_n":
            len(hr_map_corr),

        "map_etco2_windows_n":
            len(map_etco2_corr),

        "hr_etco2_windows_n":
            len(hr_etco2_corr),

        "art_pleth_lag_median_sec":
            (
                float(
                    np.median(
                        lag_values
                    )
                )
                if len(
                    lag_values
                )
                else np.nan
            ),

        "art_pleth_lag_iqr_sec":
            calc_iqr(
                lag_values
            ),

        "art_pleth_xcorr_median":
            (
                float(
                    np.median(
                        xcorr_values
                    )
                )
                if len(
                    xcorr_values
                )
                else np.nan
            ),

        "art_pleth_xcorr_iqr":
            calc_iqr(
                xcorr_values
            ),

        "art_pleth_xcorr_lag_median_sec":
            (
                float(
                    np.median(
                        xcorr_lag_values
                    )
                )
                if len(
                    xcorr_lag_values
                )
                else np.nan
            ),

        "art_pleth_amplitude_spearman":
            amplitude_rho,

        "hr_map_corr_median":
            (
                float(
                    np.median(
                        hr_map_corr
                    )
                )
                if len(
                    hr_map_corr
                )
                else np.nan
            ),

        "hr_map_corr_iqr":
            calc_iqr(
                hr_map_corr
            ),

        "map_etco2_corr_median":
            (
                float(
                    np.median(
                        map_etco2_corr
                    )
                )
                if len(
                    map_etco2_corr
                )
                else np.nan
            ),

        "map_etco2_corr_iqr":
            calc_iqr(
                map_etco2_corr
            ),

        "hr_etco2_corr_median":
            (
                float(
                    np.median(
                        hr_etco2_corr
                    )
                )
                if len(
                    hr_etco2_corr
                )
                else np.nan
            ),

        "hr_etco2_corr_iqr":
            calc_iqr(
                hr_etco2_corr
            ),

        "qc_pass_20min":
            int(
                qc_pass
            ),

        "qc_reason_20min":
            ";".join(
                qc_reasons
            ),
    }

    return result


# ============================================================
# 7. LOAD PRIMARY FROZEN COHORT
# ============================================================

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        "Primary frozen cohort "
        f"not found:\n{INPUT_PATH}"
    )

cohort = pd.read_csv(
    INPUT_PATH
)

cohort.columns = (
    cohort.columns
    .str.replace(
        "\ufeff",
        "",
        regex=False,
    )
    .str.strip()
)

required = [
    "caseid",
    "subjectid",
    "window_start_sec",
]

missing = [
    c
    for c in required
    if c not in cohort.columns
]

if missing:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            missing
        )
    )

cohort["caseid"] = pd.to_numeric(
    cohort["caseid"],
    errors="coerce",
)

cohort[
    "window_start_sec"
] = pd.to_numeric(
    cohort[
        "window_start_sec"
    ],
    errors="coerce",
)

cohort = cohort.dropna(
    subset=[
        "caseid",
        "window_start_sec",
    ]
).copy()

cohort["caseid"] = (
    cohort["caseid"]
    .astype(int)
)

cohort = (
    cohort
    .sort_values("caseid")
    .reset_index(drop=True)
)

print("\n" + "=" * 96)
print("PRIMARY COHORT")
print("=" * 96)

print(
    f"Primary 30-min frozen cases: "
    f"{len(cohort):,}"
)

print(
    f"Unique subjects: "
    f"{cohort['subjectid'].nunique():,}"
)

print(
    "Sensitivity window: "
    "first 20 min"
)

print(
    "Outcome variables used: NO"
)


# ============================================================
# 8. RESUME CHECKPOINT
# ============================================================

result_rows = []
error_rows = []

processed_caseids = set()

if CHECKPOINT_PATH.exists():

    try:

        checkpoint = pd.read_csv(
            CHECKPOINT_PATH
        )

        if len(checkpoint) > 0:

            checkpoint["caseid"] = (
                pd.to_numeric(
                    checkpoint[
                        "caseid"
                    ],
                    errors="coerce",
                )
            )

            checkpoint = (
                checkpoint
                .dropna(
                    subset=[
                        "caseid"
                    ]
                )
                .copy()
            )

            checkpoint[
                "caseid"
            ] = (
                checkpoint[
                    "caseid"
                ]
                .astype(int)
            )

            result_rows = (
                checkpoint
                .to_dict(
                    "records"
                )
            )

            processed_caseids = set(
                checkpoint[
                    "caseid"
                ].tolist()
            )

            print(
                "\nCheckpoint loaded: "
                f"{len(processed_caseids):,} "
                "cases"
            )

    except Exception as error:

        print(
            "\nCheckpoint load "
            f"failed: {error}"
        )

        result_rows = []
        processed_caseids = set()


if ERROR_PATH.exists():

    try:

        old_errors = pd.read_csv(
            ERROR_PATH
        )

        if len(old_errors) > 0:

            error_rows = (
                old_errors
                .to_dict(
                    "records"
                )
            )

    except Exception:

        error_rows = []


remaining = cohort.loc[
    ~cohort[
        "caseid"
    ].isin(
        processed_caseids
    )
].copy()

print(
    f"Remaining cases: "
    f"{len(remaining):,}"
)


# ============================================================
# 9. EXTRACTION
# ============================================================

print("\n" + "=" * 96)
print("20-MIN EXTRACTION START")
print("=" * 96)

run_success = 0
run_errors = 0
run_qc_pass = 0

total_remaining = len(
    remaining
)

for run_index, (
    _,
    row,
) in enumerate(
    remaining.iterrows(),
    start=1,
):

    caseid = int(
        row["caseid"]
    )

    try:

        result = extract_one_case(
            row
        )

        result_rows.append(
            result
        )

        run_success += 1

        run_qc_pass += int(
            result[
                "qc_pass_20min"
            ]
        )

    except Exception as error:

        run_errors += 1

        error_rows.append({
            "caseid":
                caseid,

            "error":
                str(error),

            "traceback":
                traceback.format_exc(),
        })

    # Progress
    if (
        run_index
        % PRINT_EVERY
        == 0
        or run_index
        == total_remaining
    ):

        print(
            f"{run_index:,}"
            f"/{total_remaining:,}"
            " | "
            f"success={run_success:,}"
            " | "
            f"QC pass="
            f"{run_qc_pass:,}"
            " | "
            f"errors={run_errors:,}"
        )

    # Checkpoint
    if (
        run_index
        % CHECKPOINT_EVERY
        == 0
        or run_index
        == total_remaining
    ):

        pd.DataFrame(
            result_rows
        ).to_csv(
            CHECKPOINT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(
            error_rows
        ).to_csv(
            ERROR_PATH,
            index=False,
            encoding="utf-8-sig",
        )


# ============================================================
# 10. FINAL DATA
# ============================================================

result_df = pd.DataFrame(
    result_rows
)

if len(result_df) == 0:

    raise RuntimeError(
        "No successful "
        "20-min extractions."
    )

result_df = (
    result_df
    .drop_duplicates(
        subset=["caseid"],
        keep="last",
    )
    .sort_values("caseid")
    .reset_index(drop=True)
)

qc_pass_df = result_df.loc[
    result_df[
        "qc_pass_20min"
    ] == 1
].copy()

error_df = pd.DataFrame(
    error_rows
)


# ============================================================
# 11. FEATURE FINITE CHECK
# ============================================================

for feature in LOCKED_FEATURES:

    result_df[
        feature
    ] = pd.to_numeric(
        result_df[
            feature
        ],
        errors="coerce",
    )


# ============================================================
# 12. QC REASON COUNTS
# ============================================================

reason_counts = {}

for reasons in result_df.loc[
    result_df[
        "qc_pass_20min"
    ] == 0,
    "qc_reason_20min",
].fillna(""):

    for reason in str(
        reasons
    ).split(";"):

        reason = reason.strip()

        if not reason:
            continue

        reason_counts[
            reason
        ] = (
            reason_counts.get(
                reason,
                0,
            )
            + 1
        )

qc_reason_df = pd.DataFrame(
    [
        {
            "reason":
                key,
            "n_cases":
                value,
        }
        for (
            key,
            value,
        ) in reason_counts.items()
    ]
)

if len(qc_reason_df) > 0:

    qc_reason_df = (
        qc_reason_df
        .sort_values(
            "n_cases",
            ascending=False,
        )
    )


# ============================================================
# 13. FEATURE DISTRIBUTIONS
# ============================================================

distribution_rows = []

for feature in LOCKED_FEATURES:

    vals = (
        pd.to_numeric(
            qc_pass_df[
                feature
            ],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    distribution_rows.append({
        "feature":
            feature,

        "n_nonmissing":
            len(vals),

        "missing_n":
            len(
                qc_pass_df
            )
            - len(vals),

        "median":
            (
                float(
                    vals.median()
                )
                if len(vals)
                else np.nan
            ),

        "q1":
            (
                float(
                    vals.quantile(
                        0.25
                    )
                )
                if len(vals)
                else np.nan
            ),

        "q3":
            (
                float(
                    vals.quantile(
                        0.75
                    )
                )
                if len(vals)
                else np.nan
            ),
    })

distribution_df = (
    pd.DataFrame(
        distribution_rows
    )
)


# ============================================================
# 14. SAVE
# ============================================================

result_df.to_csv(
    OUTPUT_ALL_PATH,
    index=False,
    encoding="utf-8-sig",
)

qc_pass_df.to_csv(
    OUTPUT_QC_PASS_PATH,
    index=False,
    encoding="utf-8-sig",
)

error_df.to_csv(
    ERROR_PATH,
    index=False,
    encoding="utf-8-sig",
)

qc_reason_df.to_csv(
    QC_REASON_PATH,
    index=False,
    encoding="utf-8-sig",
)

distribution_df.to_csv(
    DISTRIBUTION_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 15. SUMMARY
# ============================================================

primary_n = len(cohort)

success_n = len(
    result_df
)

qc_pass_n = len(
    qc_pass_df
)

qc_fail_n = (
    success_n
    - qc_pass_n
)

qc_pass_subjects = int(
    qc_pass_df[
        "subjectid"
    ].nunique()
)

summary_df = pd.DataFrame([
    {
        "primary_30min_frozen_cases":
            primary_n,

        "successful_20min_extractions":
            success_n,

        "20min_qc_pass_cases":
            qc_pass_n,

        "20min_qc_fail_cases":
            qc_fail_n,

        "20min_qc_pass_percent":
            (
                qc_pass_n
                / success_n
                * 100
                if success_n > 0
                else np.nan
            ),

        "20min_qc_pass_unique_subjects":
            qc_pass_subjects,

        "sensitivity_window_min":
            20,

        "waveform_resample_hz":
            FS,

        "min_lag_pairs":
            MIN_LAG_PAIRS,

        "min_xcorr_windows":
            MIN_XCORR_WINDOWS,

        "min_numeric_windows":
            MIN_NUMERIC_WINDOWS,

        "coupling_feature_count":
            len(
                LOCKED_FEATURES
            ),
    }
])

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 16. PRINT RESULTS
# ============================================================

print("\n" + "=" * 96)
print("STEP 14A SUMMARY")
print("=" * 96)

print(
    f"\nPrimary frozen cases: "
    f"{primary_n:,}"
)

print(
    f"Successful extractions: "
    f"{success_n:,}"
)

print(
    f"20-min QC PASS: "
    f"{qc_pass_n:,}"
)

print(
    f"20-min QC FAIL: "
    f"{qc_fail_n:,}"
)

if success_n > 0:

    print(
        f"20-min QC PASS rate: "
        f"{qc_pass_n / success_n * 100:.1f}%"
    )

print(
    f"QC PASS unique subjects: "
    f"{qc_pass_subjects:,}"
)

print("\n20-MIN QC EXCLUSION REASONS")
print("-" * 96)

if len(qc_reason_df) > 0:

    print(
        qc_reason_df
        .to_string(
            index=False
        )
    )

else:

    print(
        "No QC exclusions."
    )

print("\n20-MIN FEATURE DISTRIBUTIONS")
print("-" * 96)

print(
    distribution_df
    .to_string(
        index=False
    )
)

print("\n" + "=" * 96)
print("SAVED FILES")
print("=" * 96)

print(
    OUTPUT_ALL_PATH
)

print(
    OUTPUT_QC_PASS_PATH
)

print(
    SUMMARY_PATH
)

print(
    QC_REASON_PATH
)

print(
    DISTRIBUTION_PATH
)

print(
    ERROR_PATH
)

print("\nSTEP 14A COMPLETE")