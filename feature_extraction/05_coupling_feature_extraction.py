from pathlib import Path
import time
import warnings
import traceback

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, correlate, correlation_lags
import vitaldb

warnings.filterwarnings('ignore')

# ============================================================
# VitalDB Multimodal Physiological Coupling Study
# STEP 05 - Full Cohort Extraction of LOCKED Coupling Features
# ============================================================
#
# IMPORTANT
# ------------------------------------------------------------
# This script MUST use the exact feature definitions and QC
# rules locked in STEP 04, before any outcome analysis.
#
# Primary analysis window:
#   first 30 min after operation start
#
# Primary waveform coupling:
#   ART <-> PLETH
#
# Primary numeric coupling:
#   HR <-> MAP
#   MAP <-> ETCO2
#   HR <-> ETCO2
#
# ECG waveform timing remains excluded from the PRIMARY
# locked feature set.
#
# No postoperative outcome is used during feature extraction.
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / 'data_raw'
PROCESSED_DIR = PROJECT_DIR / 'data_processed'
RESULT_DIR = PROJECT_DIR / 'results'
NUMERIC_CACHE_DIR = RAW_DIR / 'numeric_tracks'

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. INPUT / OUTPUT FILES
# ============================================================

INPUT_PATH = PROCESSED_DIR / '02_numeric_quality_eligible_cases.csv'
LOCKED_FEATURE_DEF_PATH = RESULT_DIR / '04_locked_primary_feature_definition.csv'

CHECKPOINT_PATH = RESULT_DIR / '05_locked_coupling_features_checkpoint.csv'
OUTPUT_ALL_PATH = PROCESSED_DIR / '05_locked_coupling_features_all_cases.csv'
OUTPUT_QC_PASS_PATH = PROCESSED_DIR / '05_locked_coupling_features_qc_pass.csv'
ERROR_PATH = RESULT_DIR / '05_locked_coupling_feature_errors.csv'
QC_REASON_PATH = RESULT_DIR / '05_locked_coupling_qc_reason_counts.csv'
SUMMARY_PATH = RESULT_DIR / '05_locked_coupling_feature_summary.csv'
FEATURE_DISTRIBUTION_PATH = RESULT_DIR / '05_locked_feature_distributions.csv'


# ============================================================
# 3. LOCKED SETTINGS FROM STEP 04
# ============================================================

FS = 50
INTERVAL = 1.0 / FS
WINDOW_SEC = 30 * 60
TRACK_NAMES = ['SNUADC/PLETH', 'SNUADC/ART']

MIN_WAVE_COVERAGE = 0.80
MAX_RATE_DIFF_BPM = 8.0
MIN_LAG_PAIRS = 500
MIN_XCORR_WINDOWS = 15

ROLLING_WINDOW_SEC = 5 * 60
ROLLING_STEP_SEC = 150
MIN_NUMERIC_WINDOW_COVERAGE = 0.80

# Network retry for VitalDB waveform loading
MAX_RETRIES = 3
RETRY_WAIT_SEC = 3

# Save progress frequently because full-cohort extraction is long
CHECKPOINT_EVERY = 10
PRINT_EVERY = 10


# ============================================================
# 4. LOCKED FEATURE LIST
# ============================================================

LOCKED_FEATURES = [
    'art_pleth_lag_median_sec',
    'art_pleth_lag_iqr_sec',
    'art_pleth_xcorr_median',
    'art_pleth_xcorr_iqr',
    'art_pleth_xcorr_lag_median_sec',
    'art_pleth_amplitude_spearman',
    'hr_map_corr_median',
    'hr_map_corr_iqr',
    'map_etco2_corr_median',
    'map_etco2_corr_iqr',
    'hr_etco2_corr_median',
    'hr_etco2_corr_iqr',
]


# ============================================================
# 5. HELPER FUNCTIONS - IDENTICAL TO STEP 04
# ============================================================

def calc_iqr(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def bandpass_filter(signal, fs, lowcut, highcut, order=3):
    x = np.asarray(signal, dtype=float)
    nyquist = fs / 2.0
    sos = butter(
        order,
        [lowcut / nyquist, highcut / nyquist],
        btype='bandpass',
        output='sos'
    )
    return sosfiltfilt(sos, x)


def interpolate_waveform(signal):
    series = pd.Series(signal, dtype=float)
    missing_fraction = float(series.isna().mean())
    series = series.interpolate(method='linear', limit_direction='both')
    return series.to_numpy(), missing_fraction


def detect_pulse_peaks(signal, fs, signal_type):
    highcut = 12.0 if signal_type == 'ART' else 8.0
    filtered = bandpass_filter(signal, fs, 0.5, highcut, order=3)

    positive_strength = np.percentile(filtered, 99)
    negative_strength = abs(np.percentile(filtered, 1))
    detection = -filtered if negative_strength > positive_strength else filtered

    median_value = np.median(detection)
    mad = np.median(np.abs(detection - median_value))
    prominence = max(1.5 * mad, np.std(detection) * 0.10, 1e-8)

    peaks, _ = find_peaks(
        detection,
        distance=int(0.35 * fs),
        prominence=prominence
    )

    return peaks, filtered


def rate_from_peaks(peaks, fs):
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 3:
        return np.nan

    intervals = np.diff(peaks) / fs
    intervals = intervals[(intervals >= 0.30) & (intervals <= 2.0)]

    if len(intervals) < 2:
        return np.nan

    return float(60.0 / np.median(intervals))


def pair_art_pleth(art_peaks, pleth_peaks, fs, min_lag=0.0, max_lag=0.8):
    art_peaks = np.asarray(art_peaks, dtype=int)
    pleth_peaks = np.asarray(pleth_peaks, dtype=int)

    pairs = []
    j = 0

    for art_peak in art_peaks:
        min_target = art_peak + int(round(min_lag * fs))
        max_target = art_peak + int(round(max_lag * fs))

        while j < len(pleth_peaks) and pleth_peaks[j] < min_target:
            j += 1

        if j >= len(pleth_peaks):
            break

        if pleth_peaks[j] <= max_target:
            pairs.append((art_peak, pleth_peaks[j]))
            j += 1

    return pairs


def local_pulse_amplitude(signal, peak_index, fs, pre_sec=0.6):
    start = max(0, int(peak_index - pre_sec * fs))
    if peak_index <= start:
        return np.nan

    segment = signal[start:peak_index + 1]
    if len(segment) == 0:
        return np.nan

    baseline = np.nanmin(segment)
    return float(signal[peak_index] - baseline)


def positive_lag_xcorr(source, target, fs, max_lag_sec=0.8):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if np.std(source) == 0 or np.std(target) == 0:
        return np.nan, np.nan

    source_z = (source - np.mean(source)) / np.std(source)
    target_z = (target - np.mean(target)) / np.std(target)

    corr = correlate(target_z, source_z, mode='full', method='fft')
    lags = correlation_lags(len(target_z), len(source_z), mode='full')

    max_lag_samples = int(max_lag_sec * fs)
    mask = (lags >= 0) & (lags <= max_lag_samples)

    corr = corr[mask] / len(source_z)
    lags = lags[mask]

    if len(corr) == 0:
        return np.nan, np.nan

    best = int(np.argmax(corr))
    return float(corr[best]), float(lags[best] / fs)


def load_numeric_window(caseid, signal_name, start_sec, end_sec, min_value, max_value):
    path = (
        NUMERIC_CACHE_DIR
        / signal_name
        / f'case_{caseid:05d}_{signal_name}.csv.gz'
    )

    if not path.exists():
        return None

    data = pd.read_csv(path)

    if 'time_sec' not in data.columns or 'value' not in data.columns:
        return None

    data['time_sec'] = pd.to_numeric(data['time_sec'], errors='coerce')
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    data = data.dropna(subset=['time_sec'])

    data = data.loc[
        (data['time_sec'] >= start_sec)
        & (data['time_sec'] < end_sec)
    ].copy()

    data.loc[
        (data['value'] < min_value) | (data['value'] > max_value),
        'value'
    ] = np.nan

    if len(data) > 0:
        data['second'] = np.floor(data['time_sec']).astype(int)
        per_second = data.groupby('second')['value'].median()
    else:
        per_second = pd.Series(dtype=float)

    full_index = np.arange(
        int(np.floor(start_sec)),
        int(np.floor(end_sec)),
        dtype=int
    )

    series = per_second.reindex(full_index)
    series = series.interpolate(
        method='linear',
        limit=10,
        limit_area='inside'
    )

    return series.reset_index(drop=True)


def rolling_spearman(series_a, series_b):
    a = pd.Series(series_a, dtype=float).reset_index(drop=True)
    b = pd.Series(series_b, dtype=float).reset_index(drop=True)

    n = min(len(a), len(b))
    correlations = []

    if n < ROLLING_WINDOW_SEC:
        return np.asarray([], dtype=float)

    for start in range(0, n - ROLLING_WINDOW_SEC + 1, ROLLING_STEP_SEC):
        end = start + ROLLING_WINDOW_SEC
        x = a.iloc[start:end]
        y = b.iloc[start:end]

        valid = x.notna() & y.notna()
        if valid.mean() < MIN_NUMERIC_WINDOW_COVERAGE:
            continue

        x_valid = x[valid]
        y_valid = y[valid]

        if x_valid.nunique() < 3 or y_valid.nunique() < 3:
            continue

        rho = x_valid.corr(y_valid, method='spearman')
        if pd.notna(rho):
            correlations.append(float(rho))

    return np.asarray(correlations, dtype=float)


# ============================================================
# 6. VALIDATE LOCKED FEATURE DEFINITION
# ============================================================

def validate_locked_feature_file():
    if not LOCKED_FEATURE_DEF_PATH.exists():
        raise FileNotFoundError(
            'STEP 04 locked feature definition file not found:\n'
            f'{LOCKED_FEATURE_DEF_PATH}'
        )

    locked = pd.read_csv(LOCKED_FEATURE_DEF_PATH)
    locked.columns = locked.columns.str.replace('\ufeff', '', regex=False).str.strip()

    if 'feature' not in locked.columns:
        raise ValueError('04_locked_primary_feature_definition.csv에 feature column 없음')

    actual = locked['feature'].astype(str).tolist()

    if actual != LOCKED_FEATURES:
        raise ValueError(
            'Locked feature list mismatch.\n'
            'STEP 04 이후 feature 정의를 변경하면 안 됩니다.\n\n'
            f'Expected:\n{LOCKED_FEATURES}\n\n'
            f'Found:\n{actual}'
        )

    return locked


# ============================================================
# 7. WAVEFORM LOADING WITH RETRY
# ============================================================

def load_waveforms_with_retry(caseid):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            values = vitaldb.load_case(
                caseid,
                TRACK_NAMES,
                interval=INTERVAL
            )

            values = np.asarray(values, dtype=float)

            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError(f'unexpected waveform shape: {values.shape}')

            return values

        except Exception as error:
            last_error = error
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SEC * attempt)

    raise RuntimeError(f'waveform load failed after {MAX_RETRIES} attempts: {last_error}')


# ============================================================
# 8. EXTRACT ONE CASE
# ============================================================

def extract_one_case(row):
    caseid = int(row['caseid'])
    start_sec = float(row['opstart_sec'])
    end_sec = start_sec + WINDOW_SEC

    # --------------------------------------------------------
    # WAVEFORMS
    # --------------------------------------------------------
    values = load_waveforms_with_retry(caseid)

    start_index = int(np.floor(start_sec * FS))
    end_index = int(np.floor(end_sec * FS))

    if start_index < 0:
        raise ValueError('negative waveform start index')

    if end_index > len(values):
        raise ValueError('waveform shorter than locked first 30-min operation window')

    window = values[start_index:end_index, :]

    expected_samples = WINDOW_SEC * FS
    if len(window) != expected_samples:
        raise ValueError(
            f'unexpected waveform window length: {len(window)} != {expected_samples}'
        )

    pleth_raw = window[:, 0]
    art_raw = window[:, 1]

    pleth, pleth_missing = interpolate_waveform(pleth_raw)
    art, art_missing = interpolate_waveform(art_raw)

    pleth_coverage = 1.0 - pleth_missing
    art_coverage = 1.0 - art_missing

    if not np.all(np.isfinite(pleth)):
        raise ValueError('PLETH contains non-finite values after interpolation')

    if not np.all(np.isfinite(art)):
        raise ValueError('ART contains non-finite values after interpolation')

    art_peaks, art_filtered = detect_pulse_peaks(art, FS, 'ART')
    pleth_peaks, pleth_filtered = detect_pulse_peaks(pleth, FS, 'PLETH')

    art_rate = rate_from_peaks(art_peaks, FS)
    pleth_rate = rate_from_peaks(pleth_peaks, FS)

    if np.isfinite(art_rate) and np.isfinite(pleth_rate):
        rate_diff = abs(art_rate - pleth_rate)
    else:
        rate_diff = np.nan

    # --------------------------------------------------------
    # ART -> PLETH BEAT PAIRING
    # --------------------------------------------------------
    pairs = pair_art_pleth(
        art_peaks,
        pleth_peaks,
        FS,
        min_lag=0.0,
        max_lag=0.8
    )

    lag_values = np.asarray(
        [(p - a) / FS for a, p in pairs],
        dtype=float
    )

    # --------------------------------------------------------
    # BEAT AMPLITUDE COUPLING
    # --------------------------------------------------------
    art_amplitudes = []
    pleth_amplitudes = []

    for art_peak, pleth_peak in pairs:
        art_amp = local_pulse_amplitude(art, art_peak, FS)
        pleth_amp = local_pulse_amplitude(pleth, pleth_peak, FS)

        if np.isfinite(art_amp) and np.isfinite(pleth_amp):
            art_amplitudes.append(art_amp)
            pleth_amplitudes.append(pleth_amp)

    if len(art_amplitudes) >= 30:
        amplitude_rho = pd.Series(art_amplitudes).corr(
            pd.Series(pleth_amplitudes),
            method='spearman'
        )
    else:
        amplitude_rho = np.nan

    # --------------------------------------------------------
    # 60-sec ART-PLETH CROSS-CORRELATION WINDOWS
    # --------------------------------------------------------
    xcorr_values = []
    xcorr_lag_values = []
    xcorr_window_samples = 60 * FS

    for local_start in range(
        0,
        len(art_filtered) - xcorr_window_samples + 1,
        xcorr_window_samples
    ):
        local_end = local_start + xcorr_window_samples

        corr_value, corr_lag = positive_lag_xcorr(
            art_filtered[local_start:local_end],
            pleth_filtered[local_start:local_end],
            FS,
            max_lag_sec=0.8
        )

        if np.isfinite(corr_value) and np.isfinite(corr_lag):
            xcorr_values.append(corr_value)
            xcorr_lag_values.append(corr_lag)

    xcorr_values = np.asarray(xcorr_values, dtype=float)
    xcorr_lag_values = np.asarray(xcorr_lag_values, dtype=float)

    # --------------------------------------------------------
    # NUMERIC SIGNALS FROM STEP 02 CACHE
    # --------------------------------------------------------
    hr = load_numeric_window(caseid, 'HR', start_sec, end_sec, 20, 250)
    map_values = load_numeric_window(caseid, 'MAP', start_sec, end_sec, 20, 200)
    etco2 = load_numeric_window(caseid, 'ETCO2', start_sec, end_sec, 5, 80)

    if hr is None or map_values is None or etco2 is None:
        raise ValueError('numeric cache missing')

    hr_map_corr = rolling_spearman(hr, map_values)
    map_etco2_corr = rolling_spearman(map_values, etco2)
    hr_etco2_corr = rolling_spearman(hr, etco2)

    # --------------------------------------------------------
    # LOCKED QC - IDENTICAL TO STEP 04
    # --------------------------------------------------------
    qc_reasons = []

    if art_coverage < MIN_WAVE_COVERAGE:
        qc_reasons.append('art_coverage_lt80pct')

    if pleth_coverage < MIN_WAVE_COVERAGE:
        qc_reasons.append('pleth_coverage_lt80pct')

    if not np.isfinite(rate_diff) or rate_diff > MAX_RATE_DIFF_BPM:
        qc_reasons.append('art_pleth_rate_diff_gt8bpm')

    if len(lag_values) < MIN_LAG_PAIRS:
        qc_reasons.append('lag_pairs_lt500')

    if len(xcorr_values) < MIN_XCORR_WINDOWS:
        qc_reasons.append('xcorr_windows_lt15')

    if len(hr_map_corr) < 6:
        qc_reasons.append('hr_map_windows_lt6')

    if len(map_etco2_corr) < 6:
        qc_reasons.append('map_etco2_windows_lt6')

    if len(hr_etco2_corr) < 6:
        qc_reasons.append('hr_etco2_windows_lt6')

    qc_pass = len(qc_reasons) == 0

    # --------------------------------------------------------
    # LOCKED FEATURES
    # --------------------------------------------------------
    result = {
        'caseid': caseid,
        'subjectid': row.get('subjectid', np.nan),
        'window_start_sec': start_sec,
        'window_end_sec': end_sec,
        'art_coverage': art_coverage,
        'pleth_coverage': pleth_coverage,
        'art_rate_bpm': art_rate,
        'pleth_rate_bpm': pleth_rate,
        'art_pleth_rate_diff_bpm': rate_diff,
        'art_pleth_pairs_n': len(lag_values),
        'art_pleth_xcorr_windows_n': len(xcorr_values),
        'hr_map_windows_n': len(hr_map_corr),
        'map_etco2_windows_n': len(map_etco2_corr),
        'hr_etco2_windows_n': len(hr_etco2_corr),

        'art_pleth_lag_median_sec': (
            float(np.median(lag_values)) if len(lag_values) else np.nan
        ),
        'art_pleth_lag_iqr_sec': calc_iqr(lag_values),
        'art_pleth_xcorr_median': (
            float(np.median(xcorr_values)) if len(xcorr_values) else np.nan
        ),
        'art_pleth_xcorr_iqr': calc_iqr(xcorr_values),
        'art_pleth_xcorr_lag_median_sec': (
            float(np.median(xcorr_lag_values)) if len(xcorr_lag_values) else np.nan
        ),
        'art_pleth_amplitude_spearman': amplitude_rho,
        'hr_map_corr_median': (
            float(np.median(hr_map_corr)) if len(hr_map_corr) else np.nan
        ),
        'hr_map_corr_iqr': calc_iqr(hr_map_corr),
        'map_etco2_corr_median': (
            float(np.median(map_etco2_corr)) if len(map_etco2_corr) else np.nan
        ),
        'map_etco2_corr_iqr': calc_iqr(map_etco2_corr),
        'hr_etco2_corr_median': (
            float(np.median(hr_etco2_corr)) if len(hr_etco2_corr) else np.nan
        ),
        'hr_etco2_corr_iqr': calc_iqr(hr_etco2_corr),

        'qc_pass': int(qc_pass),
        'qc_reason': ';'.join(qc_reasons),
    }

    return result


# ============================================================
# 9. LOAD / VALIDATE STUDY COHORT
# ============================================================

print('=' * 90)
print('VitalDB Multimodal Physiological Coupling Study')
print('STEP 05 - Full Cohort Locked Coupling Feature Extraction')
print('=' * 90)

locked_def = validate_locked_feature_file()

print('\nLocked primary features verified:')
for i, feature in enumerate(LOCKED_FEATURES, start=1):
    print(f'{i:02d}. {feature}')

if not INPUT_PATH.exists():
    raise FileNotFoundError(f'입력 파일 없음: {INPUT_PATH}')

cohort = pd.read_csv(INPUT_PATH)
cohort.columns = cohort.columns.str.replace('\ufeff', '', regex=False).str.strip()

required_columns = ['caseid', 'opstart_sec', 'opend_sec']
for col in required_columns:
    if col not in cohort.columns:
        raise ValueError(f'필수 column 없음: {col}')

cohort['caseid'] = pd.to_numeric(cohort['caseid'], errors='coerce')
cohort['opstart_sec'] = pd.to_numeric(cohort['opstart_sec'], errors='coerce')
cohort['opend_sec'] = pd.to_numeric(cohort['opend_sec'], errors='coerce')

cohort = cohort.dropna(subset=required_columns).copy()
cohort['caseid'] = cohort['caseid'].astype(int)

# Locked 30-min analysis window must be available
cohort = cohort.loc[
    (cohort['opend_sec'] - cohort['opstart_sec']) >= WINDOW_SEC
].copy()

# Keep ALL cases, including repeated subjects.
# Repeated subjects will be handled later by subject-grouped CV.
cohort = cohort.sort_values('caseid').reset_index(drop=True)

print(f'\nEligible full-cohort cases for STEP 05: {len(cohort):,}')

if 'subjectid' in cohort.columns:
    print(f'Unique subjects: {cohort["subjectid"].nunique():,}')

print('Analysis window: first 30 min after opstart')
print('Waveform sampling for locked features: 50 Hz')
print('Primary ECG waveform timing: excluded')
print('Outcome variables: NOT used in this step')


# ============================================================
# 10. RESUME FROM CHECKPOINT
# ============================================================

result_rows = []
error_rows = []
processed_caseids = set()

if CHECKPOINT_PATH.exists():
    try:
        checkpoint = pd.read_csv(CHECKPOINT_PATH)
        checkpoint.columns = checkpoint.columns.str.replace('\ufeff', '', regex=False).str.strip()

        if 'caseid' in checkpoint.columns:
            checkpoint['caseid'] = pd.to_numeric(checkpoint['caseid'], errors='coerce')
            checkpoint = checkpoint.dropna(subset=['caseid']).copy()
            checkpoint['caseid'] = checkpoint['caseid'].astype(int)

            result_rows = checkpoint.to_dict('records')
            processed_caseids = set(checkpoint['caseid'].tolist())

            print(
                f'\nCheckpoint found: {len(processed_caseids):,} successful cases already processed.'
            )
    except Exception as error:
        print(f'\nWARNING: checkpoint load failed, starting fresh: {error}')
        result_rows = []
        processed_caseids = set()

if ERROR_PATH.exists():
    try:
        old_errors = pd.read_csv(ERROR_PATH)
        if len(old_errors) > 0:
            error_rows = old_errors.to_dict('records')
    except Exception:
        error_rows = []

remaining = cohort.loc[
    ~cohort['caseid'].isin(processed_caseids)
].copy()

print(f'Remaining cases this run: {len(remaining):,}')


# ============================================================
# 11. FULL COHORT EXTRACTION
# ============================================================

print('\n' + '=' * 90)
print('FULL COHORT EXTRACTION START')
print('=' * 90)

run_success = 0
run_errors = 0
run_qc_pass = 0

for run_index, (_, row) in enumerate(remaining.iterrows(), start=1):
    caseid = int(row['caseid'])

    try:
        result = extract_one_case(row)
        result_rows.append(result)

        run_success += 1
        run_qc_pass += int(result['qc_pass'])

    except Exception as error:
        run_errors += 1

        error_rows.append({
            'caseid': caseid,
            'error': str(error),
            'traceback': traceback.format_exc(),
        })

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------
    if run_index % PRINT_EVERY == 0 or run_index == len(remaining):
        total_success = len(result_rows)
        total_qc_pass = sum(int(item.get('qc_pass', 0)) for item in result_rows)

        print(
            f'{run_index:,} / {len(remaining):,} this run | '
            f'total success={total_success:,} | '
            f'QC pass={total_qc_pass:,} | '
            f'run errors={run_errors:,}'
        )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------
    if run_index % CHECKPOINT_EVERY == 0 or run_index == len(remaining):
        checkpoint_df = pd.DataFrame(result_rows)
        checkpoint_df.to_csv(
            CHECKPOINT_PATH,
            index=False,
            encoding='utf-8-sig'
        )

        pd.DataFrame(error_rows).to_csv(
            ERROR_PATH,
            index=False,
            encoding='utf-8-sig'
        )


# ============================================================
# 12. FINAL DATAFRAMES
# ============================================================

result_df = pd.DataFrame(result_rows)

if len(result_df) == 0:
    raise RuntimeError('No successfully extracted cases.')

result_df['caseid'] = pd.to_numeric(result_df['caseid'], errors='coerce')
result_df = result_df.dropna(subset=['caseid']).copy()
result_df['caseid'] = result_df['caseid'].astype(int)

# If the script was resumed multiple times, keep one row per case.
result_df = (
    result_df
    .sort_values('caseid')
    .drop_duplicates('caseid', keep='last')
    .reset_index(drop=True)
)

qc_pass_df = result_df.loc[
    result_df['qc_pass'] == 1
].copy()

error_df = pd.DataFrame(error_rows)


# ============================================================
# 13. QC REASON COUNTS
# ============================================================

reason_counts = {}

for text in result_df.loc[result_df['qc_pass'] == 0, 'qc_reason'].fillna(''):
    for reason in str(text).split(';'):
        reason = reason.strip()
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

qc_reason_df = pd.DataFrame([
    {'reason': reason, 'n_cases': count}
    for reason, count in reason_counts.items()
])

if len(qc_reason_df) > 0:
    qc_reason_df = qc_reason_df.sort_values(
        'n_cases', ascending=False
    ).reset_index(drop=True)


# ============================================================
# 14. FEATURE DISTRIBUTIONS - NO OUTCOME ANALYSIS
# ============================================================

feature_distribution_rows = []

for feature in LOCKED_FEATURES:
    values = pd.to_numeric(
        qc_pass_df[feature],
        errors='coerce'
    ).replace([np.inf, -np.inf], np.nan)

    valid = values.dropna()

    feature_distribution_rows.append({
        'feature': feature,
        'n_total_qc_pass': len(qc_pass_df),
        'n_nonmissing': int(valid.notna().sum()),
        'missing_n': int(values.isna().sum()),
        'missing_percent': float(values.isna().mean() * 100),
        'mean': float(valid.mean()) if len(valid) else np.nan,
        'std': float(valid.std()) if len(valid) else np.nan,
        'median': float(valid.median()) if len(valid) else np.nan,
        'q1': float(valid.quantile(0.25)) if len(valid) else np.nan,
        'q3': float(valid.quantile(0.75)) if len(valid) else np.nan,
        'min': float(valid.min()) if len(valid) else np.nan,
        'max': float(valid.max()) if len(valid) else np.nan,
    })

feature_distribution_df = pd.DataFrame(feature_distribution_rows)


# ============================================================
# 15. SAVE FINAL OUTPUTS
# ============================================================

result_df.to_csv(
    OUTPUT_ALL_PATH,
    index=False,
    encoding='utf-8-sig'
)

qc_pass_df.to_csv(
    OUTPUT_QC_PASS_PATH,
    index=False,
    encoding='utf-8-sig'
)

error_df.to_csv(
    ERROR_PATH,
    index=False,
    encoding='utf-8-sig'
)

qc_reason_df.to_csv(
    QC_REASON_PATH,
    index=False,
    encoding='utf-8-sig'
)

feature_distribution_df.to_csv(
    FEATURE_DISTRIBUTION_PATH,
    index=False,
    encoding='utf-8-sig'
)

# Refresh checkpoint with deduplicated final successes
result_df.to_csv(
    CHECKPOINT_PATH,
    index=False,
    encoding='utf-8-sig'
)


# ============================================================
# 16. SUMMARY
# ============================================================

initial_n = len(cohort)
success_n = len(result_df)
qc_pass_n = len(qc_pass_df)
qc_fail_n = success_n - qc_pass_n
unique_subjects_qc = (
    qc_pass_df['subjectid'].nunique()
    if 'subjectid' in qc_pass_df.columns
    else np.nan
)

summary_df = pd.DataFrame([{
    'step02_input_cases': initial_n,
    'successfully_extracted_cases': success_n,
    'waveform_or_processing_error_cases': len(error_df),
    'qc_pass_cases': qc_pass_n,
    'qc_fail_cases': qc_fail_n,
    'qc_pass_percent_of_success': (
        qc_pass_n / success_n * 100 if success_n > 0 else np.nan
    ),
    'qc_pass_unique_subjects': unique_subjects_qc,
    'analysis_window_min': WINDOW_SEC / 60,
    'waveform_resample_hz': FS,
    'locked_primary_feature_count': len(LOCKED_FEATURES),
}])

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding='utf-8-sig'
)


# ============================================================
# 17. PRINT FINAL SUMMARY
# ============================================================

print('\n' + '=' * 90)
print('STEP 05 FULL EXTRACTION SUMMARY')
print('=' * 90)

print(f'\nInput eligible cases: {initial_n:,}')
print(f'Successfully extracted: {success_n:,}')
print(f'Errors recorded: {len(error_df):,}')
print(f'QC PASS: {qc_pass_n:,}')
print(f'QC FAIL: {qc_fail_n:,}')

if success_n > 0:
    print(f'QC PASS rate: {qc_pass_n / success_n * 100:.1f}%')

if np.isfinite(unique_subjects_qc):
    print(f'QC PASS unique subjects: {int(unique_subjects_qc):,}')

print('\nLOCKED FEATURE DISTRIBUTIONS (QC PASS COHORT)')
print('-' * 90)

for _, item in feature_distribution_df.iterrows():
    print(
        f'{item["feature"]:<38} '
        f'N={int(item["n_nonmissing"]):>4,} | '
        f'missing={item["missing_percent"]:5.1f}% | '
        f'median={item["median"]: .4f} | '
        f'IQR={item["q1"]: .4f} to {item["q3"]: .4f}'
    )

print('\nQC EXCLUSION REASONS')
print('-' * 90)

if len(qc_reason_df) > 0:
    print(qc_reason_df.to_string(index=False))
else:
    print('No QC exclusions')


# ============================================================
# 18. FEASIBILITY DECISION
# ============================================================

print('\n' + '=' * 90)
print('STEP 05 DECISION')
print('=' * 90)

if qc_pass_n >= 2500:
    decision = 'VERY GOOD'
    reason = 'Locked coupling features were successfully extracted in a large cohort.'
elif qc_pass_n >= 1800:
    decision = 'GOOD'
    reason = 'Full-cohort feature extraction remains adequately powered for subsequent analysis.'
elif qc_pass_n >= 1200:
    decision = 'ACCEPTABLE'
    reason = 'Analysis can proceed, but cohort reduction should be reported carefully.'
else:
    decision = 'REVIEW'
    reason = 'Waveform QC reduced the analyzable cohort substantially; review before outcome modeling.'

print(f'\nDecision: {decision}')
print(f'Reason: {reason}')

print('\nIMPORTANT:')
print('The 12 primary coupling features remain LOCKED.')
print('Do NOT add/remove features after examining postoperative outcomes or model performance.')


# ============================================================
# 19. FINISH
# ============================================================

print('\n' + '=' * 90)
print('STEP 05 COMPLETE')
print('=' * 90)

print('\nGenerated files:')
print(OUTPUT_ALL_PATH)
print(OUTPUT_QC_PASS_PATH)
print(ERROR_PATH)
print(QC_REASON_PATH)
print(SUMMARY_PATH)
print(FEATURE_DISTRIBUTION_PATH)
print(CHECKPOINT_PATH)

print('\nNEXT STEP:')
print(
    '''\nSTEP 06 = outcome-independent feature audit.\n\nBefore any ML/outcome modeling:\n1. Check missingness of the 12 locked features\n2. Check pairwise Spearman correlations / redundancy\n3. Check extreme distributions and finite ranges\n4. Compare repeated-case / unique-subject structure\n5. Freeze the final analysis cohort\n\nOnly after STEP 06 will outcomes be merged for statistical / ML analysis.\n'''
)