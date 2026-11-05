from pathlib import Path
import warnings
import traceback

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, correlate, correlation_lags
import vitaldb

warnings.filterwarnings('ignore')

# ============================================================
# VitalDB Multimodal Physiological Coupling Study
# STEP 04 - Locked Final Coupling Feature Pilot
# ============================================================
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
# ECG waveform timing is intentionally excluded from the
# primary locked feature set because STEP03B still showed
# beat-count inconsistency in some pilot cases.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / 'data_raw'
PROCESSED_DIR = PROJECT_DIR / 'data_processed'
RESULT_DIR = PROJECT_DIR / 'results'
NUMERIC_CACHE_DIR = RAW_DIR / 'numeric_tracks'

INPUT_PATH = PROCESSED_DIR / '02_numeric_quality_eligible_cases.csv'
SELECTED_PATH = RESULT_DIR / '04_final_feature_pilot_selected_cases.csv'
OUTPUT_PATH = RESULT_DIR / '04_final_feature_pilot_results.csv'
ERROR_PATH = RESULT_DIR / '04_final_feature_pilot_errors.csv'
FEATURE_DEF_PATH = RESULT_DIR / '04_locked_primary_feature_definition.csv'

N_PILOT = 30
RANDOM_STATE = 20260819

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

RESULT_DIR.mkdir(parents=True, exist_ok=True)


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
# LOCK FEATURE SET BEFORE OUTCOME ANALYSIS
# ============================================================

feature_definition = pd.DataFrame([
    ['art_pleth_lag_median_sec', 'vascular timing', 1],
    ['art_pleth_lag_iqr_sec', 'vascular timing variability', 1],
    ['art_pleth_xcorr_median', 'waveform coupling strength', 1],
    ['art_pleth_xcorr_iqr', 'waveform coupling variability', 1],
    ['art_pleth_xcorr_lag_median_sec', 'waveform coupling delay', 1],
    ['art_pleth_amplitude_spearman', 'vascular amplitude coupling', 1],
    ['hr_map_corr_median', 'cardio-hemodynamic coupling', 1],
    ['hr_map_corr_iqr', 'cardio-hemodynamic variability', 1],
    ['map_etco2_corr_median', 'hemodynamic-respiratory coupling', 1],
    ['map_etco2_corr_iqr', 'hemodynamic-respiratory variability', 1],
    ['hr_etco2_corr_median', 'cardiorespiratory coupling', 1],
    ['hr_etco2_corr_iqr', 'cardiorespiratory variability', 1],
], columns=['feature', 'domain', 'primary'])

feature_definition.to_csv(
    FEATURE_DEF_PATH,
    index=False,
    encoding='utf-8-sig'
)


# ============================================================
# LOAD ELIGIBLE COHORT
# ============================================================

print('=' * 90)
print('VitalDB Multimodal Physiological Coupling Study')
print('STEP 04 - Locked Final Coupling Feature Pilot')
print('=' * 90)

if not INPUT_PATH.exists():
    raise FileNotFoundError(f'입력 파일 없음: {INPUT_PATH}')

cohort = pd.read_csv(INPUT_PATH)
cohort.columns = cohort.columns.str.replace('\ufeff', '', regex=False).str.strip()

required = ['caseid', 'opstart_sec', 'opend_sec', 'common_valid_minutes']
for col in required:
    if col not in cohort.columns:
        raise ValueError(f'필수 column 없음: {col}')

cohort['caseid'] = pd.to_numeric(cohort['caseid'], errors='coerce')
cohort['opstart_sec'] = pd.to_numeric(cohort['opstart_sec'], errors='coerce')
cohort['opend_sec'] = pd.to_numeric(cohort['opend_sec'], errors='coerce')
cohort['common_valid_minutes'] = pd.to_numeric(
    cohort['common_valid_minutes'], errors='coerce'
)

cohort = cohort.dropna(subset=required).copy()
cohort['caseid'] = cohort['caseid'].astype(int)

# first 30 min after opstart must exist
cohort = cohort.loc[
    (cohort['opend_sec'] - cohort['opstart_sec']) >= WINDOW_SEC
].copy()

# avoid repeated subject in pilot
if 'subjectid' in cohort.columns:
    cohort = (
        cohort.sort_values(['subjectid', 'caseid'])
        .drop_duplicates('subjectid', keep='first')
        .copy()
    )

print(f'\nPilot selection pool: {len(cohort):,} cases')

# stratified sampling by common valid duration quintile
cohort['duration_bin'] = pd.qcut(
    cohort['common_valid_minutes'].rank(method='first'),
    q=5,
    labels=False
)

selected_parts = []
per_bin = N_PILOT // 5

for bin_id in range(5):
    pool = cohort.loc[cohort['duration_bin'] == bin_id]
    take_n = min(per_bin, len(pool))
    if take_n > 0:
        selected_parts.append(
            pool.sample(n=take_n, random_state=RANDOM_STATE + bin_id)
        )

selected = pd.concat(selected_parts, ignore_index=True)

if len(selected) < N_PILOT:
    remaining = cohort.loc[~cohort['caseid'].isin(selected['caseid'])]
    extra_n = min(N_PILOT - len(selected), len(remaining))
    if extra_n > 0:
        extra = remaining.sample(
            n=extra_n,
            random_state=RANDOM_STATE + 99
        )
        selected = pd.concat([selected, extra], ignore_index=True)

selected = selected.head(N_PILOT).copy()
selected['pilot_order'] = np.arange(1, len(selected) + 1)
selected.to_csv(SELECTED_PATH, index=False, encoding='utf-8-sig')

print(f'Selected pilot cases: {len(selected)}')
print('Analysis window: first 30 min after opstart')
print('ECG waveform timing: excluded from PRIMARY feature set')


# ============================================================
# RUN 30-CASE PILOT
# ============================================================

result_rows = []
error_rows = []

for pilot_no, (_, row) in enumerate(selected.iterrows(), start=1):
    caseid = int(row['caseid'])
    start_sec = float(row['opstart_sec'])
    end_sec = start_sec + WINDOW_SEC

    print(f'\n[{pilot_no:02d}/{len(selected):02d}] caseid={caseid}')

    try:
        # ----------------------------------------------------
        # WAVEFORM
        # ----------------------------------------------------
        values = vitaldb.load_case(
            caseid,
            TRACK_NAMES,
            interval=INTERVAL
        )
        values = np.asarray(values, dtype=float)

        start_index = int(np.floor(start_sec * FS))
        end_index = int(np.floor(end_sec * FS))

        if end_index > len(values):
            raise ValueError('waveform shorter than first 30-min operation window')

        window = values[start_index:end_index, :]

        if len(window) != WINDOW_SEC * FS:
            raise ValueError('unexpected waveform window length')

        pleth_raw = window[:, 0]
        art_raw = window[:, 1]

        pleth, pleth_missing = interpolate_waveform(pleth_raw)
        art, art_missing = interpolate_waveform(art_raw)

        pleth_coverage = 1.0 - pleth_missing
        art_coverage = 1.0 - art_missing

        art_peaks, art_filtered = detect_pulse_peaks(art, FS, 'ART')
        pleth_peaks, pleth_filtered = detect_pulse_peaks(pleth, FS, 'PLETH')

        art_rate = rate_from_peaks(art_peaks, FS)
        pleth_rate = rate_from_peaks(pleth_peaks, FS)

        if np.isfinite(art_rate) and np.isfinite(pleth_rate):
            rate_diff = abs(art_rate - pleth_rate)
        else:
            rate_diff = np.nan

        # ----------------------------------------------------
        # ART -> PLETH beat pairing
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Beat amplitude coupling
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # 60-sec waveform cross-correlation windows
        # ----------------------------------------------------
        xcorr_values = []
        xcorr_lag_values = []
        xcorr_window_samples = 60 * FS

        for start in range(
            0,
            len(art_filtered) - xcorr_window_samples + 1,
            xcorr_window_samples
        ):
            end = start + xcorr_window_samples

            corr_value, corr_lag = positive_lag_xcorr(
                art_filtered[start:end],
                pleth_filtered[start:end],
                FS,
                max_lag_sec=0.8
            )

            if np.isfinite(corr_value) and np.isfinite(corr_lag):
                xcorr_values.append(corr_value)
                xcorr_lag_values.append(corr_lag)

        xcorr_values = np.asarray(xcorr_values, dtype=float)
        xcorr_lag_values = np.asarray(xcorr_lag_values, dtype=float)

        # ----------------------------------------------------
        # NUMERIC SIGNALS FROM STEP02 CACHE
        # ----------------------------------------------------
        hr = load_numeric_window(
            caseid, 'HR', start_sec, end_sec, 20, 250
        )
        map_values = load_numeric_window(
            caseid, 'MAP', start_sec, end_sec, 20, 200
        )
        etco2 = load_numeric_window(
            caseid, 'ETCO2', start_sec, end_sec, 5, 80
        )

        if hr is None or map_values is None or etco2 is None:
            raise ValueError('numeric cache missing')

        hr_map_corr = rolling_spearman(hr, map_values)
        map_etco2_corr = rolling_spearman(map_values, etco2)
        hr_etco2_corr = rolling_spearman(hr, etco2)

        # ----------------------------------------------------
        # QC
        # ----------------------------------------------------
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

        pilot_pass = len(qc_reasons) == 0

        result_rows.append({
            'pilot_order': pilot_no,
            'caseid': caseid,
            'window_start_sec': start_sec,
            'window_end_sec': end_sec,
            'art_coverage': art_coverage,
            'pleth_coverage': pleth_coverage,
            'art_rate_bpm': art_rate,
            'pleth_rate_bpm': pleth_rate,
            'art_pleth_rate_diff_bpm': rate_diff,
            'art_pleth_pairs_n': len(lag_values),
            'art_pleth_lag_median_sec': (
                float(np.median(lag_values)) if len(lag_values) else np.nan
            ),
            'art_pleth_lag_iqr_sec': calc_iqr(lag_values),
            'art_pleth_xcorr_windows_n': len(xcorr_values),
            'art_pleth_xcorr_median': (
                float(np.median(xcorr_values)) if len(xcorr_values) else np.nan
            ),
            'art_pleth_xcorr_iqr': calc_iqr(xcorr_values),
            'art_pleth_xcorr_lag_median_sec': (
                float(np.median(xcorr_lag_values))
                if len(xcorr_lag_values) else np.nan
            ),
            'art_pleth_amplitude_spearman': amplitude_rho,
            'hr_map_windows_n': len(hr_map_corr),
            'hr_map_corr_median': (
                float(np.median(hr_map_corr)) if len(hr_map_corr) else np.nan
            ),
            'hr_map_corr_iqr': calc_iqr(hr_map_corr),
            'map_etco2_windows_n': len(map_etco2_corr),
            'map_etco2_corr_median': (
                float(np.median(map_etco2_corr))
                if len(map_etco2_corr) else np.nan
            ),
            'map_etco2_corr_iqr': calc_iqr(map_etco2_corr),
            'hr_etco2_windows_n': len(hr_etco2_corr),
            'hr_etco2_corr_median': (
                float(np.median(hr_etco2_corr))
                if len(hr_etco2_corr) else np.nan
            ),
            'hr_etco2_corr_iqr': calc_iqr(hr_etco2_corr),
            'pilot_pass': int(pilot_pass),
            'qc_reason': ';'.join(qc_reasons),
        })

        lag_text = np.median(lag_values) if len(lag_values) else np.nan
        xcorr_text = np.median(xcorr_values) if len(xcorr_values) else np.nan

        print(
            f'  ART/PLETH rate = {art_rate:.1f}/{pleth_rate:.1f} bpm | '
            f'lag = {lag_text:.3f} s | '
            f'xcorr = {xcorr_text:.3f} | '
            f'PASS = {pilot_pass}'
        )

    except Exception as error:
        print(f'  ERROR: {error}')
        error_rows.append({
            'pilot_order': pilot_no,
            'caseid': caseid,
            'error': str(error),
            'traceback': traceback.format_exc(),
        })


# ============================================================
# SAVE AND SUMMARY
# ============================================================

result_df = pd.DataFrame(result_rows)
error_df = pd.DataFrame(error_rows)

result_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
error_df.to_csv(ERROR_PATH, index=False, encoding='utf-8-sig')

print('\n' + '=' * 90)
print('STEP 04 FINAL FEATURE PILOT SUMMARY')
print('=' * 90)

print(f'\nSelected: {len(selected)}')
print(f'Analyzed: {len(result_df)}')
print(f'Errors: {len(error_df)}')

if len(result_df) > 0:
    pass_n = int(result_df['pilot_pass'].sum())
    pass_rate = float(result_df['pilot_pass'].mean())

    print(
        f'Pilot PASS: {pass_n}/{len(result_df)} '
        f'({pass_rate * 100:.1f}%)'
    )

    print('\nLocked feature medians:')

    for feature in feature_definition['feature']:
        if feature in result_df.columns:
            values = pd.to_numeric(
                result_df[feature],
                errors='coerce'
            ).dropna()

            if len(values) > 0:
                print(
                    f'{feature:<38} '
                    f'median={values.median():.4f}'
                )

    if pass_rate >= 0.80 and len(error_df) <= 2:
        decision = 'PASS'
    elif pass_rate >= 0.60:
        decision = 'REVIEW'
    else:
        decision = 'FAIL'
else:
    decision = 'FAIL'

print('\n' + '=' * 90)
print('STEP 04 DECISION')
print('=' * 90)
print(f'\nDecision: {decision}')

if decision == 'PASS':
    print('''
Final primary coupling feature definitions are technically stable
in the 30-case pilot.

Next step:
STEP 05 = extract the locked features for the full eligible cohort.
Do not change the feature definitions after seeing outcome results.
''')
elif decision == 'REVIEW':
    print('''
Some pilot cases failed QC.
Review the failure reasons before full-cohort extraction.
Do not change features based on postoperative outcome performance.
''')
else:
    print('''
The locked pipeline is not yet technically stable enough for
full-cohort extraction. Review QC failures first.
''')

print('\nGenerated files:')
print(SELECTED_PATH)
print(FEATURE_DEF_PATH)
print(OUTPUT_PATH)
print(ERROR_PATH)