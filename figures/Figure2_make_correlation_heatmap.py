# ============================================================
# 07B_make_publication_crossblock_heatmap.py
#
# Publication-quality 20 x 12 Spearman correlation heatmap
# Conventional physiological features vs coupling features
#
# Project:
# vitaldb_multimodal_coupling_BSPC
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Project path
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]


OUTPUT_DIR = PROJECT_DIR / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Feature definitions
# ============================================================

# ------------------------------------------------------------
# Conventional physiological features: 20
# ------------------------------------------------------------

CONVENTIONAL_FEATURES = [
    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",
    "hr_range",

    "map_mean",
    "map_std",
    "map_min",
    "map_max",
    "map_range",

    "spo2_mean",
    "spo2_std",
    "spo2_min",
    "spo2_max",
    "spo2_range",

    "etco2_mean",
    "etco2_std",
    "etco2_min",
    "etco2_max",
    "etco2_range",
]


# ------------------------------------------------------------
# Coupling features: 12
# ------------------------------------------------------------

COUPLING_FEATURES = [
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

ALL_REQUIRED_FEATURES = CONVENTIONAL_FEATURES + COUPLING_FEATURES


# ============================================================
# 3. Publication-friendly labels
# ============================================================

CONVENTIONAL_LABELS = [
    "HR mean",
    "HR SD",
    "HR minimum",
    "HR maximum",
    "HR range",

    "MAP mean",
    "MAP SD",
    "MAP minimum",
    "MAP maximum",
    "MAP range",

    "SpO₂ mean",
    "SpO₂ SD",
    "SpO₂ minimum",
    "SpO₂ maximum",
    "SpO₂ range",

    "ETCO₂ mean",
    "ETCO₂ SD",
    "ETCO₂ minimum",
    "ETCO₂ maximum",
    "ETCO₂ range",
]


COUPLING_LABELS = [
    "ART–PLETH\nlag median",
    "ART–PLETH\nlag IQR",
    "ART–PLETH\nxcorr median",
    "ART–PLETH\nxcorr IQR",
    "ART–PLETH\nxcorr lag median",
    "ART–PLETH\namplitude ρ",
    "HR–MAP\ncorr median",
    "HR–MAP\ncorr IQR",
    "MAP–ETCO₂\ncorr median",
    "MAP–ETCO₂\ncorr IQR",
    "HR–ETCO₂\ncorr median",
    "HR–ETCO₂\ncorr IQR",
]


# ============================================================
# 4. Automatically find CSV containing all 32 features
# ============================================================

def find_input_csv(project_dir, required_features):
    print("=" * 80)
    print("Searching for CSV containing all 32 signal features")
    print("=" * 80)

    candidates = []

    for csv_path in project_dir.rglob("*.csv"):

        # Ignore newly generated result folders if necessary
        if "figures" in [p.lower() for p in csv_path.parts]:
            continue

        try:
            header = pd.read_csv(csv_path, nrows=0)
            cols = set(header.columns)

            if set(required_features).issubset(cols):
                # Get row count only for valid candidates
                try:
                    temp = pd.read_csv(
                        csv_path,
                        usecols=required_features
                    )
                    n_rows = len(temp)
                except Exception:
                    n_rows = 0

                candidates.append((csv_path, n_rows))

                print(
                    f"Candidate: {csv_path}\n"
                    f"  rows = {n_rows:,}"
                )

        except Exception:
            continue

    if len(candidates) == 0:
        raise FileNotFoundError(
            "\nNo CSV containing all 32 required features was found.\n"
            "Check that the STEP 07/07B analysis CSV exists in the project folder."
        )

    # Prefer the file containing the largest number of rows
    candidates.sort(key=lambda x: x[1], reverse=True)

    selected_path, selected_n = candidates[0]

    print("\nSelected input file:")
    print(selected_path)
    print(f"Rows: {selected_n:,}")
    print()

    return selected_path


INPUT_CSV = find_input_csv(
    PROJECT_DIR,
    ALL_REQUIRED_FEATURES
)


# ============================================================
# 5. Load data
# ============================================================

df = pd.read_csv(
    INPUT_CSV,
    usecols=ALL_REQUIRED_FEATURES
)

print("=" * 80)
print("Input data")
print("=" * 80)
print(f"Cases: {len(df):,}")
print(f"Conventional features: {len(CONVENTIONAL_FEATURES)}")
print(f"Coupling features:     {len(COUPLING_FEATURES)}")
print()


# ============================================================
# 6. Convert to numeric
# ============================================================

for col in ALL_REQUIRED_FEATURES:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# 7. Spearman correlation
# ============================================================

corr_all = df[ALL_REQUIRED_FEATURES].corr(
    method="spearman"
)

# Only:
# rows    = 20 conventional features
# columns = 12 coupling features

cross_corr = corr_all.loc[
    CONVENTIONAL_FEATURES,
    COUPLING_FEATURES
]


# ============================================================
# 8. Numerical audit
# ============================================================

abs_corr = cross_corr.abs()

max_abs_rho = abs_corr.to_numpy().max()

max_location = np.unravel_index(
    np.argmax(abs_corr.to_numpy()),
    abs_corr.shape
)

max_conv = cross_corr.index[max_location[0]]
max_coupling = cross_corr.columns[max_location[1]]
max_rho = cross_corr.iloc[max_location]


n_ge_070 = int(
    (abs_corr >= 0.70).sum().sum()
)

print("=" * 80)
print("Cross-block Spearman correlation results")
print("=" * 80)

print(
    f"Maximum |rho| = {max_abs_rho:.4f}"
)

print(
    f"Pair          = {max_conv} vs {max_coupling}"
)

print(
    f"rho           = {max_rho:.4f}"
)

print(
    f"Number of pairs with |rho| >= 0.70: {n_ge_070}"
)

print()


# ============================================================
# 9. Save numerical matrix
# ============================================================

OUTPUT_CSV = (
    OUTPUT_DIR
    / "Fig1_conventional_vs_coupling_spearman_matrix.csv"
)

cross_corr.to_csv(
    OUTPUT_CSV,
    encoding="utf-8-sig"
)

print(f"Saved correlation matrix:")
print(OUTPUT_CSV)
print()


# ============================================================
# 10. Create publication-quality heatmap
# ============================================================

fig, ax = plt.subplots(
    figsize=(15, 11)
)

image = ax.imshow(
    cross_corr.values,
    aspect="auto",
    interpolation="nearest",
    cmap="viridis",
    vmin=-1,
    vmax=1
)


# ------------------------------------------------------------
# X axis
# ------------------------------------------------------------

ax.set_xticks(
    np.arange(len(COUPLING_LABELS))
)

ax.set_xticklabels(
    COUPLING_LABELS,
    fontsize=10,
    rotation=45,
    ha="right",
    rotation_mode="anchor"
)


# ------------------------------------------------------------
# Y axis
# ------------------------------------------------------------

ax.set_yticks(
    np.arange(len(CONVENTIONAL_LABELS))
)

ax.set_yticklabels(
    CONVENTIONAL_LABELS,
    fontsize=10
)


# ------------------------------------------------------------
# Cell values
# ------------------------------------------------------------

for i in range(cross_corr.shape[0]):
    for j in range(cross_corr.shape[1]):

        value = cross_corr.iloc[i, j]

        if np.isnan(value):
            text = ""
        else:
            text = f"{value:.2f}"

        # Improve contrast automatically
        if abs(value) >= 0.55:
            text_color = "white"
        else:
            text_color = "black"

        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=7.5,
            color=text_color
        )


# ------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------

cbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.035,
    pad=0.025
)

cbar.set_label(
    "Spearman correlation coefficient (ρ)",
    fontsize=11
)

cbar.ax.tick_params(
    labelsize=9
)


# ------------------------------------------------------------
# Title / labels
# ------------------------------------------------------------

ax.set_title(
    "Spearman Correlations Between Conventional Physiological "
    "and Multimodal Coupling Features",
    fontsize=14,
    pad=16
)

ax.set_xlabel(
    "Multimodal physiological coupling features",
    fontsize=11,
    labelpad=12
)

ax.set_ylabel(
    "Conventional physiological features",
    fontsize=11,
    labelpad=12
)


# ------------------------------------------------------------
# Grid separating cells
# ------------------------------------------------------------

ax.set_xticks(
    np.arange(-0.5, len(COUPLING_FEATURES), 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, len(CONVENTIONAL_FEATURES), 1),
    minor=True
)

ax.grid(
    which="minor",
    linewidth=0.25,
    alpha=0.35
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)


# ============================================================
# 11. Layout and export
# ============================================================

plt.tight_layout()


PNG_PATH = (
    OUTPUT_DIR
    / "Fig1_conventional_vs_coupling_spearman_heatmap.png"
)

PDF_PATH = (
    OUTPUT_DIR
    / "Fig1_conventional_vs_coupling_spearman_heatmap.pdf"
)


plt.savefig(
    PNG_PATH,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    PDF_PATH,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. Done
# ============================================================

print("=" * 80)
print("FIGURE CREATION COMPLETE")
print("=" * 80)

print("\nPNG:")
print(PNG_PATH)

print("\nPDF:")
print(PDF_PATH)

print("\nKey result:")
print(
    f"Maximum absolute Spearman correlation = "
    f"{max_abs_rho:.4f}"
)

print(
    f"|rho| >= 0.70 pairs = {n_ge_070}"
)