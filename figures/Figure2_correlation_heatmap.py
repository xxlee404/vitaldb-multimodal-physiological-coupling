# ============================================================
# 07B_make_publication_crossblock_heatmap.py
#
# Publication-quality Spearman correlation heatmap
# Conventional physiological features (20)
# vs multimodal coupling features (12)
#
# Project:
# vitaldb_multimodal_coupling_BSPC
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_DIR / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Conventional physiological features
# ============================================================

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


# ============================================================
# 3. Multimodal coupling features
# ============================================================

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

ALL_REQUIRED_FEATURES = (
    CONVENTIONAL_FEATURES
    + COUPLING_FEATURES
)


# ============================================================
# 4. Publication-friendly display labels
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
    "ART–PLETH\nxcorr lag\nmedian",
    "ART–PLETH\namplitude ρ",

    "HR–MAP\ncorr median",
    "HR–MAP\ncorr IQR",

    "MAP–ETCO₂\ncorr median",
    "MAP–ETCO₂\ncorr IQR",

    "HR–ETCO₂\ncorr median",
    "HR–ETCO₂\ncorr IQR",
]


# ============================================================
# 5. Find CSV containing all required features
# ============================================================

def find_input_csv(project_dir, required_features):

    print("=" * 80)
    print("Searching for CSV containing all required features")
    print("=" * 80)

    candidates = []

    for csv_path in project_dir.rglob("*.csv"):

        if "figures" in [
            str(part).lower()
            for part in csv_path.parts
        ]:
            continue

        try:

            header = pd.read_csv(
                csv_path,
                nrows=0
            )

            columns = set(header.columns)

            if set(required_features).issubset(columns):

                try:
                    temp = pd.read_csv(
                        csv_path,
                        usecols=required_features
                    )

                    n_rows = len(temp)

                except Exception:
                    n_rows = 0

                candidates.append(
                    (csv_path, n_rows)
                )

                print()
                print("Candidate:")
                print(csv_path)
                print(f"Rows: {n_rows:,}")

        except Exception:
            continue

    if len(candidates) == 0:

        raise FileNotFoundError(
            "\nNo CSV containing all 32 required features "
            "was found.\n"
            "Check the STEP07/07B output CSV files."
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    selected_path, selected_rows = candidates[0]

    print()
    print("=" * 80)
    print("Selected input file")
    print("=" * 80)

    print(selected_path)
    print(f"Rows: {selected_rows:,}")
    print()

    return selected_path


INPUT_CSV = find_input_csv(
    PROJECT_DIR,
    ALL_REQUIRED_FEATURES
)


# ============================================================
# 6. Load data
# ============================================================

df = pd.read_csv(
    INPUT_CSV,
    usecols=ALL_REQUIRED_FEATURES
)

print("=" * 80)
print("Input data summary")
print("=" * 80)

print(f"Cases: {len(df):,}")
print(
    f"Conventional features: "
    f"{len(CONVENTIONAL_FEATURES)}"
)
print(
    f"Coupling features: "
    f"{len(COUPLING_FEATURES)}"
)
print()


# ============================================================
# 7. Convert all features to numeric
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
# 8. Missing value audit
# ============================================================

print("=" * 80)
print("Missing-value audit")
print("=" * 80)

missing = (
    df[ALL_REQUIRED_FEATURES]
    .isna()
    .sum()
)

missing = missing[
    missing > 0
]

if len(missing) == 0:

    print(
        "No missing values among the "
        "32 signal features."
    )

else:

    print(missing)

print()


# ============================================================
# 9. Calculate Spearman correlation matrix
# ============================================================

corr_all = df[
    ALL_REQUIRED_FEATURES
].corr(
    method="spearman"
)

cross_corr = corr_all.loc[
    CONVENTIONAL_FEATURES,
    COUPLING_FEATURES
]


# ============================================================
# 10. Numerical audit
# ============================================================

abs_corr = cross_corr.abs()

max_abs_rho = (
    abs_corr
    .to_numpy()
    .max()
)

max_location = np.unravel_index(
    np.argmax(
        abs_corr.to_numpy()
    ),
    abs_corr.shape
)

max_conv = (
    cross_corr
    .index[
        max_location[0]
    ]
)

max_coupling = (
    cross_corr
    .columns[
        max_location[1]
    ]
)

max_rho = cross_corr.iloc[
    max_location
]

n_ge_070 = int(
    (
        abs_corr >= 0.70
    )
    .sum()
    .sum()
)


print("=" * 80)
print(
    "Conventional vs coupling "
    "Spearman correlation results"
)
print("=" * 80)

print(
    f"Maximum |rho| = "
    f"{max_abs_rho:.4f}"
)

print(
    f"Pair = "
    f"{max_conv} "
    f"vs "
    f"{max_coupling}"
)

print(
    f"rho = "
    f"{max_rho:.4f}"
)

print(
    f"Number of pairs "
    f"with |rho| >= 0.70: "
    f"{n_ge_070}"
)

print()


# ============================================================
# 11. Save correlation matrix
# ============================================================

OUTPUT_CSV = (
    OUTPUT_DIR
    / "Fig1_conventional_vs_coupling_spearman_matrix.csv"
)

cross_corr.to_csv(
    OUTPUT_CSV,
    encoding="utf-8-sig"
)

print(
    "Saved numerical correlation matrix:"
)

print(
    OUTPUT_CSV
)

print()


# ============================================================
# 12. Create publication-quality heatmap
# ============================================================

fig, ax = plt.subplots(
    figsize=(15, 10.5)
)

image = ax.imshow(
    cross_corr.values,
    aspect="auto",
    interpolation="nearest",
    cmap="viridis",
    vmin=-1,
    vmax=1
)


# ============================================================
# 13. X-axis labels
# ============================================================

ax.set_xticks(
    np.arange(
        len(COUPLING_LABELS)
    )
)

ax.set_xticklabels(
    COUPLING_LABELS,
    fontsize=10.5,
    rotation=25,
    ha="right",
    rotation_mode="anchor"
)


# ============================================================
# 14. Y-axis labels
# ============================================================

ax.set_yticks(
    np.arange(
        len(CONVENTIONAL_LABELS)
    )
)

ax.set_yticklabels(
    CONVENTIONAL_LABELS,
    fontsize=10.5
)


# ============================================================
# 15. Add Spearman rho values inside cells
# ============================================================

for i in range(
    cross_corr.shape[0]
):

    for j in range(
        cross_corr.shape[1]
    ):

        value = (
            cross_corr
            .iloc[i, j]
        )

        if np.isnan(value):

            text = ""

        else:

            text = (
                f"{value:.2f}"
            )

        if (
            not np.isnan(value)
            and abs(value) >= 0.55
        ):

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


# ============================================================
# 16. Colorbar
# ============================================================

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


# ============================================================
# 17. Title
# ============================================================

ax.set_title(
    "Correlation Structure of Conventional and Multimodal Coupling Features",
    fontsize=14,
    pad=16
)


# ============================================================
# 18. Axis titles
# ============================================================

ax.set_xlabel(
    "Multimodal physiological coupling features",
    fontsize=11,
    labelpad=14
)

ax.set_ylabel(
    "Conventional physiological features",
    fontsize=11,
    labelpad=14
)


# ============================================================
# 19. Thin cell grid
# ============================================================

ax.set_xticks(
    np.arange(
        -0.5,
        len(COUPLING_FEATURES),
        1
    ),
    minor=True
)

ax.set_yticks(
    np.arange(
        -0.5,
        len(CONVENTIONAL_FEATURES),
        1
    ),
    minor=True
)

ax.grid(
    which="minor",
    linewidth=0.30,
    alpha=0.35
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)


# ============================================================
# 20. Figure layout
# ============================================================

plt.subplots_adjust(
    bottom=0.19,
    left=0.14,
    right=0.92,
    top=0.92
)

plt.tight_layout()


# ============================================================
# 21. Output paths
# ============================================================

PNG_PATH = (
    OUTPUT_DIR
    / "Fig1_conventional_vs_coupling_spearman_heatmap.png"
)

PDF_PATH = (
    OUTPUT_DIR
    / "Fig1_conventional_vs_coupling_spearman_heatmap.pdf"
)

TIFF_PATH = (
    OUTPUT_DIR
    / "Fig1_conventional_vs_coupling_spearman_heatmap.tiff"
)


# ============================================================
# 22. Save PNG
# ============================================================

plt.savefig(
    PNG_PATH,
    dpi=600,
    bbox_inches="tight"
)


# ============================================================
# 23. Save PDF
# ============================================================

plt.savefig(
    PDF_PATH,
    bbox_inches="tight"
)


# ============================================================
# 24. Save TIFF
# ============================================================

plt.savefig(
    TIFF_PATH,
    dpi=600,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 25. Final console report
# ============================================================

print("=" * 80)
print("FIGURE CREATION COMPLETE")
print("=" * 80)

print()
print("PNG:")
print(PNG_PATH)

print()
print("PDF:")
print(PDF_PATH)

print()
print("TIFF:")
print(TIFF_PATH)

print()

print("=" * 80)
print("Key numerical result")
print("=" * 80)

print(
    f"Maximum absolute Spearman correlation "
    f"= {max_abs_rho:.4f}"
)

print(
    f"Maximum-correlation pair "
    f"= {max_conv} "
    f"vs {max_coupling}"
)

print(
    f"Spearman rho "
    f"= {max_rho:.4f}"
)

print(
    f"Number of conventional–coupling "
    f"pairs with |rho| >= 0.70 "
    f"= {n_ge_070}"
)

print()
print("Done.")