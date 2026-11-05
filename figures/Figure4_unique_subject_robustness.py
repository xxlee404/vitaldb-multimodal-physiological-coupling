# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_DIR
    / "results"
    / "12C_comparison_with_STEP12B.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("FIG. 3 — UNIQUE-SUBJECT ROBUSTNESS")
print("=" * 70)

print("\nInput:")
print(INPUT_CSV)


if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"\n파일을 찾을 수 없습니다:\n{INPUT_CSV}"
    )


df = pd.read_csv(
    INPUT_CSV,
    encoding="utf-8-sig"
)


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

x_col = "delta_adjusted_R2_12B"
y_col = "delta_adjusted_R2_12C"

required = [
    x_col,
    y_col
]


missing = [
    col
    for col in required
    if col not in df.columns
]


if missing:

    raise ValueError(
        "\n필요한 열이 없습니다:\n"
        + "\n".join(missing)
        + "\n\n현재 CSV 열:\n"
        + "\n".join(df.columns.astype(str))
    )


# ============================================================
# 4. NUMERIC CONVERSION
# ============================================================

df[x_col] = pd.to_numeric(
    df[x_col],
    errors="coerce"
)

df[y_col] = pd.to_numeric(
    df[y_col],
    errors="coerce"
)


plot_df = (
    df[
        [
            x_col,
            y_col
        ]
    ]
    .dropna()
    .copy()
)


print(
    "\nNumber of feature-context combinations:",
    len(plot_df)
)


# ============================================================
# 5. SPEARMAN CORRELATION
# ============================================================

rho, p_value = spearmanr(
    plot_df[x_col],
    plot_df[y_col]
)


print("\nRobustness agreement")
print("-" * 40)

print(
    f"Spearman rho = {rho:.4f}"
)

print(
    f"p-value      = {p_value:.4g}"
)


# ============================================================
# 6. AXIS RANGE
# ============================================================

minimum = float(
    min(
        plot_df[x_col].min(),
        plot_df[y_col].min()
    )
)

maximum = float(
    max(
        plot_df[x_col].max(),
        plot_df[y_col].max()
    )
)


# 약간의 여백 추가
data_range = maximum - minimum

padding = (
    data_range * 0.04
    if data_range > 0
    else 0.001
)

axis_min = minimum - padding
axis_max = maximum + padding


# ============================================================
# 7. CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.2, 7.0)
)


# ------------------------------------------------------------
# Scatter points
# ------------------------------------------------------------

ax.scatter(
    plot_df[x_col],
    plot_df[y_col],
    s=48,
    alpha=0.9
)


# ------------------------------------------------------------
# Identity line: y = x
# ------------------------------------------------------------

ax.plot(
    [axis_min, axis_max],
    [axis_min, axis_max],
    linewidth=1.8
)


# ============================================================
# 8. AXIS LIMITS
# ============================================================

ax.set_xlim(
    axis_min,
    axis_max
)

ax.set_ylim(
    axis_min,
    axis_max
)


# 같은 scale을 사용하여 y=x가 정확한 대각선이 되게 함
ax.set_aspect(
    "equal",
    adjustable="box"
)


# ============================================================
# 9. AXIS LABELS
# ============================================================

ax.set_xlabel(
    "Primary analysis Δadjusted R²",
    fontsize=13,
    labelpad=10
)


ax.set_ylabel(
    "Unique-subject sensitivity analysis Δadjusted R²",
    fontsize=13,
    labelpad=10
)


# ============================================================
# 10. TICK STYLE
# ============================================================

ax.tick_params(
    axis="both",
    which="major",
    labelsize=11
)


# ============================================================
# 11. NO FIGURE TITLE
#
# BSPC 원고에서는 제목을 figure 내부에 넣지 않고
# caption에서 설명
# ============================================================


# ============================================================
# 12. CLEAN SPINES
# ============================================================

ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)


# ============================================================
# 13. OPTIONAL AGREEMENT TEXT
#
# 그림 안에 rho를 넣고 싶다면 아래 블록을 살리면 됨.
# 현재는 caption에 넣는 방식으로 두므로 주석 처리.
# ============================================================

# ax.text(
#     0.05,
#     0.95,
#     f"Spearman ρ = {rho:.3f}",
#     transform=ax.transAxes,
#     ha="left",
#     va="top",
#     fontsize=11
# )


# ============================================================
# 14. LAYOUT
# ============================================================

fig.tight_layout()


# ============================================================
# 15. OUTPUT PATHS
# ============================================================

PNG_PATH = (
    OUTPUT_DIR
    / "Fig3_primary_vs_unique_subject_delta_adjR2.png"
)

PDF_PATH = (
    OUTPUT_DIR
    / "Fig3_primary_vs_unique_subject_delta_adjR2.pdf"
)

TIFF_PATH = (
    OUTPUT_DIR
    / "Fig3_primary_vs_unique_subject_delta_adjR2.tiff"
)


# ============================================================
# 16. SAVE
# ============================================================

fig.savefig(
    PNG_PATH,
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    PDF_PATH,
    bbox_inches="tight"
)

fig.savefig(
    TIFF_PATH,
    dpi=600,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FIGURE 3 CREATION COMPLETED")
print("=" * 70)

print("\nPNG:")
print(PNG_PATH)

print("\nPDF:")
print(PDF_PATH)

print("\nTIFF:")
print(TIFF_PATH)

print("\nFigure statistics:")
print(f"n = {len(plot_df)}")
print(f"Spearman rho = {rho:.4f}")
print(f"p = {p_value:.4g}")