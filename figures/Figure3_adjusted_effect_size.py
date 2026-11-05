from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 프로젝트 폴더
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

R2_FILE = (
    BASE_DIR
    / "results"
    / "tables"
    / "Fig2_adjusted_delta_adjR2_matrix.csv"
)

Q_FILE = (
    BASE_DIR
    / "results"
    / "tables"
    / "Fig2_adjusted_FDR_q_matrix.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "figures"
    / "Figure3_Adjusted_Effect_Size_Patterns.png"
)


# ============================================================
# 2. CSV 읽기
# ============================================================

r2 = pd.read_csv(R2_FILE, index_col=0)
q = pd.read_csv(Q_FILE, index_col=0)

print("\nΔadjusted R² matrix:")
print(r2.shape)
print(r2.columns.tolist())

print("\nFDR q-value matrix:")
print(q.shape)
print(q.columns.tolist())


# ============================================================
# 3. Clinical / surgical context 순서
# ============================================================

column_order = [
    "age",
    "bmi",
    "sex",
    "asa",
    "emop",
    "department",
    "ane_type",
    "optype",
    "approach",
    "position",
    "preop_htn",
    "preop_dm"
]

r2 = r2.reindex(columns=column_order)
q = q.reindex(columns=column_order)


# ============================================================
# 4. Anesthesia type을 NA 처리
# ============================================================

anesthesia_col = "ane_type"

r2.loc[:, anesthesia_col] = np.nan
q.loc[:, anesthesia_col] = np.nan

print(f"\nAnesthesia type 컬럼: {anesthesia_col}")
print("해당 열 전체를 NA로 처리했습니다.")


# ============================================================
# 5. 숫자형으로 변환
# ============================================================

r2 = r2.apply(pd.to_numeric, errors="coerce")
q = q.apply(pd.to_numeric, errors="coerce")


# ============================================================
# 6. 논문용 X축 표시 이름
# ============================================================

display_names = {
    "age": "Age",
    "bmi": "BMI",
    "sex": "Sex",
    "asa": "ASA",
    "emop": "Emergency surgery",
    "department": "Department",
    "ane_type": "Anesthesia type",
    "optype": "Operation type",
    "approach": "Surgical approach",
    "position": "Surgical position",
    "preop_htn": "Preoperative hypertension",
    "preop_dm": "Preoperative diabetes"
}


# ============================================================
# 7. Figure 생성
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))

# NaN을 mask 처리
masked = np.ma.masked_invalid(
    r2.to_numpy(dtype=float)
)

# 컬러맵
cmap = plt.get_cmap("viridis").copy()

# NA 셀을 회색으로 표시
cmap.set_bad("lightgray")


# ============================================================
# 8. Color scale 설정
# ============================================================

finite_values = r2.to_numpy(dtype=float)
finite_values = finite_values[np.isfinite(finite_values)]

if len(finite_values) > 0:
    vmax = max(
        0.04,
        float(np.nanmax(finite_values))
    )
else:
    vmax = 0.04


# ============================================================
# 9. Heatmap 그리기
# ============================================================

im = ax.imshow(
    masked,
    aspect="auto",
    cmap=cmap,
    vmin=0,
    vmax=vmax
)


# ============================================================
# 10. X축
# ============================================================

x_labels = [
    display_names.get(col, col)
    for col in r2.columns
]

ax.set_xticks(
    np.arange(len(r2.columns))
)

ax.set_xticklabels(
    x_labels,
    rotation=45,
    ha="right",
    fontsize=9
)


# ============================================================
# 11. Y축
# ============================================================

ax.set_yticks(
    np.arange(len(r2.index))
)

ax.set_yticklabels(
    r2.index,
    fontsize=9
)


# ============================================================
# 12. 축 제목
# ============================================================

ax.set_xlabel(
    "Clinical and surgical contexts",
    fontsize=11
)

ax.set_ylabel(
    "Multimodal physiological coupling features",
    fontsize=11
)


# ============================================================
# 13. 각 셀 숫자 표시
#
# - BH-FDR q < 0.05 → *
# - anesthesia type → NA
# - -0.000 → 0.000으로 표시
# ============================================================

for i, feature in enumerate(r2.index):

    for j, context in enumerate(r2.columns):

        value = r2.iloc[i, j]

        # ----------------------------------------------------
        # Anesthesia type
        # ----------------------------------------------------
        if context == anesthesia_col:

            ax.text(
                j,
                i,
                "NA",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black"
            )

            continue

        # ----------------------------------------------------
        # 일반 셀
        # ----------------------------------------------------
        if pd.notna(value):

            q_value = np.nan

            if (
                feature in q.index
                and context in q.columns
            ):
                q_value = q.loc[
                    feature,
                    context
                ]

            # --------------------------------------------
            # -0.000 방지
            # 소수 셋째 자리에서 0.000이 되는 작은 값은
            # 그냥 0.000으로 표시
            # --------------------------------------------

            if abs(value) < 0.0005:
                display_value = 0.0
            else:
                display_value = value

            # --------------------------------------------
            # BH-FDR 유의성 별표
            # --------------------------------------------

            if (
                pd.notna(q_value)
                and q_value < 0.05
            ):
                label = f"{display_value:.3f}*"
            else:
                label = f"{display_value:.3f}"

            # --------------------------------------------
            # 배경색에 따라 글씨색 조정
            # --------------------------------------------

            normalized = (
                value / vmax
                if vmax > 0
                else 0
            )

            if normalized > 0.55:
                text_color = "white"
            else:
                text_color = "black"

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=text_color
            )


# ============================================================
# 14. 셀 경계선
# ============================================================

ax.set_xticks(
    np.arange(
        -0.5,
        len(r2.columns),
        1
    ),
    minor=True
)

ax.set_yticks(
    np.arange(
        -0.5,
        len(r2.index),
        1
    ),
    minor=True
)

ax.grid(
    which="minor",
    linewidth=0.4,
    alpha=0.4
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)


# ============================================================
# 15. Colorbar
# ============================================================

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02
)

cbar.set_label(
    "Δadjusted R²",
    fontsize=10
)


# ============================================================
# 16. Figure 하단 설명
# ============================================================

fig.text(
    0.5,
    0.01,
    "* Benjamini–Hochberg FDR q < 0.05; NA, not estimable.",
    ha="center",
    fontsize=9
)


# ============================================================
# 17. 저장
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.tight_layout(
    rect=[0, 0.05, 1, 1]
)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 18. 완료 메시지
# ============================================================

print("\n==============================================")
print("Figure 3 생성 완료")
print("저장 위치:")
print(OUTPUT_FILE)
print("==============================================")