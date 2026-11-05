# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import norm, rankdata, spearmanr
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


# ============================================================
# BSPC REVISION
# STEP 14B
# 30-min vs 20-min Window Sensitivity
# Common-sample adjusted analysis
# ============================================================

print("=" * 100)
print("BSPC REVISION - STEP 14B")
print("30-MIN vs 20-MIN WINDOW SENSITIVITY")
print("=" * 100)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_DIR / "data_processed"
RESULT_DIR = PROJECT_DIR / "results"

INPUT_30MIN = (
    PROCESSED_DIR
    / "07D_preoutcome_modeling_dataset_frozen.csv"
)

INPUT_20MIN = (
    PROCESSED_DIR
    / "14A_20min_coupling_features_qc_pass.csv"
)

OUT_30 = (
    RESULT_DIR
    / "14B_30min_common_adjusted_tests.csv"
)

OUT_20 = (
    RESULT_DIR
    / "14B_20min_common_adjusted_tests.csv"
)

OUT_COMPARE = (
    RESULT_DIR
    / "14B_30_vs_20_adjusted_comparison.csv"
)

OUT_MAJOR6 = (
    RESULT_DIR
    / "14B_major6_window_comparison.csv"
)

OUT_FEATURE_AGREEMENT = (
    RESULT_DIR
    / "14B_feature_value_agreement.csv"
)

OUT_CONTEXT_COUNTS = (
    RESULT_DIR
    / "14B_context_level_counts.csv"
)

OUT_SUMMARY = (
    RESULT_DIR
    / "14B_window_sensitivity_summary.csv"
)


# ============================================================
# 2. COUPLING FEATURES
# ============================================================

COUPLING12 = [
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


DOMAIN_MAP = {
    "art_pleth_lag_median_sec": "ART_PLETH",
    "art_pleth_lag_iqr_sec": "ART_PLETH",
    "art_pleth_xcorr_median": "ART_PLETH",
    "art_pleth_xcorr_iqr": "ART_PLETH",
    "art_pleth_xcorr_lag_median_sec": "ART_PLETH",
    "art_pleth_amplitude_spearman": "ART_PLETH",

    "hr_map_corr_median": "HR_MAP",
    "hr_map_corr_iqr": "HR_MAP",

    "map_etco2_corr_median": "MAP_ETCO2",
    "map_etco2_corr_iqr": "MAP_ETCO2",

    "hr_etco2_corr_median": "HR_ETCO2",
    "hr_etco2_corr_iqr": "HR_ETCO2",
}


# ============================================================
# 3. ESTIMABLE CONTEXT VARIABLES
# ============================================================

# ane_type excluded because the adjusted complete-case cohort
# contains only one anesthesia category (General).

CONTINUOUS_CONTEXT = [
    "age",
    "bmi",
]

CATEGORICAL_CONTEXT = [
    "sex",
    "asa",
    "emop",
    "department",
    "optype",
    "approach",
    "position",
    "preop_htn",
    "preop_dm",
]

ALL_CONTEXT = (
    CONTINUOUS_CONTEXT
    + CATEGORICAL_CONTEXT
)


print(
    f"\nEstimable context variables: "
    f"{len(ALL_CONTEXT)}"
)

print(
    f"Planned estimable tests per window: "
    f"{len(COUPLING12) * len(ALL_CONTEXT)}"
)


# ============================================================
# 4. HELPERS
# ============================================================

def benjamini_hochberg(p_values):

    p = np.asarray(
        p_values,
        dtype=float,
    )

    q = np.full(
        len(p),
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(p)
    idx = np.where(valid)[0]

    if len(idx) == 0:
        return q

    pv = p[idx]

    order = np.argsort(pv)
    ranked = pv[order]

    m = len(ranked)

    adjusted = (
        ranked
        * m
        / np.arange(
            1,
            m + 1,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1,
    )

    unsorted = np.empty(
        m,
        dtype=float,
    )

    unsorted[order] = adjusted

    q[idx] = unsorted

    return q


def inverse_normal_transform(series):

    x = pd.to_numeric(
        series,
        errors="coerce",
    )

    out = pd.Series(
        np.nan,
        index=x.index,
        dtype=float,
    )

    mask = x.notna()

    n = int(mask.sum())

    if n == 0:
        return out

    ranks = rankdata(
        x.loc[mask].to_numpy(
            dtype=float
        ),
        method="average",
    )

    probs = (
        ranks - 0.375
    ) / (
        n + 0.25
    )

    probs = np.clip(
        probs,
        1e-6,
        1 - 1e-6,
    )

    out.loc[mask] = norm.ppf(
        probs
    )

    return out


def standardize(series):

    x = pd.to_numeric(
        series,
        errors="coerce",
    )

    mean = x.mean()
    sd = x.std(ddof=1)

    if (
        pd.isna(sd)
        or sd == 0
    ):
        return pd.Series(
            np.nan,
            index=x.index,
            dtype=float,
        )

    return (
        x - mean
    ) / sd


def term_to_formula(term):

    if term in CONTINUOUS_CONTEXT:
        return f"{term}_z"

    return f"C({term})"


def get_term_columns(
    design_info,
    term_label,
):

    slices = (
        design_info
        .term_name_slices
    )

    if term_label not in slices:
        return []

    sl = slices[
        term_label
    ]

    return list(
        range(
            sl.start,
            sl.stop,
        )
    )


def robust_wald_for_term(
    result,
    design_info,
    term_label,
):

    col_indices = get_term_columns(
        design_info,
        term_label,
    )

    if len(col_indices) == 0:

        return (
            np.nan,
            np.nan,
            0,
        )

    p = len(
        result.params
    )

    R = np.zeros(
        (
            len(col_indices),
            p,
        ),
        dtype=float,
    )

    for row_i, col_i in enumerate(
        col_indices
    ):

        R[
            row_i,
            col_i
        ] = 1.0

    test = result.wald_test(
        R,
        scalar=True,
    )

    statistic = float(
        np.asarray(
            test.statistic
        ).squeeze()
    )

    p_value = float(
        np.asarray(
            test.pvalue
        ).squeeze()
    )

    return (
        statistic,
        p_value,
        len(col_indices),
    )


def reduced_formula(
    full_terms,
    dropped_term,
):

    kept = [
        t
        for t in full_terms
        if t != dropped_term
    ]

    if len(kept) == 0:
        return "coupling_z ~ 1"

    return (
        "coupling_z ~ "
        + " + ".join(kept)
    )


# ============================================================
# 5. LOAD DATA
# ============================================================

if not INPUT_30MIN.exists():

    raise FileNotFoundError(
        f"30-min dataset missing:\n"
        f"{INPUT_30MIN}"
    )


if not INPUT_20MIN.exists():

    raise FileNotFoundError(
        f"20-min dataset missing:\n"
        f"{INPUT_20MIN}"
    )


primary30 = pd.read_csv(
    INPUT_30MIN
)

twenty = pd.read_csv(
    INPUT_20MIN
)


primary30.columns = (
    primary30.columns
    .str.replace(
        "\ufeff",
        "",
        regex=False,
    )
    .str.strip()
)


twenty.columns = (
    twenty.columns
    .str.replace(
        "\ufeff",
        "",
        regex=False,
    )
    .str.strip()
)


print("\n" + "=" * 100)
print("INPUT DATA")
print("=" * 100)

print(
    f"Primary 30-min dataset: "
    f"{len(primary30):,}"
)

print(
    f"20-min QC-pass dataset: "
    f"{len(twenty):,}"
)


# ============================================================
# 6. REQUIRED COLUMNS
# ============================================================

required30 = (
    [
        "caseid",
        "subjectid",
    ]
    + COUPLING12
    + ALL_CONTEXT
)

missing30 = [
    c
    for c in required30
    if c not in primary30.columns
]

if missing30:

    raise ValueError(
        "30-min dataset missing columns:\n"
        + "\n".join(
            missing30
        )
    )


required20 = (
    ["caseid"]
    + COUPLING12
)

missing20 = [
    c
    for c in required20
    if c not in twenty.columns
]

if missing20:

    raise ValueError(
        "20-min dataset missing columns:\n"
        + "\n".join(
            missing20
        )
    )


# ============================================================
# 7. CASE ID CLEANING
# ============================================================

primary30["caseid"] = pd.to_numeric(
    primary30["caseid"],
    errors="coerce",
)

twenty["caseid"] = pd.to_numeric(
    twenty["caseid"],
    errors="coerce",
)

primary30 = (
    primary30
    .dropna(
        subset=["caseid"]
    )
    .copy()
)

twenty = (
    twenty
    .dropna(
        subset=["caseid"]
    )
    .copy()
)

primary30["caseid"] = (
    primary30["caseid"]
    .astype(int)
)

twenty["caseid"] = (
    twenty["caseid"]
    .astype(int)
)


# ============================================================
# 8. PREPARE 20-MIN FEATURE NAMES
# ============================================================

twenty_features = twenty[
    ["caseid"] + COUPLING12
].copy()


rename20 = {
    feature:
        f"{feature}_20min"
    for feature in COUPLING12
}


twenty_features = (
    twenty_features
    .rename(
        columns=rename20
    )
)


# ============================================================
# 9. COMMON WINDOW COHORT
# ============================================================

common = primary30.merge(
    twenty_features,
    on="caseid",
    how="inner",
    validate="one_to_one",
)


print(
    f"\nCases present in both "
    f"30-min and 20-min datasets: "
    f"{len(common):,}"
)


# ============================================================
# 10. COMPLETE-CASE FILTER
# ============================================================

complete_columns = (
    ["subjectid"]
    + ALL_CONTEXT
    + COUPLING12
    + [
        f"{feature}_20min"
        for feature in COUPLING12
    ]
)


complete_mask = (
    common[
        complete_columns
    ]
    .notna()
    .all(axis=1)
)


common = (
    common.loc[
        complete_mask
    ]
    .copy()
    .reset_index(drop=True)
)


print(
    f"Common complete-case cases: "
    f"{len(common):,}"
)

print(
    f"Common complete-case subjects: "
    f"{common['subjectid'].nunique():,}"
)


# ============================================================
# 11. CHECK CATEGORICAL LEVELS
# ============================================================

level_rows = []


print("\n" + "=" * 100)
print("CONTEXT LEVEL CHECK")
print("=" * 100)


for context in CATEGORICAL_CONTEXT:

    counts = (
        common[
            context
        ]
        .value_counts(
            dropna=False
        )
    )

    n_levels = common[
        context
    ].nunique(
        dropna=True
    )

    print(
        f"\n{context}: "
        f"{n_levels} levels"
    )

    print(
        counts.to_string()
    )

    level_rows.append({
        "context_variable":
            context,

        "n_levels":
            n_levels,

        "minimum_level_n":
            (
                int(
                    counts.min()
                )
                if len(counts)
                else np.nan
            ),

        "maximum_level_n":
            (
                int(
                    counts.max()
                )
                if len(counts)
                else np.nan
            ),
    })

    if n_levels < 2:

        raise ValueError(
            f"{context} has <2 levels "
            "in common cohort."
        )


pd.DataFrame(
    level_rows
).to_csv(
    OUT_CONTEXT_COUNTS,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 12. STANDARDIZE CONTINUOUS COVARIATES
# ============================================================

for c in CONTINUOUS_CONTEXT:

    common[
        f"{c}_z"
    ] = standardize(
        common[c]
    )


# ============================================================
# 13. MODEL FORMULA
# ============================================================

FULL_TERMS = (
    [
        f"{c}_z"
        for c
        in CONTINUOUS_CONTEXT
    ]
    + [
        f"C({c})"
        for c
        in CATEGORICAL_CONTEXT
    ]
)


FULL_FORMULA = (
    "coupling_z ~ "
    + " + ".join(
        FULL_TERMS
    )
)


print("\nFull adjusted formula:")
print(FULL_FORMULA)


# ============================================================
# 14. RUN ONE WINDOW
# ============================================================

def run_window_analysis(
    data,
    window_label,
):

    print("\n" + "=" * 100)

    print(
        f"ADJUSTED ANALYSIS: "
        f"{window_label}"
    )

    print("=" * 100)


    rows = []


    for feature_index, feature in enumerate(
        COUPLING12,
        start=1,
    ):

        source_col = (
            feature
            if window_label == "30min"
            else f"{feature}_20min"
        )


        temp = data.copy()


        temp[
            "coupling_z"
        ] = inverse_normal_transform(
            temp[source_col]
        )


        temp = temp.dropna(
            subset=[
                "coupling_z",
                "subjectid",
            ]
        ).copy()


        # Ordinary OLS for R2
        ols_model = smf.ols(
            formula=FULL_FORMULA,
            data=temp,
        )


        ols_result = (
            ols_model.fit()
        )


        # Subject-cluster robust inference
        robust_result = (
            ols_model.fit(
                cov_type="cluster",
                cov_kwds={
                    "groups":
                        temp[
                            "subjectid"
                        ],

                    "use_correction":
                        True,
                },
            )
        )


        design_info = (
            ols_model
            .data
            .design_info
        )


        full_adj_r2 = float(
            ols_result
            .rsquared_adj
        )


        full_r2 = float(
            ols_result
            .rsquared
        )


        params = (
            robust_result
            .params
        )


        # ----------------------------------------------------
        # Omnibus context tests
        # ----------------------------------------------------

        for context in ALL_CONTEXT:

            formula_term = (
                term_to_formula(
                    context
                )
            )


            (
                stat,
                p_value,
                df_term,
            ) = robust_wald_for_term(
                robust_result,
                design_info,
                formula_term,
            )


            reduced = reduced_formula(
                FULL_TERMS,
                formula_term,
            )


            reduced_result = smf.ols(
                formula=reduced,
                data=temp,
            ).fit()


            delta_adj_r2 = (
                full_adj_r2
                - float(
                    reduced_result
                    .rsquared_adj
                )
            )


            delta_r2 = (
                full_r2
                - float(
                    reduced_result
                    .rsquared
                )
            )


            single_beta = np.nan


            if context in CONTINUOUS_CONTEXT:

                param_name = (
                    f"{context}_z"
                )

                if param_name in params.index:

                    single_beta = float(
                        params[
                            param_name
                        ]
                    )


            rows.append({
                "window":
                    window_label,

                "coupling_feature":
                    feature,

                "domain":
                    DOMAIN_MAP[
                        feature
                    ],

                "context_variable":
                    context,

                "context_type":
                    (
                        "continuous"
                        if context
                        in CONTINUOUS_CONTEXT
                        else "categorical"
                    ),

                "n_cases":
                    len(temp),

                "n_subjects":
                    temp[
                        "subjectid"
                    ].nunique(),

                "wald_chi2":
                    stat,

                "wald_df":
                    df_term,

                "p_value_cluster_robust":
                    p_value,

                "delta_R2":
                    delta_r2,

                "delta_adjusted_R2":
                    delta_adj_r2,

                "standardized_beta_if_single_df":
                    single_beta,

                "full_model_adjusted_R2":
                    full_adj_r2,
            })


        print(
            f"{feature_index:02d}/12 "
            f"{feature:<38} "
            f"N={len(temp):,} "
            f"subjects="
            f"{temp['subjectid'].nunique():,} "
            f"adjR2="
            f"{full_adj_r2:.4f}"
        )


    result = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # BH-FDR
    # --------------------------------------------------------

    result[
        "q_value_BH"
    ] = benjamini_hochberg(
        result[
            "p_value_cluster_robust"
        ].to_numpy(
            dtype=float
        )
    )


    result[
        "BH_q_lt_0_05"
    ] = (
        result[
            "q_value_BH"
        ] < 0.05
    )


    result[
        "BH_q_lt_0_01"
    ] = (
        result[
            "q_value_BH"
        ] < 0.01
    )


    return result


# ============================================================
# 15. RUN 30-MIN AND 20-MIN ON IDENTICAL SAMPLE
# ============================================================

result30 = run_window_analysis(
    common,
    "30min",
)


result20 = run_window_analysis(
    common,
    "20min",
)


# ============================================================
# 16. VALID TEST CHECK
# ============================================================

valid30 = int(
    result30[
        "p_value_cluster_robust"
    ].notna().sum()
)


valid20 = int(
    result20[
        "p_value_cluster_robust"
    ].notna().sum()
)


print("\n" + "=" * 100)
print("VALID TEST CHECK")
print("=" * 100)


print(
    f"30-min valid tests: "
    f"{valid30}/132"
)


print(
    f"20-min valid tests: "
    f"{valid20}/132"
)


# ============================================================
# 17. SAVE WINDOW-SPECIFIC RESULTS
# ============================================================

result30.to_csv(
    OUT_30,
    index=False,
    encoding="utf-8-sig",
)


result20.to_csv(
    OUT_20,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 18. COMPARE 30 vs 20
# ============================================================

compare = result30.merge(
    result20,
    on=[
        "coupling_feature",
        "domain",
        "context_variable",
        "context_type",
    ],
    how="inner",
    suffixes=(
        "_30min",
        "_20min",
    ),
)


compare[
    "significant_both_q05"
] = (
    compare[
        "BH_q_lt_0_05_30min"
    ]
    &
    compare[
        "BH_q_lt_0_05_20min"
    ]
)


# ============================================================
# 19. FIXED BOOLEAN DIRECTION COLUMN
# ============================================================

continuous_mask = (
    compare[
        "context_type"
    ] == "continuous"
)


# IMPORTANT:
# Nullable boolean dtype avoids pandas 3.x / Python 3.14
# float-to-bool assignment error.
compare[
    "same_continuous_beta_direction"
] = pd.Series(
    pd.NA,
    index=compare.index,
    dtype="boolean",
)


beta30 = pd.to_numeric(
    compare.loc[
        continuous_mask,
        "standardized_beta_if_single_df_30min"
    ],
    errors="coerce",
)


beta20 = pd.to_numeric(
    compare.loc[
        continuous_mask,
        "standardized_beta_if_single_df_20min"
    ],
    errors="coerce",
)


valid_beta = (
    beta30.notna()
    & beta20.notna()
)


valid_beta_indices = (
    beta30.index[
        valid_beta
    ]
)


same_direction_values = (
    np.sign(
        beta30.loc[
            valid_beta_indices
        ]
    )
    ==
    np.sign(
        beta20.loc[
            valid_beta_indices
        ]
    )
)


compare.loc[
    valid_beta_indices,
    "same_continuous_beta_direction"
] = (
    same_direction_values
    .astype(bool)
    .to_numpy()
)


compare.to_csv(
    OUT_COMPARE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 20. Δadjusted R2 CONCORDANCE
# ============================================================

valid_delta = (
    compare[
        "delta_adjusted_R2_30min"
    ].notna()
    &
    compare[
        "delta_adjusted_R2_20min"
    ].notna()
)


rho_delta, p_delta = spearmanr(
    compare.loc[
        valid_delta,
        "delta_adjusted_R2_30min",
    ],
    compare.loc[
        valid_delta,
        "delta_adjusted_R2_20min",
    ],
)


# ============================================================
# 21. SIGNIFICANCE RETENTION
# ============================================================

n30_sig = int(
    compare[
        "BH_q_lt_0_05_30min"
    ].sum()
)


n20_sig = int(
    compare[
        "BH_q_lt_0_05_20min"
    ].sum()
)


n_both_sig = int(
    compare[
        "significant_both_q05"
    ].sum()
)


retention_percent = (
    n_both_sig
    / n30_sig
    * 100
    if n30_sig > 0
    else np.nan
)


# ============================================================
# 22. MAJOR SIX
# ============================================================

MAJOR6 = [
    (
        "art_pleth_xcorr_iqr",
        "optype",
    ),
    (
        "hr_map_corr_median",
        "optype",
    ),
    (
        "map_etco2_corr_iqr",
        "approach",
    ),
    (
        "hr_etco2_corr_iqr",
        "approach",
    ),
    (
        "hr_etco2_corr_median",
        "optype",
    ),
    (
        "art_pleth_xcorr_median",
        "age",
    ),
]


major_rows = []


for feature, context in MAJOR6:

    row = compare.loc[
        (
            compare[
                "coupling_feature"
            ] == feature
        )
        &
        (
            compare[
                "context_variable"
            ] == context
        )
    ]


    if len(row) != 1:

        raise ValueError(
            "Major association "
            f"not uniquely found: "
            f"{feature} / {context}"
        )


    r = row.iloc[0]


    major_rows.append({
        "coupling_feature":
            feature,

        "context_variable":
            context,

        "delta_adjusted_R2_30min":
            r[
                "delta_adjusted_R2_30min"
            ],

        "delta_adjusted_R2_20min":
            r[
                "delta_adjusted_R2_20min"
            ],

        "q_BH_30min":
            r[
                "q_value_BH_30min"
            ],

        "q_BH_20min":
            r[
                "q_value_BH_20min"
            ],

        "significant_30min":
            r[
                "BH_q_lt_0_05_30min"
            ],

        "significant_20min":
            r[
                "BH_q_lt_0_05_20min"
            ],

        "standardized_beta_30min":
            r[
                "standardized_beta_if_single_df_30min"
            ],

        "standardized_beta_20min":
            r[
                "standardized_beta_if_single_df_20min"
            ],
    })


major_df = pd.DataFrame(
    major_rows
)


major_df[
    "retained_at_20min"
] = (
    major_df[
        "significant_20min"
    ].astype(bool)
)


major_retained = int(
    major_df[
        "retained_at_20min"
    ].sum()
)


major_df.to_csv(
    OUT_MAJOR6,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 23. FEATURE-VALUE AGREEMENT
# ============================================================

agreement_rows = []


for feature in COUPLING12:

    f20 = (
        f"{feature}_20min"
    )


    valid = (
        common[
            feature
        ].notna()
        &
        common[
            f20
        ].notna()
    )


    rho, p = spearmanr(
        common.loc[
            valid,
            feature,
        ],
        common.loc[
            valid,
            f20,
        ],
    )


    agreement_rows.append({
        "coupling_feature":
            feature,

        "domain":
            DOMAIN_MAP[
                feature
            ],

        "n_cases":
            int(
                valid.sum()
            ),

        "spearman_rho_30_vs_20":
            float(rho),

        "p_value":
            float(p),

        "median_30min":
            float(
                common.loc[
                    valid,
                    feature,
                ].median()
            ),

        "median_20min":
            float(
                common.loc[
                    valid,
                    f20,
                ].median()
            ),
    })


agreement_df = pd.DataFrame(
    agreement_rows
)


agreement_df.to_csv(
    OUT_FEATURE_AGREEMENT,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 24. CONTINUOUS EFFECT DIRECTION
# ============================================================

continuous_compare = (
    compare.loc[
        continuous_mask
    ]
    .copy()
)


same_direction_n = int(
    continuous_compare[
        "same_continuous_beta_direction"
    ]
    .fillna(False)
    .astype(bool)
    .sum()
)


continuous_total = int(
    continuous_compare[
        "same_continuous_beta_direction"
    ]
    .notna()
    .sum()
)


# ============================================================
# 25. SUMMARY
# ============================================================

summary_df = pd.DataFrame([
    {
        "20min_qc_pass_cases":
            len(twenty),

        "common_complete_cases":
            len(common),

        "common_complete_subjects":
            common[
                "subjectid"
            ].nunique(),

        "estimable_context_variables":
            len(
                ALL_CONTEXT
            ),

        "tests_per_window":
            len(COUPLING12)
            * len(ALL_CONTEXT),

        "valid_30min_tests":
            valid30,

        "valid_20min_tests":
            valid20,

        "30min_BH_q05":
            n30_sig,

        "20min_BH_q05":
            n20_sig,

        "significant_both_q05":
            n_both_sig,

        "30min_significance_retention_percent":
            retention_percent,

        "delta_adjR2_spearman_rho":
            rho_delta,

        "delta_adjR2_spearman_p":
            p_delta,

        "major6_retained_20min":
            major_retained,

        "major6_total":
            len(MAJOR6),

        "continuous_beta_same_direction_n":
            same_direction_n,

        "continuous_beta_total":
            continuous_total,
    }
])


summary_df.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 26. CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 100)
print("WINDOW SENSITIVITY SUMMARY")
print("=" * 100)


print(
    f"\nCommon complete cases: "
    f"{len(common):,}"
)


print(
    f"Common subjects: "
    f"{common['subjectid'].nunique():,}"
)


print(
    f"\n30-min BH q<0.05: "
    f"{n30_sig}"
)


print(
    f"20-min BH q<0.05: "
    f"{n20_sig}"
)


print(
    f"Significant in both: "
    f"{n_both_sig}"
)


print(
    f"30-min significance retained "
    f"at 20 min: "
    f"{retention_percent:.1f}%"
)


print(
    f"\nΔadjusted R² "
    f"30 vs 20 Spearman rho: "
    f"{rho_delta:.4f}"
)


print(
    f"p-value: "
    f"{p_delta:.3e}"
)


print(
    f"\nMajor six retained at 20 min: "
    f"{major_retained}/6"
)


print(
    f"Continuous beta directions "
    f"consistent: "
    f"{same_direction_n}/"
    f"{continuous_total}"
)


# ============================================================
# 27. MAJOR SIX OUTPUT
# ============================================================

print("\n" + "=" * 100)
print("MAJOR SIX WINDOW COMPARISON")
print("=" * 100)


print(
    major_df.to_string(
        index=False
    )
)


# ============================================================
# 28. FEATURE AGREEMENT OUTPUT
# ============================================================

print("\n" + "=" * 100)
print("FEATURE-LEVEL 30 vs 20 AGREEMENT")
print("=" * 100)


print(
    agreement_df[
        [
            "coupling_feature",
            "spearman_rho_30_vs_20",
            "median_30min",
            "median_20min",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 29. SAVED FILES
# ============================================================

print("\n" + "=" * 100)
print("SAVED FILES")
print("=" * 100)


for path in [
    OUT_30,
    OUT_20,
    OUT_COMPARE,
    OUT_MAJOR6,
    OUT_FEATURE_AGREEMENT,
    OUT_CONTEXT_COUNTS,
    OUT_SUMMARY,
]:

    print(path)


print("\nSTEP 14B COMPLETE")