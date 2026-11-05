# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    mannwhitneyu,
    chi2_contingency,
)

warnings.filterwarnings("ignore")


# ============================================================
# BSPC REVISION
# STEP 15
# QC Selection-Bias / Generalizability Check
# ============================================================

print("=" * 100)
print("BSPC REVISION - STEP 15")
print("QC SELECTION-BIAS CHECK")
print("=" * 100)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_DIR / "data_raw"
PROCESSED_DIR = PROJECT_DIR / "data_processed"
RESULT_DIR = PROJECT_DIR / "results"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


INPUT_ELIGIBLE = (
    PROCESSED_DIR
    / "02_numeric_quality_eligible_cases.csv"
)

INPUT_STEP05_ALL = (
    PROCESSED_DIR
    / "05_locked_coupling_features_all_cases.csv"
)

INPUT_STEP05_ERRORS = (
    RESULT_DIR
    / "05_locked_coupling_feature_errors.csv"
)


OUT_SOURCE_AUDIT = (
    RESULT_DIR
    / "15_QC_selection_bias_source_audit.csv"
)

OUT_PRIMARY = (
    RESULT_DIR
    / "15_QC_pass_vs_fail_effect_sizes.csv"
)

OUT_PRIMARY_LEVELS = (
    RESULT_DIR
    / "15_QC_pass_vs_fail_category_levels.csv"
)

OUT_OVERALL = (
    RESULT_DIR
    / "15_included_vs_all_excluded_effect_sizes.csv"
)

OUT_OVERALL_LEVELS = (
    RESULT_DIR
    / "15_included_vs_all_excluded_category_levels.csv"
)

OUT_MISSINGNESS = (
    RESULT_DIR
    / "15_QC_selection_bias_missingness.csv"
)

OUT_SUMMARY = (
    RESULT_DIR
    / "15_QC_selection_bias_summary.csv"
)


# ============================================================
# 2. VARIABLES
# ============================================================

CONTINUOUS_VARIABLES = [
    "age",
    "bmi",
]


CATEGORICAL_VARIABLES = [
    "sex",
    "asa",
    "emop",
    "department",
    "ane_type",
    "optype",
    "approach",
    "position",
    "preop_htn",
    "preop_dm",
]


ALL_CLINICAL_VARIABLES = (
    CONTINUOUS_VARIABLES
    + CATEGORICAL_VARIABLES
)


# Effect-size screening threshold.
# This is used only as a descriptive review flag,
# not as a formal significance threshold.
REVIEW_THRESHOLD = 0.10


# Primary 30-min window requirement
WINDOW_SEC = 30 * 60


# ============================================================
# 3. HELPERS
# ============================================================

def clean_columns(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace(
            "\ufeff",
            "",
            regex=False,
        )
        .str.strip()
    )

    return df


def standardized_mean_difference(
    group1,
    group0,
):

    x1 = pd.to_numeric(
        group1,
        errors="coerce",
    ).dropna()

    x0 = pd.to_numeric(
        group0,
        errors="coerce",
    ).dropna()

    if (
        len(x1) < 2
        or len(x0) < 2
    ):
        return np.nan

    mean1 = float(
        x1.mean()
    )

    mean0 = float(
        x0.mean()
    )

    var1 = float(
        x1.var(ddof=1)
    )

    var0 = float(
        x0.var(ddof=1)
    )

    pooled_sd = np.sqrt(
        (
            var1
            + var0
        )
        / 2.0
    )

    if (
        not np.isfinite(
            pooled_sd
        )
        or pooled_sd == 0
    ):
        return np.nan

    return float(
        (
            mean1
            - mean0
        )
        / pooled_sd
    )


def bias_corrected_cramers_v(
    x,
    y,
):

    temp = pd.DataFrame({
        "group": x,
        "value": y,
    }).dropna()

    if len(temp) == 0:
        return np.nan

    table = pd.crosstab(
        temp["group"],
        temp["value"],
    )

    if (
        table.shape[0] < 2
        or table.shape[1] < 2
    ):
        return np.nan

    observed = table.to_numpy(
        dtype=float
    )

    n = observed.sum()

    if n <= 1:
        return np.nan

    try:

        chi2, _, _, _ = (
            chi2_contingency(
                observed,
                correction=False,
            )
        )

    except Exception:

        return np.nan

    phi2 = (
        chi2
        / n
    )

    r, k = observed.shape

    if n <= 1:
        return np.nan

    phi2_corrected = max(
        0.0,
        phi2
        - (
            (
                k - 1
            )
            * (
                r - 1
            )
        )
        / (
            n - 1
        ),
    )

    r_corrected = (
        r
        - (
            (
                r - 1
            ) ** 2
        )
        / (
            n - 1
        )
    )

    k_corrected = (
        k
        - (
            (
                k - 1
            ) ** 2
        )
        / (
            n - 1
        )
    )

    denominator = min(
        r_corrected - 1,
        k_corrected - 1,
    )

    if denominator <= 0:
        return np.nan

    return float(
        np.sqrt(
            phi2_corrected
            / denominator
        )
    )


def safe_mannwhitney(
    group1,
    group0,
):

    x1 = pd.to_numeric(
        group1,
        errors="coerce",
    ).dropna()

    x0 = pd.to_numeric(
        group0,
        errors="coerce",
    ).dropna()

    if (
        len(x1) == 0
        or len(x0) == 0
    ):
        return np.nan

    try:

        result = mannwhitneyu(
            x1,
            x0,
            alternative="two-sided",
        )

        return float(
            result.pvalue
        )

    except Exception:

        return np.nan


def safe_chi_square(
    group,
    variable,
):

    temp = pd.DataFrame({
        "group": group,
        "value": variable,
    }).dropna()

    if len(temp) == 0:
        return np.nan

    table = pd.crosstab(
        temp["group"],
        temp["value"],
    )

    if (
        table.shape[0] < 2
        or table.shape[1] < 2
    ):
        return np.nan

    try:

        _, p_value, _, _ = (
            chi2_contingency(
                table,
                correction=False,
            )
        )

        return float(
            p_value
        )

    except Exception:

        return np.nan


def effect_label(value):

    if pd.isna(value):
        return "not_estimable"

    value = abs(
        float(value)
    )

    if value < 0.10:
        return "negligible"

    if value < 0.20:
        return "small"

    if value < 0.50:
        return "moderate"

    return "large"


# ============================================================
# 4. CHECK REQUIRED INPUT FILES
# ============================================================

for path in [
    INPUT_ELIGIBLE,
    INPUT_STEP05_ALL,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{path}"
        )


# ============================================================
# 5. LOAD STEP05 STATUS
# ============================================================

step05 = pd.read_csv(
    INPUT_STEP05_ALL
)

step05 = clean_columns(
    step05
)


required_step05 = [
    "caseid",
    "qc_pass",
]


missing_step05 = [
    col
    for col
    in required_step05
    if col
    not in step05.columns
]


if missing_step05:

    raise ValueError(
        "STEP05 all-case file is missing:\n"
        + "\n".join(
            missing_step05
        )
    )


step05["caseid"] = pd.to_numeric(
    step05["caseid"],
    errors="coerce",
)


step05["qc_pass"] = pd.to_numeric(
    step05["qc_pass"],
    errors="coerce",
)


step05 = (
    step05
    .dropna(
        subset=[
            "caseid",
            "qc_pass",
        ]
    )
    .copy()
)


step05["caseid"] = (
    step05["caseid"]
    .astype(int)
)


step05["qc_pass"] = (
    step05["qc_pass"]
    .astype(int)
)


step05 = (
    step05
    .drop_duplicates(
        subset=["caseid"],
        keep="last",
    )
    .copy()
)


step05["qc_group"] = np.where(
    step05["qc_pass"] == 1,
    "QC_PASS",
    "QC_FAIL",
)


print("\n" + "=" * 100)
print("STEP05 QC STATUS")
print("=" * 100)


print(
    step05[
        "qc_group"
    ]
    .value_counts()
    .to_string()
)


print(
    f"\nSuccessful extractions: "
    f"{len(step05):,}"
)


# ============================================================
# 6. LOAD STEP05 ERRORS
# ============================================================

error_caseids = set()


if INPUT_STEP05_ERRORS.exists():

    error_df = pd.read_csv(
        INPUT_STEP05_ERRORS
    )

    error_df = clean_columns(
        error_df
    )

    if (
        "caseid"
        in error_df.columns
    ):

        error_df[
            "caseid"
        ] = pd.to_numeric(
            error_df[
                "caseid"
            ],
            errors="coerce",
        )

        error_df = (
            error_df
            .dropna(
                subset=["caseid"]
            )
            .copy()
        )

        error_df[
            "caseid"
        ] = (
            error_df[
                "caseid"
            ]
            .astype(int)
        )

        error_caseids = set(
            error_df[
                "caseid"
            ]
            .unique()
            .tolist()
        )


print(
    f"Extraction error cases: "
    f"{len(error_caseids):,}"
)


# ============================================================
# 7. LOAD ELIGIBLE COHORT
# ============================================================

eligible = pd.read_csv(
    INPUT_ELIGIBLE
)

eligible = clean_columns(
    eligible
)


required_eligible = [
    "caseid",
    "opstart_sec",
    "opend_sec",
]


missing_eligible = [
    col
    for col
    in required_eligible
    if col
    not in eligible.columns
]


if missing_eligible:

    raise ValueError(
        "Eligible cohort file missing:\n"
        + "\n".join(
            missing_eligible
        )
    )


eligible[
    "caseid"
] = pd.to_numeric(
    eligible[
        "caseid"
    ],
    errors="coerce",
)


eligible[
    "opstart_sec"
] = pd.to_numeric(
    eligible[
        "opstart_sec"
    ],
    errors="coerce",
)


eligible[
    "opend_sec"
] = pd.to_numeric(
    eligible[
        "opend_sec"
    ],
    errors="coerce",
)


eligible = (
    eligible
    .dropna(
        subset=[
            "caseid",
            "opstart_sec",
            "opend_sec",
        ]
    )
    .copy()
)


eligible[
    "caseid"
] = (
    eligible[
        "caseid"
    ]
    .astype(int)
)


# Reproduce STEP05 duration eligibility
eligible = eligible.loc[
    (
        eligible[
            "opend_sec"
        ]
        - eligible[
            "opstart_sec"
        ]
    )
    >= WINDOW_SEC
].copy()


eligible = (
    eligible
    .drop_duplicates(
        subset=["caseid"],
        keep="first",
    )
    .copy()
)


eligible_caseids = set(
    eligible[
        "caseid"
    ]
    .tolist()
)


print("\n" + "=" * 100)
print("STEP05 ELIGIBLE COHORT")
print("=" * 100)


print(
    f"Eligible first-30-min cases: "
    f"{len(eligible):,}"
)


# ============================================================
# 8. FIND BEST CLINICAL SOURCE
# ============================================================

candidate_paths = [
    INPUT_ELIGIBLE,

    PROCESSED_DIR
    / "01_primary_multimodal_candidate_cases.csv",

    PROCESSED_DIR
    / "01_full_multimodal_candidate_cases.csv",

    PROCESSED_DIR
    / "01_art_multimodal_candidate_cases.csv",

    RAW_DIR
    / "cases.csv",

    PROJECT_DIR
    / "cases.csv",
]


# Add other local processed CSV files,
# but later prioritize coverage of STEP05 eligible cases.
for path in sorted(
    PROCESSED_DIR.glob(
        "*.csv"
    )
):

    if path not in candidate_paths:

        candidate_paths.append(
            path
        )


source_rows = []

best_source = None
best_columns = []
best_overlap = -1
best_variable_count = -1


for path in candidate_paths:

    if (
        not path.exists()
        or path == INPUT_STEP05_ALL
    ):
        continue

    try:

        header = pd.read_csv(
            path,
            nrows=0,
        )

        cols = [
            str(c)
            .replace(
                "\ufeff",
                "",
            )
            .strip()
            for c
            in header.columns
        ]

        if "caseid" not in cols:
            continue


        present = [
            col
            for col
            in ALL_CLINICAL_VARIABLES
            if col in cols
        ]


        # Need at least one clinical variable.
        if len(present) == 0:
            continue


        try:

            case_only = pd.read_csv(
                path,
                usecols=["caseid"],
            )

            case_only[
                "caseid"
            ] = pd.to_numeric(
                case_only[
                    "caseid"
                ],
                errors="coerce",
            )

            case_ids = set(
                case_only[
                    "caseid"
                ]
                .dropna()
                .astype(int)
                .tolist()
            )

            overlap = len(
                eligible_caseids
                & case_ids
            )

        except Exception:

            overlap = 0


        source_rows.append({
            "path":
                str(path),

            "eligible_case_overlap":
                overlap,

            "clinical_variable_count":
                len(present),

            "clinical_variables":
                ", ".join(
                    present
                ),
        })


        # Prioritize coverage first.
        # If coverage ties, choose source with more variables.
        if (
            overlap > best_overlap
            or (
                overlap
                == best_overlap
                and len(present)
                > best_variable_count
            )
        ):

            best_source = path
            best_columns = present
            best_overlap = overlap
            best_variable_count = len(
                present
            )


    except Exception as error:

        source_rows.append({
            "path":
                str(path),

            "eligible_case_overlap":
                -1,

            "clinical_variable_count":
                -1,

            "clinical_variables":
                f"ERROR: {error}",
        })


source_audit = pd.DataFrame(
    source_rows
)


if len(source_audit) > 0:

    source_audit = (
        source_audit
        .sort_values(
            [
                "eligible_case_overlap",
                "clinical_variable_count",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


source_audit.to_csv(
    OUT_SOURCE_AUDIT,
    index=False,
    encoding="utf-8-sig",
)


print("\n" + "=" * 100)
print("CLINICAL SOURCE SEARCH")
print("=" * 100)


if len(source_audit) > 0:

    print(
        source_audit.head(
            10
        ).to_string(
            index=False
        )
    )


if best_source is None:

    raise FileNotFoundError(
        "No suitable clinical source "
        "with caseid was found."
    )


print(
    f"\nSelected clinical source:\n"
    f"{best_source}"
)


print(
    f"Eligible case overlap: "
    f"{best_overlap:,}"
)


print(
    "Clinical variables available:"
)


for variable in best_columns:

    print(
        f"- {variable}"
    )


# ============================================================
# 9. LOAD CLINICAL SOURCE
# ============================================================

clinical = pd.read_csv(
    best_source,
    usecols=(
        ["caseid"]
        + best_columns
    ),
)


clinical = clean_columns(
    clinical
)


clinical[
    "caseid"
] = pd.to_numeric(
    clinical[
        "caseid"
    ],
    errors="coerce",
)


clinical = (
    clinical
    .dropna(
        subset=[
            "caseid"
        ]
    )
    .copy()
)


clinical[
    "caseid"
] = (
    clinical[
        "caseid"
    ]
    .astype(int)
)


clinical = (
    clinical
    .drop_duplicates(
        subset=[
            "caseid"
        ],
        keep="first",
    )
    .copy()
)


# ============================================================
# 10. BUILD QC-SPECIFIC ANALYSIS COHORT
# ============================================================

qc_analysis = step05[
    [
        "caseid",
        "qc_pass",
        "qc_group",
    ]
].merge(
    clinical,
    on="caseid",
    how="left",
    validate="one_to_one",
)


print("\n" + "=" * 100)
print("PRIMARY QC COMPARISON")
print("=" * 100)


print(
    f"QC PASS: "
    f"{int((qc_analysis['qc_pass'] == 1).sum()):,}"
)


print(
    f"QC FAIL: "
    f"{int((qc_analysis['qc_pass'] == 0).sum()):,}"
)


# ============================================================
# 11. BUILD OVERALL INCLUDED-vs-EXCLUDED COHORT
# ============================================================

overall = eligible[
    ["caseid"]
].copy()


status_map = (
    step05[
        [
            "caseid",
            "qc_pass",
        ]
    ]
    .set_index(
        "caseid"
    )[
        "qc_pass"
    ]
    .to_dict()
)


def classify_case(caseid):

    if caseid in status_map:

        if status_map[
            caseid
        ] == 1:

            return "INCLUDED"

        return "QC_FAIL"

    if caseid in error_caseids:

        return "EXTRACTION_ERROR"

    return "UNKNOWN_EXCLUSION"


overall[
    "inclusion_status"
] = (
    overall[
        "caseid"
    ]
    .apply(
        classify_case
    )
)


overall[
    "included"
] = (
    overall[
        "inclusion_status"
    ]
    == "INCLUDED"
).astype(int)


overall = overall.merge(
    clinical,
    on="caseid",
    how="left",
    validate="one_to_one",
)


print("\n" + "=" * 100)
print("OVERALL INCLUSION STATUS")
print("=" * 100)


print(
    overall[
        "inclusion_status"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


# ============================================================
# 12. MISSINGNESS AUDIT
# ============================================================

missing_rows = []


for variable in best_columns:

    for group_name, subset in [
        (
            "QC_PASS",
            qc_analysis.loc[
                qc_analysis[
                    "qc_pass"
                ] == 1
            ],
        ),
        (
            "QC_FAIL",
            qc_analysis.loc[
                qc_analysis[
                    "qc_pass"
                ] == 0
            ],
        ),
        (
            "INCLUDED",
            overall.loc[
                overall[
                    "included"
                ] == 1
            ],
        ),
        (
            "ALL_EXCLUDED",
            overall.loc[
                overall[
                    "included"
                ] == 0
            ],
        ),
    ]:

        n_total = len(
            subset
        )

        n_missing = int(
            subset[
                variable
            ].isna().sum()
        )

        missing_rows.append({
            "variable":
                variable,

            "group":
                group_name,

            "n_total":
                n_total,

            "n_missing":
                n_missing,

            "missing_percent":
                (
                    n_missing
                    / n_total
                    * 100
                    if n_total > 0
                    else np.nan
                ),
        })


missing_df = pd.DataFrame(
    missing_rows
)


missing_df.to_csv(
    OUT_MISSINGNESS,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 13. COMPARISON FUNCTION
# ============================================================

def compare_groups(
    data,
    group_column,
    group1_value,
    group0_value,
):

    result_rows = []
    level_rows = []


    # --------------------------------------------------------
    # Continuous variables
    # --------------------------------------------------------

    for variable in CONTINUOUS_VARIABLES:

        if variable not in data.columns:
            continue


        group1 = data.loc[
            data[
                group_column
            ] == group1_value,
            variable,
        ]


        group0 = data.loc[
            data[
                group_column
            ] == group0_value,
            variable,
        ]


        x1 = pd.to_numeric(
            group1,
            errors="coerce",
        ).dropna()


        x0 = pd.to_numeric(
            group0,
            errors="coerce",
        ).dropna()


        smd = (
            standardized_mean_difference(
                x1,
                x0,
            )
        )


        p_value = (
            safe_mannwhitney(
                x1,
                x0,
            )
        )


        result_rows.append({
            "variable":
                variable,

            "variable_type":
                "continuous",

            "n_group1":
                len(x1),

            "n_group0":
                len(x0),

            "group1_mean":
                (
                    float(
                        x1.mean()
                    )
                    if len(x1)
                    else np.nan
                ),

            "group0_mean":
                (
                    float(
                        x0.mean()
                    )
                    if len(x0)
                    else np.nan
                ),

            "group1_median":
                (
                    float(
                        x1.median()
                    )
                    if len(x1)
                    else np.nan
                ),

            "group0_median":
                (
                    float(
                        x0.median()
                    )
                    if len(x0)
                    else np.nan
                ),

            "effect_size":
                smd,

            "absolute_effect_size":
                (
                    abs(smd)
                    if pd.notna(
                        smd
                    )
                    else np.nan
                ),

            "effect_measure":
                "standardized_mean_difference",

            "p_value":
                p_value,

            "effect_label":
                effect_label(
                    smd
                ),

            "review_flag_ge_0_10":
                (
                    bool(
                        abs(smd)
                        >= REVIEW_THRESHOLD
                    )
                    if pd.notna(
                        smd
                    )
                    else False
                ),
        })


    # --------------------------------------------------------
    # Categorical variables
    # --------------------------------------------------------

    for variable in CATEGORICAL_VARIABLES:

        if variable not in data.columns:
            continue


        temp = data[
            [
                group_column,
                variable,
            ]
        ].copy()


        temp = temp.loc[
            temp[
                group_column
            ].isin(
                [
                    group1_value,
                    group0_value,
                ]
            )
        ].copy()


        temp = temp.dropna(
            subset=[
                group_column,
                variable,
            ]
        )


        n_group1 = int(
            (
                temp[
                    group_column
                ]
                == group1_value
            ).sum()
        )


        n_group0 = int(
            (
                temp[
                    group_column
                ]
                == group0_value
            ).sum()
        )


        if (
            n_group1 == 0
            or n_group0 == 0
        ):

            continue


        cramers_v = (
            bias_corrected_cramers_v(
                temp[
                    group_column
                ],
                temp[
                    variable
                ],
            )
        )


        p_value = (
            safe_chi_square(
                temp[
                    group_column
                ],
                temp[
                    variable
                ],
            )
        )


        levels = sorted(
            temp[
                variable
            ]
            .astype(str)
            .unique()
            .tolist()
        )


        max_prop_diff = 0.0


        for level in levels:

            g1_values = (
                temp.loc[
                    temp[
                        group_column
                    ]
                    == group1_value,
                    variable,
                ]
                .astype(str)
            )


            g0_values = (
                temp.loc[
                    temp[
                        group_column
                    ]
                    == group0_value,
                    variable,
                ]
                .astype(str)
            )


            prop1 = float(
                (
                    g1_values
                    == level
                ).mean()
            )


            prop0 = float(
                (
                    g0_values
                    == level
                ).mean()
            )


            prop_diff = (
                prop1
                - prop0
            )


            max_prop_diff = max(
                max_prop_diff,
                abs(
                    prop_diff
                ),
            )


            level_rows.append({
                "variable":
                    variable,

                "level":
                    level,

                "n_group1":
                    int(
                        (
                            g1_values
                            == level
                        ).sum()
                    ),

                "n_group0":
                    int(
                        (
                            g0_values
                            == level
                        ).sum()
                    ),

                "proportion_group1":
                    prop1,

                "proportion_group0":
                    prop0,

                "proportion_difference":
                    prop_diff,

                "absolute_proportion_difference":
                    abs(
                        prop_diff
                    ),
            })


        result_rows.append({
            "variable":
                variable,

            "variable_type":
                "categorical",

            "n_group1":
                n_group1,

            "n_group0":
                n_group0,

            "group1_mean":
                np.nan,

            "group0_mean":
                np.nan,

            "group1_median":
                np.nan,

            "group0_median":
                np.nan,

            "effect_size":
                cramers_v,

            "absolute_effect_size":
                (
                    abs(
                        cramers_v
                    )
                    if pd.notna(
                        cramers_v
                    )
                    else np.nan
                ),

            "effect_measure":
                "bias_corrected_cramers_v",

            "p_value":
                p_value,

            "effect_label":
                effect_label(
                    cramers_v
                ),

            "max_absolute_proportion_difference":
                max_prop_diff,

            "review_flag_ge_0_10":
                (
                    (
                        pd.notna(
                            cramers_v
                        )
                        and abs(
                            cramers_v
                        )
                        >= REVIEW_THRESHOLD
                    )
                    or (
                        max_prop_diff
                        >= REVIEW_THRESHOLD
                    )
                ),
        })


    result_df = pd.DataFrame(
        result_rows
    )


    level_df = pd.DataFrame(
        level_rows
    )


    if len(result_df) > 0:

        result_df = (
            result_df
            .sort_values(
                [
                    "review_flag_ge_0_10",
                    "absolute_effect_size",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )


    return (
        result_df,
        level_df,
    )


# ============================================================
# 14. PRIMARY COMPARISON:
# QC PASS vs QC FAIL
# ============================================================

primary_result, primary_levels = (
    compare_groups(
        data=qc_analysis,
        group_column="qc_group",
        group1_value="QC_PASS",
        group0_value="QC_FAIL",
    )
)


primary_result[
    "comparison"
] = (
    "QC_PASS_vs_QC_FAIL"
)


primary_result.to_csv(
    OUT_PRIMARY,
    index=False,
    encoding="utf-8-sig",
)


primary_levels.to_csv(
    OUT_PRIMARY_LEVELS,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 15. SECONDARY COMPARISON:
# INCLUDED vs ALL EXCLUDED
# ============================================================

overall[
    "overall_group"
] = np.where(
    overall[
        "included"
    ] == 1,
    "INCLUDED",
    "ALL_EXCLUDED",
)


overall_result, overall_levels = (
    compare_groups(
        data=overall,
        group_column="overall_group",
        group1_value="INCLUDED",
        group0_value="ALL_EXCLUDED",
    )
)


overall_result[
    "comparison"
] = (
    "INCLUDED_vs_ALL_EXCLUDED"
)


overall_result.to_csv(
    OUT_OVERALL,
    index=False,
    encoding="utf-8-sig",
)


overall_levels.to_csv(
    OUT_OVERALL_LEVELS,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 16. SUMMARY
# ============================================================

primary_review = (
    primary_result.loc[
        primary_result[
            "review_flag_ge_0_10"
        ]
    ]
    if len(
        primary_result
    )
    else pd.DataFrame()
)


overall_review = (
    overall_result.loc[
        overall_result[
            "review_flag_ge_0_10"
        ]
    ]
    if len(
        overall_result
    )
    else pd.DataFrame()
)


max_primary_effect = (
    float(
        primary_result[
            "absolute_effect_size"
        ].max()
    )
    if (
        len(
            primary_result
        )
        and primary_result[
            "absolute_effect_size"
        ].notna().any()
    )
    else np.nan
)


max_overall_effect = (
    float(
        overall_result[
            "absolute_effect_size"
        ].max()
    )
    if (
        len(
            overall_result
        )
        and overall_result[
            "absolute_effect_size"
        ].notna().any()
    )
    else np.nan
)


summary_df = pd.DataFrame([
    {
        "step05_eligible_cases":
            len(
                eligible
            ),

        "successful_extractions":
            len(
                step05
            ),

        "qc_pass_cases":
            int(
                (
                    step05[
                        "qc_pass"
                    ]
                    == 1
                ).sum()
            ),

        "qc_fail_cases":
            int(
                (
                    step05[
                        "qc_pass"
                    ]
                    == 0
                ).sum()
            ),

        "extraction_error_cases":
            len(
                error_caseids
            ),

        "clinical_source":
            str(
                best_source
            ),

        "clinical_variables_available":
            len(
                best_columns
            ),

        "primary_variables_with_effect_ge_0_10":
            len(
                primary_review
            ),

        "primary_max_absolute_effect":
            max_primary_effect,

        "overall_variables_with_effect_ge_0_10":
            len(
                overall_review
            ),

        "overall_max_absolute_effect":
            max_overall_effect,
    }
])


summary_df.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 17. CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 100)
print("QC PASS vs QC FAIL - EFFECT SIZE SUMMARY")
print("=" * 100)


if len(primary_result) > 0:

    display_columns = [
        "variable",
        "variable_type",
        "n_group1",
        "n_group0",
        "effect_measure",
        "effect_size",
        "absolute_effect_size",
        "effect_label",
        "p_value",
        "review_flag_ge_0_10",
    ]


    print(
        primary_result[
            display_columns
        ]
        .to_string(
            index=False
        )
    )


print("\n" + "=" * 100)
print("QC PASS vs QC FAIL - VARIABLES FLAGGED FOR REVIEW")
print("=" * 100)


if len(primary_review) == 0:

    print(
        "\nNo variable exceeded "
        "the descriptive effect-size "
        "review threshold of 0.10."
    )

else:

    print(
        primary_review[
            [
                "variable",
                "effect_measure",
                "effect_size",
                "absolute_effect_size",
                "effect_label",
                "p_value",
            ]
        ]
        .to_string(
            index=False
        )
    )


print("\n" + "=" * 100)
print("INCLUDED vs ALL EXCLUDED - EFFECT SIZE SUMMARY")
print("=" * 100)


if len(overall_result) > 0:

    print(
        overall_result[
            display_columns
        ]
        .to_string(
            index=False
        )
    )


print("\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)


print(
    f"\nSTEP05 eligible cases: "
    f"{len(eligible):,}"
)


print(
    f"Successfully extracted: "
    f"{len(step05):,}"
)


print(
    f"QC PASS: "
    f"{int((step05['qc_pass'] == 1).sum()):,}"
)


print(
    f"QC FAIL: "
    f"{int((step05['qc_pass'] == 0).sum()):,}"
)


print(
    f"Extraction errors: "
    f"{len(error_caseids):,}"
)


print(
    f"\nClinical variables available: "
    f"{len(best_columns)}"
)


print(
    f"QC PASS vs FAIL variables "
    f"with effect >=0.10: "
    f"{len(primary_review)}"
)


print(
    f"Maximum absolute effect "
    f"(QC PASS vs FAIL): "
    f"{max_primary_effect:.4f}"
    if pd.notna(
        max_primary_effect
    )
    else
    "Maximum absolute effect "
    "(QC PASS vs FAIL): NA"
)


print(
    f"\nIncluded vs all excluded variables "
    f"with effect >=0.10: "
    f"{len(overall_review)}"
)


print(
    f"Maximum absolute effect "
    f"(included vs excluded): "
    f"{max_overall_effect:.4f}"
    if pd.notna(
        max_overall_effect
    )
    else
    "Maximum absolute effect "
    "(included vs excluded): NA"
)


print("\n" + "=" * 100)
print("INTERPRETATION GUIDE")
print("=" * 100)


print(
    """
Effect-size interpretation used for descriptive screening:

< 0.10  : negligible
0.10-0.19: small
0.20-0.49: moderate
>= 0.50 : large

For continuous variables:
- effect size = standardized mean difference (SMD)

For categorical variables:
- effect size = bias-corrected Cramer's V
- category-level proportion differences are also saved separately

IMPORTANT:
P-values are secondary here because large samples can make very small
differences statistically significant. The main question is whether the
magnitude of imbalance is clinically/descriptively meaningful.
"""
)


print("\n" + "=" * 100)
print("SAVED FILES")
print("=" * 100)


for path in [
    OUT_SOURCE_AUDIT,
    OUT_PRIMARY,
    OUT_PRIMARY_LEVELS,
    OUT_OVERALL,
    OUT_OVERALL_LEVELS,
    OUT_MISSINGNESS,
    OUT_SUMMARY,
]:

    print(path)


print("\nSTEP 15 COMPLETE")