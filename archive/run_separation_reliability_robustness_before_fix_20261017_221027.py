from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT_ROOT
    / "research_outputs"
    / "wr_research_dataset.csv"
)

OUT = (
    PROJECT_ROOT
    / "research_outputs"
    / "model_comparison"
    / "separation_reliability_robustness"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOCKED MODEL DEFINITIONS
# ============================================================

FEATURES = {
    "A_conventional": [
        "forty",
        "vertical"
    ],
    "B_movement": [
        "speed_cv_DAGGER_ROUTE_LEFT"
    ],
    "C_combined": [
        "forty",
        "vertical",
        "speed_cv_DAGGER_ROUTE_LEFT"
    ]
}

TARGET = "mean_separation"

MIN_TARGET_ROWS = [
    0,
    10,
    20,
    30
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("SEPARATION RELIABILITY ROBUSTNESS ANALYSIS")
print("=" * 75)

print()
print("Loading research dataset...")

df = pd.read_csv(INPUT)

print(f"Dataset shape: {df.shape}")


required = [
    "nfl_id",
    "draft_year",
    "mean_separation",
    "targets",
    "separation_observed_rows",
    "forty",
    "vertical",
    "speed_cv_DAGGER_ROUTE_LEFT"
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
    )


# ============================================================
# BUILD MEDIAN SEPARATION
# ============================================================

# The research dataset contains player-level mean separation.
# If raw separation observations are not available here,
# median separation cannot be reconstructed from this file.
#
# Therefore we first check whether a median separation column
# already exists.

median_candidates = [
    "median_separation",
    "separation_median",
    "median_separation_at_pass_forward"
]

median_column = None

for candidate in median_candidates:
    if candidate in df.columns:
        median_column = candidate
        break


if median_column is None:

    print()
    print(
        "WARNING: No median separation column exists "
        "in wr_research_dataset.csv."
    )

    print(
        "Median-separation analysis will therefore be "
        "skipped in this run."
    )


# ============================================================
# COMMON FEATURE POPULATION
# ============================================================

feature_columns = [
    "forty",
    "vertical",
    "speed_cv_DAGGER_ROUTE_LEFT"
]

base = df[
    df[feature_columns].notna().all(axis=1)
].copy()

print()
print(
    "Players with complete model features:",
    len(base)
)


# ============================================================
# TARGET RELIABILITY DISTRIBUTION
# ============================================================

print()
print("=" * 75)
print("SEPARATION OBSERVATION COUNT DISTRIBUTION")
print("=" * 75)

print(
    base["separation_observed_rows"]
    .describe()
    .to_string()
)


# ============================================================
# MODEL
# ============================================================

def make_model():

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge(alpha=1.0)
            )
        ]
    )


# ============================================================
# ROBUSTNESS FUNCTION
# ============================================================

def run_experiment(
    data,
    target_column,
    population_name,
    minimum_target_rows
):

    eligible = data[
        data[target_column].notna()
        &
        (
            data["separation_observed_rows"]
            >= minimum_target_rows
        )
    ].copy()

    eligible = eligible.reset_index(drop=True)

    print()
    print("-" * 75)
    print(
        f"Population: {population_name}"
    )
    print(
        f"Minimum observed separation rows: "
        f"{minimum_target_rows}"
    )
    print(
        f"Eligible players: {len(eligible)}"
    )

    if len(eligible) < 15:

        print(
            "SKIPPED: fewer than 15 players."
        )

        return None, None

    years = (
        eligible["draft_year"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        f"Draft years: {years}"
    )

    results = []

    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

    for split_id, (
        train_idx,
        test_idx
    ) in enumerate(
        cv.split(eligible),
        start=1
    ):

        for model_name, feature_list in FEATURES.items():

            train = eligible.iloc[
                train_idx
            ]

            test = eligible.iloc[
                test_idx
            ]

            model = make_model()

            model.fit(
                train[feature_list],
                train[target_column]
            )

            prediction = model.predict(
                test[feature_list]
            )

            rmse = np.sqrt(
                mean_squared_error(
                    test[target_column],
                    prediction
                )
            )

            mae = mean_absolute_error(
                test[target_column],
                prediction
            )

            r2 = r2_score(
                test[target_column],
                prediction
            )

            results.append(
                {
                    "population": population_name,
                    "minimum_target_rows": (
                        minimum_target_rows
                    ),
                    "target": target_column,
                    "split_id": split_id,
                    "model": model_name,
                    "n_players": len(eligible),
                    "n_train": len(train),
                    "n_test": len(test),
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                }
            )


    results = pd.DataFrame(results)


    summary = (
        results
        .groupby("model")
        .agg(
            n_folds=("split_id", "count"),
            n_players=("n_players", "first"),

            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),

            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),

            r2_mean=("r2", "mean"),
            r2_std=("r2", "std")
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # Paired comparison against conventional baseline
    # --------------------------------------------------------

    wide = results.pivot(
        index="split_id",
        columns="model",
        values=[
            "rmse",
            "mae",
            "r2"
        ]
    )

    wide.columns = [
        f"{metric}_{model}"
        for metric, model in wide.columns
    ]

    wide = wide.reset_index()

    wide["rmse_B_minus_A"] = (
        wide["rmse_B_movement"]
        -
        wide["rmse_A_conventional"]
    )

    wide["rmse_C_minus_A"] = (
        wide["rmse_C_combined"]
        -
        wide["rmse_A_conventional"]
    )

    wide["mae_B_minus_A"] = (
        wide["mae_B_movement"]
        -
        wide["mae_A_conventional"]
    )

    wide["mae_C_minus_A"] = (
        wide["mae_C_combined"]
        -
        wide["mae_A_conventional"]
    )

    wide["r2_B_minus_A"] = (
        wide["r2_B_movement"]
        -
        wide["r2_A_conventional"]
    )

    wide["r2_C_minus_A"] = (
        wide["r2_C_combined"]
        -
        wide["r2_A_conventional"]
    )


    comparison = pd.DataFrame(
        [
            {
                "population": population_name,
                "minimum_target_rows": (
                    minimum_target_rows
                ),
                "comparison": (
                    "B_movement_vs_A_conventional"
                ),
                "rmse_difference_mean": (
                    wide["rmse_B_minus_A"].mean()
                ),
                "rmse_difference_std": (
                    wide["rmse_B_minus_A"].std()
                ),
                "mae_difference_mean": (
                    wide["mae_B_minus_A"].mean()
                ),
                "r2_difference_mean": (
                    wide["r2_B_minus_A"].mean()
                ),
                "folds_rmse_improved": (
                    wide["rmse_B_minus_A"] < 0
                ).sum(),
                "folds_rmse_worse": (
                    wide["rmse_B_minus_A"] > 0
                ).sum()
            },
            {
                "population": population_name,
                "minimum_target_rows": (
                    minimum_target_rows
                ),
                "comparison": (
                    "C_combined_vs_A_conventional"
                ),
                "rmse_difference_mean": (
                    wide["rmse_C_minus_A"].mean()
                ),
                "rmse_difference_std": (
                    wide["rmse_C_minus_A"].std()
                ),
                "mae_difference_mean": (
                    wide["mae_C_minus_A"].mean()
                ),
                "r2_difference_mean": (
                    wide["r2_C_minus_A"].mean()
                ),
                "folds_rmse_improved": (
                    wide["rmse_C_minus_A"] < 0
                ).sum(),
                "folds_rmse_worse": (
                    wide["rmse_C_minus_A"] > 0
                ).sum()
            }
        ]
    )

    return results, summary, comparison


# ============================================================
# RUN MEAN-SEPARATION ROBUSTNESS TESTS
# ============================================================

all_results = []
all_summaries = []
all_comparisons = []

for minimum_rows in MIN_TARGET_ROWS:

    results, summary, comparison = run_experiment(
        base,
        TARGET,
        f"mean_separation_min_{minimum_rows}",
        minimum_rows
    )

    if results is not None:

        all_results.append(results)
        all_summaries.append(summary)
        all_comparisons.append(comparison)


# ============================================================
# SAVE RESULTS
# ============================================================

if all_results:

    fold_results = pd.concat(
        all_results,
        ignore_index=True
    )

    summary_results = pd.concat(
        all_summaries,
        ignore_index=True
    )

    comparison_results = pd.concat(
        all_comparisons,
        ignore_index=True
    )

    fold_results.to_csv(
        OUT / "mean_separation_robustness_folds.csv",
        index=False
    )

    summary_results.to_csv(
        OUT / "mean_separation_robustness_summary.csv",
        index=False
    )

    comparison_results.to_csv(
        OUT / "mean_separation_robustness_comparisons.csv",
        index=False
    )


# ============================================================
# MEDIAN SEPARATION IF AVAILABLE
# ============================================================

if median_column is not None:

    print()
    print("=" * 75)
    print("MEDIAN SEPARATION ANALYSIS")
    print("=" * 75)

    median_results, median_summary, median_comparison = (
        run_experiment(
            base,
            median_column,
            "median_separation",
            0
        )
    )

    if median_results is not None:

        median_results.to_csv(
            OUT / "median_separation_folds.csv",
            index=False
        )

        median_summary.to_csv(
            OUT / "median_separation_summary.csv",
            index=False
        )

        median_comparison.to_csv(
            OUT / "median_separation_comparisons.csv",
            index=False
        )


# ============================================================
# PLAYER COUNT TABLE
# ============================================================

count_rows = []

for minimum_rows in MIN_TARGET_ROWS:

    eligible = base[
        base[TARGET].notna()
        &
        (
            base["separation_observed_rows"]
            >= minimum_rows
        )
    ]

    count_rows.append(
        {
            "minimum_target_rows": minimum_rows,
            "eligible_players": len(eligible),
            "draft_2023": (
                eligible["draft_year"]
                .eq(2023)
                .sum()
            ),
            "draft_2024": (
                eligible["draft_year"]
                .eq(2024)
                .sum()
            ),
            "draft_2025": (
                eligible["draft_year"]
                .eq(2025)
                .sum()
            )
        }
    )

count_table = pd.DataFrame(
    count_rows
)

count_table.to_csv(
    OUT / "reliability_population_counts.csv",
    index=False
)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 75)
print("ROBUSTNESS SUMMARY")
print("=" * 75)

if all_summaries:

    summary_results = pd.concat(
        all_summaries,
        ignore_index=True
    )

    display_columns = [
        "model",
        "n_players",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "r2_mean",
        "r2_std"
    ]

    print(
        summary_results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


print()
print("Population sizes:")
print(
    count_table.to_string(
        index=False
    )
)

if all_comparisons:

    comparison_results = pd.concat(
        all_comparisons,
        ignore_index=True
    )

    print()
    print("Movement/combined vs conventional:")
    print(
        comparison_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


print()
print("=" * 75)
print("DONE")
print("=" * 75)

print()
print(
    "Results saved to:"
)

print(OUT)
