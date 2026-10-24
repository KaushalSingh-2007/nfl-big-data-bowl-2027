from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT_ROOT
    / "research_outputs"
    / "separation_reliability_robustness"
    / "separation_reliability_player_table.csv"
)

# Fallback to the actual location if needed.
if not INPUT.exists():

    INPUT = (
        PROJECT_ROOT
        / "research_outputs"
        / "model_comparison"
        / "separation_reliability_robustness"
        / "separation_reliability_player_table.csv"
    )

OUT = (
    PROJECT_ROOT
    / "research_outputs"
    / "model_comparison"
    / "separation_reliability_robustness"
    / "bootstrap"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
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

# These are the two reliability thresholds we specifically
# want to test.
MINIMUM_ROWS = [20, 30]

# Same locked CV setup as the previous analysis.
N_SPLITS = 5
N_REPEATS = 10
CV_RANDOM_STATE = 42

# Bootstrap settings.
N_BOOTSTRAPS = 10000
BOOTSTRAP_RANDOM_STATE = 20261017

CONFIDENCE_LEVEL = 0.95


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 78)
print("BOOTSTRAP SEPARATION ROBUSTNESS ANALYSIS")
print("=" * 78)

print()
print("Loading reliability table:")
print(INPUT)

df = pd.read_csv(INPUT)

print()
print(f"Dataset shape: {df.shape}")


required = [
    "nfl_id",
    "draft_year",
    TARGET,
    "observed_separation_rows",
    "forty",
    "vertical",
    "speed_cv_DAGGER_ROUTE_LEFT"
]

missing = [
    c
    for c in required
    if c not in df.columns
]

if missing:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
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
# GENERATE EXACT SAME REPEATED-CV PREDICTIONS
# ============================================================

def generate_oof_predictions(
    data,
    minimum_rows
):

    eligible = data[
        data[TARGET].notna()
        &
        data["observed_separation_rows"].notna()
        &
        (
            data["observed_separation_rows"]
            >= minimum_rows
        )
    ].copy()

    eligible = eligible[
        eligible[
            [
                "forty",
                "vertical",
                "speed_cv_DAGGER_ROUTE_LEFT"
            ]
        ].notna().all(axis=1)
    ].copy()

    eligible = eligible.reset_index(drop=True)

    print()
    print("-" * 78)
    print(
        f"Reliability threshold: >= {minimum_rows} "
        "observed separation rows"
    )

    print(
        f"Eligible players: {len(eligible)}"
    )

    print(
        "Draft-year distribution:"
    )

    print(
        eligible["draft_year"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=CV_RANDOM_STATE
    )

    prediction_rows = []

    for split_id, (
        train_idx,
        test_idx
    ) in enumerate(
        cv.split(eligible),
        start=1
    ):

        train = eligible.iloc[train_idx]
        test = eligible.iloc[test_idx]

        predictions = {}

        for model_name, feature_list in FEATURES.items():

            model = make_model()

            model.fit(
                train[feature_list],
                train[TARGET]
            )

            predictions[model_name] = model.predict(
                test[feature_list]
            )

        for row_position, row_index in enumerate(test.index):

            prediction_rows.append(
                {
                    "split_id": split_id,
                    "row_index": row_index,
                    "nfl_id": test.loc[
                        row_index,
                        "nfl_id"
                    ],
                    "draft_year": test.loc[
                        row_index,
                        "draft_year"
                    ],
                    "actual": test.loc[
                        row_index,
                        TARGET
                    ],
                    "prediction_A": predictions[
                        "A_conventional"
                    ][row_position],
                    "prediction_B": predictions[
                        "B_movement"
                    ][row_position],
                    "prediction_C": predictions[
                        "C_combined"
                    ][row_position]
                }
            )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    print()
    print(
        f"Generated {len(predictions_df):,} "
        "out-of-fold prediction rows."
    )

    print(
        f"Unique players represented: "
        f"{predictions_df['nfl_id'].nunique()}"
    )

    return predictions_df


# ============================================================
# RMSE / MAE DIFFERENCES
# ============================================================

def calculate_metrics(y_true, prediction):

    errors = prediction - y_true

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    mae = np.mean(
        np.abs(errors)
    )

    return rmse, mae


def paired_differences(
    predictions
):

    y = predictions["actual"].to_numpy()

    pred_A = predictions[
        "prediction_A"
    ].to_numpy()

    pred_B = predictions[
        "prediction_B"
    ].to_numpy()

    pred_C = predictions[
        "prediction_C"
    ].to_numpy()

    rmse_A, mae_A = calculate_metrics(
        y,
        pred_A
    )

    rmse_B, mae_B = calculate_metrics(
        y,
        pred_B
    )

    rmse_C, mae_C = calculate_metrics(
        y,
        pred_C
    )

    return {
        "rmse_A": rmse_A,
        "rmse_B": rmse_B,
        "rmse_C": rmse_C,
        "mae_A": mae_A,
        "mae_B": mae_B,
        "mae_C": mae_C,

        "rmse_B_minus_A":
            rmse_B - rmse_A,

        "rmse_C_minus_A":
            rmse_C - rmse_A,

        "mae_B_minus_A":
            mae_B - mae_A,

        "mae_C_minus_A":
            mae_C - mae_A
    }


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_comparison(
    predictions,
    minimum_rows
):

    rng = np.random.default_rng(
        BOOTSTRAP_RANDOM_STATE
        + minimum_rows
    )

    n = len(predictions)

    actual = predictions[
        "actual"
    ].to_numpy()

    pred_A = predictions[
        "prediction_A"
    ].to_numpy()

    pred_B = predictions[
        "prediction_B"
    ].to_numpy()

    pred_C = predictions[
        "prediction_C"
    ].to_numpy()

    # Store paired differences.
    bootstrap_results = np.empty(
        (
            N_BOOTSTRAPS,
            4
        )
    )

    print()
    print(
        f"Running {N_BOOTSTRAPS:,} bootstrap "
        f"replicates..."
    )

    for b in range(
        N_BOOTSTRAPS
    ):

        # Resample players/OOF observations with replacement.
        indices = rng.integers(
            0,
            n,
            size=n
        )

        y = actual[
            indices
        ]

        a = pred_A[
            indices
        ]

        m = pred_B[
            indices
        ]

        c = pred_C[
            indices
        ]

        rmse_A, mae_A = calculate_metrics(
            y,
            a
        )

        rmse_B, mae_B = calculate_metrics(
            y,
            m
        )

        rmse_C, mae_C = calculate_metrics(
            y,
            c
        )

        bootstrap_results[
            b,
            0
        ] = rmse_B - rmse_A

        bootstrap_results[
            b,
            1
        ] = mae_B - mae_A

        bootstrap_results[
            b,
            2
        ] = rmse_C - rmse_A

        bootstrap_results[
            b,
            3
        ] = mae_C - mae_A


    alpha = (
        1.0
        -
        CONFIDENCE_LEVEL
    )

    lower = (
        alpha / 2
    )

    upper = (
        1.0
        -
        alpha / 2
    )

    rows = []

    labels = [
        (
            "B_movement_vs_A_conventional",
            0,
            "RMSE"
        ),
        (
            "B_movement_vs_A_conventional",
            1,
            "MAE"
        ),
        (
            "C_combined_vs_A_conventional",
            2,
            "RMSE"
        ),
        (
            "C_combined_vs_A_conventional",
            3,
            "MAE"
        )
    ]

    for comparison, column, metric in labels:

        values = bootstrap_results[
            :,
            column
        ]

        point_estimate = np.mean(
            values
        )

        ci_low = np.quantile(
            values,
            lower
        )

        ci_high = np.quantile(
            values,
            upper
        )

        # Negative difference = improvement.
        probability_improvement = np.mean(
            values < 0
        )

        rows.append(
            {
                "minimum_observed_rows":
                    minimum_rows,

                "comparison":
                    comparison,

                "metric":
                    metric,

                "point_difference":
                    point_estimate,

                "ci_95_lower":
                    ci_low,

                "ci_95_upper":
                    ci_high,

                "probability_model_improves":
                    probability_improvement,

                "bootstrap_replicates":
                    N_BOOTSTRAPS,

                "n_oof_predictions":
                    n
            }
        )

    return pd.DataFrame(rows), bootstrap_results


# ============================================================
# RUN
# ============================================================

all_summary = []
all_bootstrap = []
all_predictions = []


for minimum_rows in MINIMUM_ROWS:

    predictions = generate_oof_predictions(
        df,
        minimum_rows
    )

    predictions[
        "minimum_observed_rows"
    ] = minimum_rows

    all_predictions.append(
        predictions
    )

    # Point estimates from the exact OOF predictions.
    point = paired_differences(
        predictions
    )

    print()
    print(
        "Point estimates:"
    )

    print(
        f"Movement vs conventional RMSE: "
        f"{point['rmse_B_minus_A']:+.6f}"
    )

    print(
        f"Movement vs conventional MAE:  "
        f"{point['mae_B_minus_A']:+.6f}"
    )

    print(
        f"Combined vs conventional RMSE: "
        f"{point['rmse_C_minus_A']:+.6f}"
    )

    print(
        f"Combined vs conventional MAE:  "
        f"{point['mae_C_minus_A']:+.6f}"
    )

    summary, bootstrap_values = bootstrap_comparison(
        predictions,
        minimum_rows
    )

    all_summary.append(
        summary
    )

    # Save raw bootstrap distribution.
    bootstrap_df = pd.DataFrame(
        bootstrap_values,
        columns=[
            "movement_rmse_difference",
            "movement_mae_difference",
            "combined_rmse_difference",
            "combined_mae_difference"
        ]
    )

    bootstrap_df[
        "minimum_observed_rows"
    ] = minimum_rows

    all_bootstrap.append(
        bootstrap_df
    )


# ============================================================
# SAVE
# ============================================================

summary_df = pd.concat(
    all_summary,
    ignore_index=True
)

bootstrap_df = pd.concat(
    all_bootstrap,
    ignore_index=True
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

summary_path = (
    OUT
    / "bootstrap_comparison_summary.csv"
)

bootstrap_path = (
    OUT
    / "bootstrap_distributions.csv"
)

predictions_path = (
    OUT
    / "bootstrap_oof_predictions.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

bootstrap_df.to_csv(
    bootstrap_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# CONSOLE RESULTS
# ============================================================

print()
print("=" * 78)
print("BOOTSTRAP RESULTS")
print("=" * 78)

print()

display_columns = [
    "minimum_observed_rows",
    "comparison",
    "metric",
    "point_difference",
    "ci_95_lower",
    "ci_95_upper",
    "probability_model_improves"
]

print(
    summary_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# INTERPRETATION FLAGS
# ============================================================

print()
print("=" * 78)
print("INTERPRETATION CHECK")
print("=" * 78)

for _, row in summary_df.iterrows():

    comparison = row["comparison"]
    metric = row["metric"]
    threshold = row[
        "minimum_observed_rows"
    ]

    low = row[
        "ci_95_lower"
    ]

    high = row[
        "ci_95_upper"
    ]

    probability = row[
        "probability_model_improves"
    ]

    print()
    print(
        f">= {threshold} rows | "
        f"{comparison} | {metric}"
    )

    print(
        f"95% CI: "
        f"[{low:.6f}, {high:.6f}]"
    )

    print(
        f"P(model improves): "
        f"{probability:.4f}"
    )

    if high < 0:

        print(
            "RESULT: CI is entirely below zero "
            "-> statistically strong evidence of improvement."
        )

    elif low > 0:

        print(
            "RESULT: CI is entirely above zero "
            "-> evidence that the added model is worse."
        )

    else:

        print(
            "RESULT: CI crosses zero "
            "-> improvement is NOT statistically "
            "distinguishable from no improvement."
        )


print()
print("=" * 78)
print("DONE")
print("=" * 78)

print()
print("Summary:")
print(summary_path)

print()
print("Bootstrap distributions:")
print(bootstrap_path)

print()
print("OOF predictions:")
print(predictions_path)
