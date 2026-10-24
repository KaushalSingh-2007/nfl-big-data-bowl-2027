from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_outputs"

DATASET = OUT / "wr_research_dataset.csv"
OUTCOME = (
    OUT
    / "receiving_efficiency_final_sanity"
    / "receiving_efficiency_outcome_reconstructed.csv"
)

SAVE_DIR = OUT / "nonlinear_final"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load
# ============================================================

if not DATASET.exists():
    raise FileNotFoundError(DATASET)

if not OUTCOME.exists():
    raise FileNotFoundError(OUTCOME)

dataset = pd.read_csv(DATASET)
outcome = pd.read_csv(OUTCOME)

print("=" * 70)
print("FINAL NONLINEAR VALIDATION")
print("=" * 70)

print(f"Research dataset: {dataset.shape}")
print(f"Outcome dataset: {outcome.shape}")


# ============================================================
# Reconstruct final modelling population
# ============================================================

feature_cols = [
    "forty",
    "vertical",
    "speed_cv_DAGGER_ROUTE_LEFT",
]

for col in feature_cols:
    if col not in dataset.columns:
        raise ValueError(
            f"Missing feature in research dataset: {col}"
        )

if "nfl_id" not in dataset.columns:
    raise ValueError("nfl_id missing from research dataset")

if "nfl_id" not in outcome.columns:
    raise ValueError("nfl_id missing from outcome dataset")

if "receiving_efficiency" not in outcome.columns:
    raise ValueError(
        "receiving_efficiency missing from outcome dataset"
    )

outcome_small = outcome[
    ["nfl_id", "receiving_efficiency"]
].drop_duplicates(
    subset=["nfl_id"]
)

data = dataset[
    ["nfl_id"] + feature_cols
].merge(
    outcome_small,
    on="nfl_id",
    how="inner",
)

data = data.dropna(
    subset=feature_cols + ["receiving_efficiency"]
).copy()

data = data.drop_duplicates(
    subset=["nfl_id"]
)

print(f"\nMerged complete population: {len(data)}")
print(f"Unique players: {data['nfl_id'].nunique()}")


# ============================================================
# Target reliability
# ============================================================

# Target counts are reconstructed from the final locked
# fold population file because that file defines the exact
# reliability populations used in the final analysis.

FOLD_FILE = (
    OUT
    / "receiving_efficiency_final_sanity"
    / "final_locked_fold_predictions.csv"
)

if not FOLD_FILE.exists():
    raise FileNotFoundError(FOLD_FILE)

fold_data = pd.read_csv(FOLD_FILE)

threshold_ids = {}

for threshold in [0, 10, 20]:

    ids = set(
        fold_data[
            fold_data["threshold"] == threshold
        ]["nfl_id"].unique()
    )

    threshold_ids[threshold] = ids

    print(
        f"Threshold >= {threshold}: "
        f"{len(ids)} players"
    )


# ============================================================
# Models
# ============================================================

# Linear models reproduce the locked analysis.
#
# Polynomial models add only degree-2 terms:
#
# x1, x2, x3,
# x1^2, x2^2, x3^2,
# x1*x2, x1*x3, x2*x3
#
# No hyperparameter search is performed.

def make_linear():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def make_polynomial():
    return Pipeline(
        [
            (
                "poly",
                PolynomialFeatures(
                    degree=2,
                    include_bias=False,
                ),
            ),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


model_specs = {
    "linear_conventional": (
        ["forty", "vertical"],
        make_linear,
    ),

    "linear_movement": (
        ["speed_cv_DAGGER_ROUTE_LEFT"],
        make_linear,
    ),

    "linear_combined": (
        [
            "forty",
            "vertical",
            "speed_cv_DAGGER_ROUTE_LEFT",
        ],
        make_linear,
    ),

    "poly_conventional": (
        ["forty", "vertical"],
        make_polynomial,
    ),

    "poly_movement": (
        ["speed_cv_DAGGER_ROUTE_LEFT"],
        make_polynomial,
    ),

    "poly_combined": (
        [
            "forty",
            "vertical",
            "speed_cv_DAGGER_ROUTE_LEFT",
        ],
        make_polynomial,
    ),
}


# ============================================================
# Repeated 5 x 10 CV
# ============================================================

def run_repeated_cv(df, feature_list, model_factory):

    X = df[feature_list].to_numpy()
    y = df["receiving_efficiency"].to_numpy()

    n = len(df)

    if n < 5:
        raise ValueError(
            f"Not enough players for 5-fold CV: {n}"
        )

    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42,
    )

    all_true = []
    all_pred = []
    fold_rows = []

    split_number = 0

    for train_idx, test_idx in rkf.split(X):

        split_number += 1

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = model_factory()

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(X_test)

        all_true.extend(y_test)
        all_pred.extend(pred)

        fold_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred,
            )
        )

        fold_mae = mean_absolute_error(
            y_test,
            pred,
        )

        fold_r2 = r2_score(
            y_test,
            pred,
        )

        fold_rows.append(
            {
                "split": split_number,
                "rmse": fold_rmse,
                "mae": fold_mae,
                "r2": fold_r2,
                "n_test": len(test_idx),
            }
        )

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)

    aggregate_rmse = np.sqrt(
        mean_squared_error(
            all_true,
            all_pred,
        )
    )

    aggregate_mae = mean_absolute_error(
        all_true,
        all_pred,
    )

    aggregate_r2 = r2_score(
        all_true,
        all_pred,
    )

    fold_df = pd.DataFrame(
        fold_rows
    )

    return {
        "rmse": aggregate_rmse,
        "mae": aggregate_mae,
        "r2": aggregate_r2,
        "fold_rmse_mean": fold_df["rmse"].mean(),
        "fold_rmse_std": fold_df["rmse"].std(ddof=1),
        "fold_mae_mean": fold_df["mae"].mean(),
        "fold_mae_std": fold_df["mae"].std(ddof=1),
        "fold_r2_mean": fold_df["r2"].mean(),
        "fold_r2_std": fold_df["r2"].std(ddof=1),
        "folds": fold_df,
        "y_true": all_true,
        "pred": all_pred,
    }


# ============================================================
# Run
# ============================================================

all_results = []
all_predictions = []

for threshold in [0, 10, 20]:

    ids = threshold_ids[threshold]

    subset = data[
        data["nfl_id"].isin(ids)
    ].copy()

    subset = subset.sort_values(
        "nfl_id"
    ).reset_index(drop=True)

    print("\n" + "-" * 70)
    print(
        f"THRESHOLD >= {threshold} "
        f"(N={len(subset)})"
    )
    print("-" * 70)

    for model_name, (
        features,
        factory,
    ) in model_specs.items():

        result = run_repeated_cv(
            subset,
            features,
            factory,
        )

        all_results.append(
            {
                "threshold": threshold,
                "n": len(subset),
                "model": model_name,
                "rmse": result["rmse"],
                "mae": result["mae"],
                "r2": result["r2"],
                "fold_rmse_mean": result["fold_rmse_mean"],
                "fold_rmse_std": result["fold_rmse_std"],
                "fold_mae_mean": result["fold_mae_mean"],
                "fold_mae_std": result["fold_mae_std"],
                "fold_r2_mean": result["fold_r2_mean"],
                "fold_r2_std": result["fold_r2_std"],
            }
        )

        print(
            f"{model_name:22s} "
            f"RMSE={result['rmse']:.5f}  "
            f"MAE={result['mae']:.5f}  "
            f"R2={result['r2']:.5f}"
        )

        # Save aggregate repeated-CV predictions.
        # Each player appears once per held-out split.
        rkf = RepeatedKFold(
            n_splits=5,
            n_repeats=10,
            random_state=42,
        )

        X = subset[features].to_numpy()
        y = subset["receiving_efficiency"].to_numpy()

        split_number = 0

        for train_idx, test_idx in rkf.split(X):

            split_number += 1

            model = factory()

            model.fit(
                X[train_idx],
                y[train_idx],
            )

            pred = model.predict(
                X[test_idx]
            )

            for local_idx, prediction in zip(
                test_idx,
                pred,
            ):

                all_predictions.append(
                    {
                        "threshold": threshold,
                        "model": model_name,
                        "split": split_number,
                        "nfl_id": int(
                            subset.iloc[
                                local_idx
                            ]["nfl_id"]
                        ),
                        "y_true": float(
                            y[local_idx]
                        ),
                        "prediction": float(
                            prediction
                        ),
                    }
                )


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.DataFrame(
    all_predictions
)

results_path = (
    SAVE_DIR /
    "final_nonlinear_model_results.csv"
)

predictions_path = (
    SAVE_DIR /
    "final_nonlinear_predictions.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)


# ============================================================
# Calculate nonlinear incremental comparisons
# ============================================================

summary_rows = []

for threshold in [0, 10, 20]:

    sub = results_df[
        results_df["threshold"] == threshold
    ].set_index("model")

    linear_conventional = sub.loc[
        "linear_conventional"
    ]

    linear_movement = sub.loc[
        "linear_movement"
    ]

    linear_combined = sub.loc[
        "linear_combined"
    ]

    poly_conventional = sub.loc[
        "poly_conventional"
    ]

    poly_movement = sub.loc[
        "poly_movement"
    ]

    poly_combined = sub.loc[
        "poly_combined"
    ]

    summary_rows.append(
        {
            "threshold": threshold,

            "linear_movement_vs_linear_conventional_rmse_pct":
                100
                * (
                    linear_conventional["rmse"]
                    - linear_movement["rmse"]
                )
                / linear_conventional["rmse"],

            "linear_combined_vs_linear_conventional_rmse_pct":
                100
                * (
                    linear_conventional["rmse"]
                    - linear_combined["rmse"]
                )
                / linear_conventional["rmse"],

            "poly_movement_vs_poly_conventional_rmse_pct":
                100
                * (
                    poly_conventional["rmse"]
                    - poly_movement["rmse"]
                )
                / poly_conventional["rmse"],

            "poly_combined_vs_poly_conventional_rmse_pct":
                100
                * (
                    poly_conventional["rmse"]
                    - poly_combined["rmse"]
                )
                / poly_conventional["rmse"],

            "nonlinear_combined_vs_linear_combined_rmse_pct":
                100
                * (
                    linear_combined["rmse"]
                    - poly_combined["rmse"]
                )
                / linear_combined["rmse"],

            "nonlinear_combined_vs_linear_combined_mae_pct":
                100
                * (
                    linear_combined["mae"]
                    - poly_combined["mae"]
                )
                / linear_combined["mae"],
        }
    )

summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    SAVE_DIR /
    "nonlinear_incremental_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Print final interpretation
# ============================================================

print("\n" + "=" * 70)
print("NONLINEAR INCREMENTAL RESULTS")
print("=" * 70)

for _, row in summary_df.iterrows():

    threshold = int(
        row["threshold"]
    )

    print(
        f"\nThreshold >= {threshold}"
    )

    print(
        "Linear movement vs conventional: "
        f"{row['linear_movement_vs_linear_conventional_rmse_pct']:+.2f}% RMSE"
    )

    print(
        "Polynomial movement vs polynomial conventional: "
        f"{row['poly_movement_vs_poly_conventional_rmse_pct']:+.2f}% RMSE"
    )

    print(
        "Polynomial combined vs linear combined: "
        f"{row['nonlinear_combined_vs_linear_combined_rmse_pct']:+.2f}% RMSE"
    )

    print(
        "Polynomial combined vs linear combined: "
        f"{row['nonlinear_combined_vs_linear_combined_mae_pct']:+.2f}% MAE"
    )


# ============================================================
# Final ≥20 conclusion check
# ============================================================

row20 = summary_df[
    summary_df["threshold"] == 20
].iloc[0]

results20 = results_df[
    results_df["threshold"] == 20
].set_index("model")

linear_combined_rmse = results20.loc[
    "linear_combined",
    "rmse",
]

poly_combined_rmse = results20.loc[
    "poly_combined",
    "rmse",
]

linear_movement_rmse = results20.loc[
    "linear_movement",
    "rmse",
]

poly_movement_rmse = results20.loc[
    "poly_movement",
    "rmse",
]

print("\n" + "=" * 70)
print("KEY ≥20-TARGET RESULT")
print("=" * 70)

print(
    f"Linear conventional RMSE:  "
    f"{results20.loc['linear_conventional', 'rmse']:.5f}"
)

print(
    f"Linear movement RMSE:      "
    f"{linear_movement_rmse:.5f}"
)

print(
    f"Linear combined RMSE:      "
    f"{linear_combined_rmse:.5f}"
)

print(
    f"Polynomial conventional:   "
    f"{results20.loc['poly_conventional', 'rmse']:.5f}"
)

print(
    f"Polynomial movement:       "
    f"{poly_movement_rmse:.5f}"
)

print(
    f"Polynomial combined:       "
    f"{poly_combined_rmse:.5f}"
)

print(
    "\nNonlinear combined vs linear combined:"
    f" {row20['nonlinear_combined_vs_linear_combined_rmse_pct']:+.2f}% RMSE"
)

if (
    poly_combined_rmse
    < linear_combined_rmse
):
    print(
        "\nRESULT: The controlled nonlinear model "
        "improves on the linear combined model."
    )
else:
    print(
        "\nRESULT: The controlled nonlinear model "
        "does NOT improve on the linear combined model."
    )

print("\nFiles saved:")
print(results_path)
print(predictions_path)
print(summary_path)

print("\nDone.")
