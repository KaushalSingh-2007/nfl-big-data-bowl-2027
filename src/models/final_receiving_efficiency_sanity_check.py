import os
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RESEARCH = os.path.join(
    ROOT,
    "research_outputs",
    "wr_research_dataset.csv"
)

MECHANISM = os.path.join(
    ROOT,
    "research_outputs",
    "speed_cv_mechanism_exact",
    "exact_mechanism_player_features.csv"
)

PLAYER_PLAY = os.path.join(
    ROOT,
    "data",
    "player_play.csv"
)

PLAYERS = os.path.join(
    ROOT,
    "data",
    "players.csv"
)

GAMES = os.path.join(
    ROOT,
    "data",
    "games.csv"
)

OUT = os.path.join(
    ROOT,
    "research_outputs",
    "receiving_efficiency_final_sanity"
)

os.makedirs(OUT, exist_ok=True)

print("=" * 75)
print("FINAL RECEIVING-EFFICIENCY SANITY CHECK")
print("=" * 75)


# =====================================================================
# LOAD DATA
# =====================================================================

research = pd.read_csv(RESEARCH)
mechanism = pd.read_csv(MECHANISM)

print()
print("Research rows:", len(research))
print("Research unique players:", research["nfl_id"].nunique())

print("Mechanism rows:", len(mechanism))
print("Mechanism unique players:", mechanism["nfl_id"].nunique())

research_dupes = (
    research["nfl_id"]
    .value_counts()
    .loc[lambda x: x > 1]
)

mechanism_dupes = (
    mechanism["nfl_id"]
    .value_counts()
    .loc[lambda x: x > 1]
)

print()
print("Duplicate research players:", len(research_dupes))
print("Duplicate mechanism players:", len(mechanism_dupes))

assert len(research_dupes) == 0
assert len(mechanism_dupes) == 0


# =====================================================================
# RECONSTRUCT RECEIVING EFFICIENCY
# =====================================================================

print()
print("Reconstructing receiving efficiency...")

pp = pd.read_csv(
    PLAYER_PLAY,
    usecols=[
        "game_id",
        "play_id",
        "nfl_id",
        "target",
        "rec_yards",
    ]
)

players = pd.read_csv(
    PLAYERS,
    usecols=[
        "nfl_id",
        "nfl_position",
        "draft_year",
    ]
)

games = pd.read_csv(
    GAMES,
    usecols=[
        "game_id",
        "season",
        "season_type",
    ]
)

research_ids = set(
    research["nfl_id"]
    .dropna()
    .astype(int)
)

players["nfl_id"] = pd.to_numeric(
    players["nfl_id"],
    errors="coerce"
)

pp["nfl_id"] = pd.to_numeric(
    pp["nfl_id"],
    errors="coerce"
)

players_wr = players[
    (players["nfl_position"] == "WR") &
    (players["nfl_id"].isin(research_ids))
].copy()

pp = pp[
    pp["nfl_id"].isin(research_ids)
].copy()

pp = pp.merge(
    players_wr[
        ["nfl_id", "draft_year"]
    ],
    on="nfl_id",
    how="inner"
)

games["season"] = pd.to_numeric(
    games["season"],
    errors="coerce"
)

pp = pp.merge(
    games[
        ["game_id", "season", "season_type"]
    ],
    on="game_id",
    how="inner"
)

pp = pp[
    (pp["season_type"] == "REG") &
    (pp["season"] >= pp["draft_year"])
].copy()

print(
    "Post-draft regular-season player-play rows:",
    f"{len(pp):,}"
)

pp["target"] = pd.to_numeric(
    pp["target"],
    errors="coerce"
).fillna(0)

targets = pp[
    pp["target"] == 1
].copy()

targets["rec_yards"] = pd.to_numeric(
    targets["rec_yards"],
    errors="coerce"
).fillna(0)

outcome = (
    targets
    .groupby("nfl_id")
    .agg(
        observed_targets=("target", "size"),
        receiving_yards=("rec_yards", "sum"),
    )
    .reset_index()
)

outcome["receiving_efficiency"] = (
    outcome["receiving_yards"]
    / outcome["observed_targets"]
)

print(
    "Players with observed targets:",
    len(outcome)
)

print()
print("Target count distribution:")
print(
    outcome["observed_targets"]
    .describe()
    .round(3)
    .to_string()
)


# =====================================================================
# PLAYER-LEVEL MERGE
# =====================================================================

df = research[
    [
        "nfl_id",
        "draft_year",
        "forty",
        "vertical",
    ]
].copy()

df = df.merge(
    mechanism,
    on="nfl_id",
    how="inner",
    validate="one_to_one"
)

df = df.merge(
    outcome,
    on="nfl_id",
    how="left",
    validate="one_to_one"
)

print()
print("Final merged rows:", len(df))
print("Final unique players:", df["nfl_id"].nunique())

assert len(df) == df["nfl_id"].nunique()


# =====================================================================
# LOCKED POPULATION
# =====================================================================

locked = df.dropna(
    subset=[
        "forty",
        "vertical",
        "speed_cv",
        "receiving_efficiency",
    ]
).copy()

print()
print("=" * 75)
print("LOCKED POPULATION")
print("=" * 75)

print("N =", len(locked))
print("Unique players =", locked["nfl_id"].nunique())

print()
print("Draft-year distribution:")
print(
    locked["draft_year"]
    .value_counts()
    .sort_index()
    .to_string()
)

assert len(locked) == 64
assert locked["nfl_id"].nunique() == 64


# =====================================================================
# RELIABILITY POPULATIONS
# =====================================================================

print()
print("=" * 75)
print("RELIABILITY POPULATIONS")
print("=" * 75)

populations = {}

for threshold in [0, 5, 10, 20, 30, 50, 100]:

    subset = locked[
        locked["observed_targets"] >= threshold
    ].copy()

    populations[threshold] = subset

    print()
    print(
        f">= {threshold} observed targets: "
        f"N={len(subset)}"
    )

    print(
        subset["draft_year"]
        .value_counts()
        .sort_index()
        .to_dict()
    )


# =====================================================================
# MANUAL REPEATED-KFOLD PREDICTIONS
#
# cross_val_predict() cannot be used with RepeatedKFold because the
# same player appears in test sets multiple times across repeats.
#
# We therefore manually generate one prediction per fold and retain
# every fold prediction.
# =====================================================================

def repeated_cv_predictions(X, y, features, cv):

    X_values = X[features].to_numpy()
    y_values = np.asarray(y)

    fold_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(
        cv.split(X_values),
        start=1
    ):

        estimator = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0)
        )

        estimator.fit(
            X_values[train_idx],
            y_values[train_idx]
        )

        pred = estimator.predict(
            X_values[test_idx]
        )

        for idx, prediction in zip(
            test_idx,
            pred
        ):

            fold_rows.append({
                "fold": fold_id,
                "row_index": int(idx),
                "y_true": float(y_values[idx]),
                "prediction": float(prediction),
            })

    return pd.DataFrame(fold_rows)


def make_pipeline(*args, **kwargs):
    return __import__(
        "sklearn.pipeline",
        fromlist=["make_pipeline"]
    ).make_pipeline(*args, **kwargs)


# =====================================================================
# MODELS
# =====================================================================

models = {
    "conventional": [
        "forty",
        "vertical",
    ],
    "movement": [
        "speed_cv",
    ],
    "combined": [
        "forty",
        "vertical",
        "speed_cv",
    ],
}


cv = RepeatedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42,
)


all_results = []
paired_results = []
fold_predictions = []


# =====================================================================
# RUN MODELS
# =====================================================================

for threshold, subset in populations.items():

    if threshold not in [0, 10, 20]:
        continue

    subset = subset.reset_index(drop=True)

    y = subset[
        "receiving_efficiency"
    ].to_numpy()

    model_predictions = {}

    for model_name, features in models.items():

        pred_df = repeated_cv_predictions(
            subset,
            y,
            features,
            cv
        )

        pred_df["model"] = model_name
        pred_df["threshold"] = threshold

        pred_df["nfl_id"] = (
            subset
            .iloc[
                pred_df["row_index"].to_numpy()
            ]["nfl_id"]
            .to_numpy()
        )

        fold_predictions.append(pred_df)

        # Aggregate all repeated out-of-fold predictions.
        y_true = pred_df["y_true"].to_numpy()
        y_pred = pred_df["prediction"].to_numpy()

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        r2 = r2_score(
            y_true,
            y_pred
        )

        all_results.append({
            "threshold": threshold,
            "model": model_name,
            "n_players": len(subset),
            "n_predictions": len(pred_df),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        })

        model_predictions[
            model_name
        ] = pred_df

    # ---------------------------------------------------------------
    # Pair predictions by exact fold + player.
    # ---------------------------------------------------------------
    movement = model_predictions[
        "movement"
    ].rename(
        columns={
            "prediction": "movement_pred",
            "y_true": "y_true_movement",
        }
    )

    conventional = model_predictions[
        "conventional"
    ].rename(
        columns={
            "prediction": "conventional_pred",
            "y_true": "y_true_conventional",
        }
    )

    combined = model_predictions[
        "combined"
    ].rename(
        columns={
            "prediction": "combined_pred",
            "y_true": "y_true_combined",
        }
    )

    paired = movement[
        [
            "fold",
            "row_index",
            "nfl_id",
            "movement_pred",
        ]
    ].merge(
        conventional[
            [
                "fold",
                "row_index",
                "conventional_pred",
            ]
        ],
        on=[
            "fold",
            "row_index",
        ],
        how="inner",
        validate="one_to_one"
    )

    paired = paired.merge(
        combined[
            [
                "fold",
                "row_index",
                "combined_pred",
            ]
        ],
        on=[
            "fold",
            "row_index",
        ],
        how="inner",
        validate="one_to_one"
    )

    y_true = (
        subset
        .iloc[
            paired["row_index"].to_numpy()
        ]["receiving_efficiency"]
        .to_numpy()
    )

    movement_pred = paired[
        "movement_pred"
    ].to_numpy()

    conventional_pred = paired[
        "conventional_pred"
    ].to_numpy()

    combined_pred = paired[
        "combined_pred"
    ].to_numpy()

    movement_rmse = np.sqrt(
        mean_squared_error(
            y_true,
            movement_pred
        )
    )

    conventional_rmse = np.sqrt(
        mean_squared_error(
            y_true,
            conventional_pred
        )
    )

    combined_rmse = np.sqrt(
        mean_squared_error(
            y_true,
            combined_pred
        )
    )

    movement_mae = mean_absolute_error(
        y_true,
        movement_pred
    )

    conventional_mae = mean_absolute_error(
        y_true,
        conventional_pred
    )

    combined_mae = mean_absolute_error(
        y_true,
        combined_pred
    )

    paired_results.append({
        "threshold": threshold,
        "comparison": "movement_vs_conventional",
        "n_players": len(subset),
        "n_predictions": len(paired),
        "rmse_difference": (
            movement_rmse
            - conventional_rmse
        ),
        "mae_difference": (
            movement_mae
            - conventional_mae
        ),
        "rmse_percent_change": (
            100
            * (
                movement_rmse
                - conventional_rmse
            )
            / conventional_rmse
        ),
        "mae_percent_change": (
            100
            * (
                movement_mae
                - conventional_mae
            )
            / conventional_mae
        ),
    })

    paired_results.append({
        "threshold": threshold,
        "comparison": "combined_vs_conventional",
        "n_players": len(subset),
        "n_predictions": len(paired),
        "rmse_difference": (
            combined_rmse
            - conventional_rmse
        ),
        "mae_difference": (
            combined_mae
            - conventional_mae
        ),
        "rmse_percent_change": (
            100
            * (
                combined_rmse
                - conventional_rmse
            )
            / conventional_rmse
        ),
        "mae_percent_change": (
            100
            * (
                combined_mae
                - conventional_mae
            )
            / conventional_mae
        ),
    })


# =====================================================================
# OUTPUT
# =====================================================================

results_df = pd.DataFrame(
    all_results
)

paired_df = pd.DataFrame(
    paired_results
)

fold_df = pd.concat(
    fold_predictions,
    ignore_index=True
)

print()
print("=" * 75)
print("LOCKED MODEL COMPARISON")
print("=" * 75)

print(
    results_df
    .round(5)
    .to_string(index=False)
)

print()
print("=" * 75)
print("PAIRED DIFFERENCES")
print("=" * 75)

print(
    paired_df
    .round(5)
    .to_string(index=False)
)

# =====================================================================
# Save files
# =====================================================================

locked.to_csv(
    os.path.join(
        OUT,
        "locked_population_64.csv"
    ),
    index=False
)

outcome.to_csv(
    os.path.join(
        OUT,
        "receiving_efficiency_outcome_reconstructed.csv"
    ),
    index=False
)

results_df.to_csv(
    os.path.join(
        OUT,
        "final_locked_model_results.csv"
    ),
    index=False
)

paired_df.to_csv(
    os.path.join(
        OUT,
        "final_locked_paired_differences.csv"
    ),
    index=False
)

fold_df.to_csv(
    os.path.join(
        OUT,
        "final_locked_fold_predictions.csv"
    ),
    index=False
)


print()
print("=" * 75)
print("SANITY CHECK COMPLETE")
print("=" * 75)

print()
print("Outputs saved to:")
print(OUT)
