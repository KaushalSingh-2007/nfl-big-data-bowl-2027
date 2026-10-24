import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_predict
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

# ---------------------------------------------------------------------
# Load research + exact mechanism data.
# ---------------------------------------------------------------------
research = pd.read_csv(RESEARCH)
mechanism = pd.read_csv(MECHANISM)

print()
print("Research rows:", len(research))
print("Research unique players:", research["nfl_id"].nunique())

print("Mechanism rows:", len(mechanism))
print("Mechanism unique players:", mechanism["nfl_id"].nunique())

# Check duplicate player IDs.
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

if len(research_dupes) > 0:
    print(research_dupes)

if len(mechanism_dupes) > 0:
    print(mechanism_dupes)

# ---------------------------------------------------------------------
# Reconstruct receiving efficiency exactly.
# ---------------------------------------------------------------------
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

print(
    "Target count distribution:"
)

print(
    outcome["observed_targets"]
    .describe()
    .round(3)
    .to_string()
)

# ---------------------------------------------------------------------
# Merge at PLAYER level only.
# ---------------------------------------------------------------------
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

assert len(df) == df["nfl_id"].nunique(), (
    "ERROR: final analysis table contains duplicate players."
)

# ---------------------------------------------------------------------
# Locked population.
#
# Exact same three model features:
# A = 40 yd + vertical
# B = Dagger speed CV
# C = 40 yd + vertical + Dagger speed CV
# ---------------------------------------------------------------------
df["speed_cv"] = df["speed_cv"]

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

# ---------------------------------------------------------------------
# Reliability thresholds.
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Exact repeated CV comparison.
# ---------------------------------------------------------------------
print()
print("=" * 75)
print("LOCKED MODEL COMPARISON")
print("=" * 75)

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

for threshold, subset in populations.items():

    if threshold not in [0, 10, 20]:
        continue

    X = subset
    y = subset["receiving_efficiency"].to_numpy()

    predictions = {}

    for model_name, features in models.items():

        estimator = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0)
        )

        pred = cross_val_predict(
            estimator,
            X[features],
            y,
            cv=cv,
            method="predict",
        )

        predictions[model_name] = pred

        rmse = np.sqrt(
            mean_squared_error(
                y,
                pred
            )
        )

        mae = mean_absolute_error(
            y,
            pred
        )

        r2 = r2_score(
            y,
            pred
        )

        all_results.append({
            "threshold": threshold,
            "model": model_name,
            "n_players": len(subset),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        })

    # Paired OOF differences.
    for comparison in [
        ("movement", "conventional"),
        ("combined", "conventional"),
    ]:

        new_model, base_model = comparison

        err_new = np.abs(
            y - predictions[new_model]
        )

        err_base = np.abs(
            y - predictions[base_model]
        )

        squared_new = (
            y - predictions[new_model]
        ) ** 2

        squared_base = (
            y - predictions[base_model]
        ) ** 2

        rmse_diff = (
            np.sqrt(np.mean(squared_new))
            - np.sqrt(np.mean(squared_base))
        )

        mae_diff = (
            np.mean(err_new)
            - np.mean(err_base)
        )

        paired_results.append({
            "threshold": threshold,
            "comparison": f"{new_model}_vs_{base_model}",
            "n_players": len(subset),
            "rmse_difference": rmse_diff,
            "mae_difference": mae_diff,
            "rmse_percent_change": (
                100 * rmse_diff
                / np.sqrt(np.mean(squared_base))
            ),
            "mae_percent_change": (
                100 * mae_diff
                / np.mean(err_base)
            ),
        })

results_df = pd.DataFrame(all_results)
paired_df = pd.DataFrame(paired_results)

print()
print(
    results_df.round(5).to_string(index=False)
)

print()
print("=" * 75)
print("PAIRED DIFFERENCES")
print("=" * 75)

print(
    paired_df.round(5).to_string(index=False)
)

# ---------------------------------------------------------------------
# Save clean outputs.
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Summary table.
# ---------------------------------------------------------------------
summary_rows = []

for threshold in [0, 10, 20]:

    sub = paired_df[
        paired_df["threshold"] == threshold
    ]

    for _, row in sub.iterrows():

        summary_rows.append({
            "target_threshold": threshold,
            "comparison": row["comparison"],
            "n": row["n_players"],
            "rmse_change": row["rmse_difference"],
            "rmse_percent_change": row[
                "rmse_percent_change"
            ],
            "mae_change": row["mae_difference"],
            "mae_percent_change": row[
                "mae_percent_change"
            ],
        })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    os.path.join(
        OUT,
        "final_receiving_efficiency_summary.csv"
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
