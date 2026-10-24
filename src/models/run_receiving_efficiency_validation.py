from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[2]

RESEARCH = ROOT / "research_outputs" / "wr_research_dataset.csv"
PLAYER_PLAY = ROOT / "data" / "player_play.csv"
PLAYERS = ROOT / "data" / "players.csv"
GAMES = ROOT / "data" / "games.csv"

OUT = ROOT / "research_outputs" / "receiving_efficiency"
OUT.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("RECEIVING EFFICIENCY VALIDATION")
print("=" * 80)


# ---------------------------------------------------------------------
# 1. Load research population
# ---------------------------------------------------------------------

research = pd.read_csv(RESEARCH)

print(f"\nResearch population: {len(research)} players")

if "nfl_id" not in research.columns:
    raise ValueError("wr_research_dataset.csv does not contain nfl_id")


# ---------------------------------------------------------------------
# 2. Load raw outcome data
# ---------------------------------------------------------------------

print("\nLoading players.csv...")
players = pd.read_csv(
    PLAYERS,
    usecols=[
        "nfl_id",
        "draft_year",
        "nfl_position"
    ]
)

print("Loading games.csv...")
games = pd.read_csv(
    GAMES,
    usecols=[
        "game_id",
        "season",
        "season_type"
    ]
)

print("Loading player_play.csv...")
player_play = pd.read_csv(
    PLAYER_PLAY,
    usecols=[
        "game_id",
        "play_id",
        "nfl_id",
        "target",
        "rec_yards",
        "separation_at_pass_forward"
    ]
)

print(f"Raw player-play rows: {len(player_play):,}")


# ---------------------------------------------------------------------
# 3. Restrict to research population and WRs
# ---------------------------------------------------------------------

research_ids = set(
    research["nfl_id"]
    .dropna()
    .astype(int)
)

players = players[
    players["nfl_id"].isin(research_ids)
].copy()

players["nfl_id"] = players["nfl_id"].astype(int)

players = players[
    players["nfl_position"].eq("WR")
].copy()

print(
    f"Research WR population: "
    f"{players['nfl_id'].nunique()} players"
)


# ---------------------------------------------------------------------
# 4. Attach draft year and game information
# ---------------------------------------------------------------------

player_play["nfl_id"] = pd.to_numeric(
    player_play["nfl_id"],
    errors="coerce"
)

player_play["game_id"] = pd.to_numeric(
    player_play["game_id"],
    errors="coerce"
)

player_play = player_play.dropna(
    subset=["nfl_id", "game_id"]
).copy()

player_play["nfl_id"] = player_play["nfl_id"].astype(int)
player_play["game_id"] = player_play["game_id"].astype(int)

df = player_play.merge(
    players[
        ["nfl_id", "draft_year"]
    ],
    on="nfl_id",
    how="inner",
    validate="many_to_one"
)

df = df.merge(
    games,
    on="game_id",
    how="left",
    validate="many_to_one"
)


# ---------------------------------------------------------------------
# 5. Post-draft regular-season population
# ---------------------------------------------------------------------

df = df[
    (df["season_type"].eq("REG")) &
    (df["season"] >= df["draft_year"])
].copy()

print(
    f"Post-draft regular-season rows: "
    f"{len(df):,}"
)


# ---------------------------------------------------------------------
# 6. Receiving-efficiency outcome
#
# Numerator:
#   observed receiving yards
#
# Denominator:
#   observed targets
#
# We only use target == 1 rows for the efficiency outcome.
# ---------------------------------------------------------------------

df["target"] = pd.to_numeric(
    df["target"],
    errors="coerce"
)

df["rec_yards"] = pd.to_numeric(
    df["rec_yards"],
    errors="coerce"
)

target_rows = df[
    df["target"].eq(1)
].copy()

print(
    f"Post-draft target rows: "
    f"{len(target_rows):,}"
)

target_rows["rec_yards"] = target_rows[
    "rec_yards"
].fillna(0)


# ---------------------------------------------------------------------
# 7. Aggregate per player
# ---------------------------------------------------------------------

outcomes = (
    target_rows
    .groupby("nfl_id")
    .agg(
        observed_targets=(
            "target",
            "count"
        ),
        observed_receiving_yards=(
            "rec_yards",
            "sum"
        ),
        observed_yards_per_observed_target=(
            "rec_yards",
            "mean"
        ),
        observed_separation_count=(
            "separation_at_pass_forward",
            lambda x: x.notna().sum()
        )
    )
    .reset_index()
)

print(
    f"Players with observed targets: "
    f"{len(outcomes)}"
)


# ---------------------------------------------------------------------
# 8. Merge outcome with research features
# ---------------------------------------------------------------------

df_model = research.merge(
    outcomes,
    on="nfl_id",
    how="left",
    validate="one_to_one"
)

target = "observed_yards_per_observed_target"

print("\nTarget summary:")
print(
    df_model[target]
    .describe()
    .to_string()
)

print("\nTarget missingness:")
print(
    df_model[target]
    .isna()
    .value_counts()
    .rename({
        False: "observed",
        True: "missing"
    })
    .to_string()
)


# ---------------------------------------------------------------------
# 9. Locked model evaluation
# ---------------------------------------------------------------------

def evaluate(
    data,
    features,
    target,
    model_name,
    n_repeats=10
):

    d = data[
        features + [target]
    ].dropna().copy()

    X = d[features].to_numpy()
    y = d[target].to_numpy()

    if len(d) < 10:
        raise ValueError(
            f"Too few complete players for {model_name}: {len(d)}"
        )

    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=n_repeats,
        random_state=42
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(
        rkf.split(X),
        start=1
    ):

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge(alpha=1.0)
            )
        ])

        model.fit(
            X[train_idx],
            y[train_idx]
        )

        pred = model.predict(
            X[test_idx]
        )

        rows.append({
            "model": model_name,
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "rmse": np.sqrt(
                mean_squared_error(
                    y[test_idx],
                    pred
                )
            ),
            "mae": mean_absolute_error(
                y[test_idx],
                pred
            ),
            "r2": r2_score(
                y[test_idx],
                pred
            )
        })

    result = pd.DataFrame(rows)

    summary = {
        "model": model_name,
        "n_players": len(d),
        "rmse_mean": result["rmse"].mean(),
        "rmse_std": result["rmse"].std(ddof=1),
        "mae_mean": result["mae"].mean(),
        "mae_std": result["mae"].std(ddof=1),
        "r2_mean": result["r2"].mean(),
        "r2_std": result["r2"].std(ddof=1)
    }

    return result, summary


def run_population(
    data,
    population_name
):

    models = [
        (
            "A_conventional",
            [
                "forty",
                "vertical"
            ]
        ),
        (
            "B_movement",
            [
                "speed_cv_DAGGER_ROUTE_LEFT"
            ]
        ),
        (
            "C_combined",
            [
                "forty",
                "vertical",
                "speed_cv_DAGGER_ROUTE_LEFT"
            ]
        )
    ]

    folds_all = []
    summaries = []

    for model_name, features in models:

        folds, summary = evaluate(
            data,
            features,
            target,
            model_name
        )

        folds["population"] = population_name
        folds_all.append(folds)

        summary["population"] = population_name
        summary["features"] = ", ".join(features)

        summaries.append(summary)

    return (
        pd.concat(
            folds_all,
            ignore_index=True
        ),
        pd.DataFrame(summaries)
    )


# ---------------------------------------------------------------------
# 10. Reliability populations
# ---------------------------------------------------------------------

fold_results = []
summary_results = []


# All complete
folds, summary = run_population(
    df_model,
    "all_complete"
)

fold_results.append(folds)
summary_results.append(summary)


# >=10 observed targets
df_10 = df_model[
    df_model["observed_targets"] >= 10
].copy()

folds, summary = run_population(
    df_10,
    "at_least_10_observed_targets"
)

fold_results.append(folds)
summary_results.append(summary)


# >=20 observed targets
df_20 = df_model[
    df_model["observed_targets"] >= 20
].copy()

folds, summary = run_population(
    df_20,
    "at_least_20_observed_targets"
)

fold_results.append(folds)
summary_results.append(summary)


folds = pd.concat(
    fold_results,
    ignore_index=True
)

summary = pd.concat(
    summary_results,
    ignore_index=True
)


# ---------------------------------------------------------------------
# 11. Save results
# ---------------------------------------------------------------------

df_model.to_csv(
    OUT / "receiving_efficiency_population.csv",
    index=False
)

folds.to_csv(
    OUT / "receiving_efficiency_cv_folds.csv",
    index=False
)

summary.to_csv(
    OUT / "receiving_efficiency_cv_summary.csv",
    index=False
)


# ---------------------------------------------------------------------
# 12. Print results
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("CV SUMMARY")
print("=" * 80)

print(
    summary[
        [
            "population",
            "model",
            "n_players",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "mae_std",
            "r2_mean",
            "r2_std"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 80)
print("MODEL DIFFERENCES VS CONVENTIONAL")
print("=" * 80)

for population in summary["population"].unique():

    s = summary[
        summary["population"] == population
    ].set_index("model")

    base_rmse = s.loc[
        "A_conventional",
        "rmse_mean"
    ]

    base_mae = s.loc[
        "A_conventional",
        "mae_mean"
    ]

    print(
        f"\nPopulation: {population}"
    )

    for model_name in [
        "B_movement",
        "C_combined"
    ]:

        rmse = s.loc[
            model_name,
            "rmse_mean"
        ]

        mae = s.loc[
            model_name,
            "mae_mean"
        ]

        rmse_change = rmse - base_rmse
        mae_change = mae - base_mae

        rmse_improvement = (
            100 *
            (base_rmse - rmse) /
            base_rmse
        )

        mae_improvement = (
            100 *
            (base_mae - mae) /
            base_mae
        )

        print(
            f"{model_name}: "
            f"RMSE change {rmse_change:+.6f} "
            f"({rmse_improvement:+.2f}% improvement), "
            f"MAE change {mae_change:+.6f} "
            f"({mae_improvement:+.2f}% improvement)"
        )


print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(
    OUT /
    "receiving_efficiency_population.csv"
)

print(
    OUT /
    "receiving_efficiency_cv_folds.csv"
)

print(
    OUT /
    "receiving_efficiency_cv_summary.csv"
)

print("\nDONE")
