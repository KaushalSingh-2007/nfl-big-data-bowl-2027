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

OUT = ROOT / "research_outputs" / "receiving_efficiency_locked"
OUT.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("LOCKED RECEIVING-EFFICIENCY VALIDATION")
print("=" * 80)


# =====================================================================
# 1. LOAD RESEARCH DATA
# =====================================================================

research = pd.read_csv(RESEARCH)

print(f"\nResearch population: {len(research)}")


# =====================================================================
# 2. RECONSTRUCT RECEIVING-EFFICIENCY OUTCOME
# =====================================================================

players = pd.read_csv(
    PLAYERS,
    usecols=["nfl_id", "draft_year", "nfl_position"]
)

games = pd.read_csv(
    GAMES,
    usecols=["game_id", "season", "season_type"]
)

player_play = pd.read_csv(
    PLAYER_PLAY,
    usecols=[
        "game_id",
        "nfl_id",
        "target",
        "rec_yards"
    ]
)

research_ids = set(
    research["nfl_id"]
    .dropna()
    .astype(int)
)

players = players[
    players["nfl_id"].isin(research_ids)
    & players["nfl_position"].eq("WR")
].copy()

players["nfl_id"] = players["nfl_id"].astype(int)

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
    players[["nfl_id", "draft_year"]],
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

df = df[
    (df["season_type"] == "REG")
    & (df["season"] >= df["draft_year"])
].copy()

df["target"] = pd.to_numeric(
    df["target"],
    errors="coerce"
)

df["rec_yards"] = pd.to_numeric(
    df["rec_yards"],
    errors="coerce"
)

target_rows = df[
    df["target"] == 1
].copy()

target_rows["rec_yards"] = target_rows[
    "rec_yards"
].fillna(0)

outcomes = (
    target_rows
    .groupby("nfl_id")
    .agg(
        observed_targets=("target", "count"),
        observed_receiving_yards=("rec_yards", "sum")
    )
    .reset_index()
)

outcomes[
    "observed_yards_per_observed_target"
] = (
    outcomes["observed_receiving_yards"]
    / outcomes["observed_targets"]
)

print(
    f"Players with observed targets: "
    f"{len(outcomes)}"
)

print(
    "\nReceiving-efficiency target:"
)

print(
    outcomes["observed_yards_per_observed_target"]
    .describe()
    .to_string()
)


# =====================================================================
# 3. MERGE OUTCOME WITH RESEARCH FEATURES
# =====================================================================

data = research.merge(
    outcomes,
    on="nfl_id",
    how="left",
    validate="one_to_one"
)

TARGET = "observed_yards_per_observed_target"

CONVENTIONAL = [
    "forty",
    "vertical"
]

MOVEMENT = [
    "speed_cv_DAGGER_ROUTE_LEFT"
]

COMBINED = [
    "forty",
    "vertical",
    "speed_cv_DAGGER_ROUTE_LEFT"
]


# =====================================================================
# 4. LOCKED MODEL FUNCTION
# =====================================================================

def run_locked_cv(data, features, model_name, population_name):

    required = features + [
        TARGET,
        "nfl_id",
        "observed_targets"
    ]

    d = data[required].dropna().copy()

    # Every model gets its own supplied feature columns,
    # but this function will only be called on a population
    # where ALL three model feature sets are complete.

    X = d[features].to_numpy()
    y = d[TARGET].to_numpy()

    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(
        rkf.split(X),
        start=1
    ):

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ])

        model.fit(
            X[train_idx],
            y[train_idx]
        )

        pred = model.predict(
            X[test_idx]
        )

        rows.append({
            "population": population_name,
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

    return pd.DataFrame(rows)


# =====================================================================
# 5. LOCKED POPULATIONS
#
# IMPORTANT:
# Every population is first restricted to rows complete for:
#   forty
#   vertical
#   speed_cv
#   outcome
#
# Therefore A/B/C use IDENTICAL players.
# =====================================================================

ALL_FEATURES = [
    "forty",
    "vertical",
    "speed_cv_DAGGER_ROUTE_LEFT",
    TARGET
]

base = data.dropna(
    subset=ALL_FEATURES
).copy()

print("\n" + "=" * 80)
print("LOCKED POPULATION")
print("=" * 80)

print(
    f"All-complete common population: "
    f"{len(base)}"
)

print(
    "Draft years:"
)

print(
    base["draft_year"]
    .value_counts()
    .sort_index()
    .to_string()
)


populations = [
    (
        "all_common_complete",
        base
    ),
    (
        "at_least_10_observed_targets",
        base[
            base["observed_targets"] >= 10
        ].copy()
    ),
    (
        "at_least_20_observed_targets",
        base[
            base["observed_targets"] >= 20
        ].copy()
    )
]


# =====================================================================
# 6. RUN ALL THREE MODELS ON IDENTICAL POPULATIONS
# =====================================================================

fold_results = []

for population_name, population in populations:

    print(
        f"\n{population_name}: "
        f"{len(population)} players"
    )

    for model_name, features in [
        ("A_conventional", CONVENTIONAL),
        ("B_movement", MOVEMENT),
        ("C_combined", COMBINED)
    ]:

        folds = run_locked_cv(
            population,
            features,
            model_name,
            population_name
        )

        fold_results.append(folds)


folds = pd.concat(
    fold_results,
    ignore_index=True
)


# =====================================================================
# 7. SUMMARY
# =====================================================================

summary = (
    folds
    .groupby(
        ["population", "model"],
        as_index=False
    )
    .agg(
        n_players=("n_test", lambda x: int(
            x.iloc[0] * 0
        )),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std")
    )
)

# Put the actual population size into summary.
population_sizes = {
    name: len(pop)
    for name, pop in populations
}

summary["n_players"] = summary[
    "population"
].map(population_sizes)


# =====================================================================
# 8. PAIRED FOLD DIFFERENCES
#
# Same fold number means same train/test split.
# =====================================================================

paired_rows = []

for population_name in folds["population"].unique():

    f = folds[
        folds["population"] == population_name
    ].copy()

    pivot = f.pivot(
        index="fold",
        columns="model",
        values=["rmse", "mae", "r2"]
    )

    for comparison in [
        "B_vs_A",
        "C_vs_A"
    ]:

        if comparison == "B_vs_A":
            model = "B_movement"
        else:
            model = "C_combined"

        paired_rows.append({
            "population": population_name,
            "comparison": comparison,
            "rmse_diff_mean": (
                pivot["rmse"][model]
                - pivot["rmse"]["A_conventional"]
            ).mean(),
            "rmse_diff_std": (
                pivot["rmse"][model]
                - pivot["rmse"]["A_conventional"]
            ).std(ddof=1),
            "mae_diff_mean": (
                pivot["mae"][model]
                - pivot["mae"]["A_conventional"]
            ).mean(),
            "mae_diff_std": (
                pivot["mae"][model]
                - pivot["mae"]["A_conventional"]
            ).std(ddof=1),
            "r2_diff_mean": (
                pivot["r2"][model]
                - pivot["r2"]["A_conventional"]
            ).mean(),
            "r2_diff_std": (
                pivot["r2"][model]
                - pivot["r2"]["A_conventional"]
            ).std(ddof=1),
            "rmse_folds_improved": int(
                (
                    pivot["rmse"][model]
                    <
                    pivot["rmse"]["A_conventional"]
                ).sum()
            ),
            "rmse_folds_worse": int(
                (
                    pivot["rmse"][model]
                    >
                    pivot["rmse"]["A_conventional"]
                ).sum()
            )
        })

paired = pd.DataFrame(paired_rows)


# =====================================================================
# 9. SAVE
# =====================================================================

data.to_csv(
    OUT / "receiving_efficiency_locked_population.csv",
    index=False
)

folds.to_csv(
    OUT / "receiving_efficiency_locked_folds.csv",
    index=False
)

summary.to_csv(
    OUT / "receiving_efficiency_locked_summary.csv",
    index=False
)

paired.to_csv(
    OUT / "receiving_efficiency_locked_paired_differences.csv",
    index=False
)


# =====================================================================
# 10. PRINT RESULTS
# =====================================================================

print("\n" + "=" * 80)
print("LOCKED CV SUMMARY")
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
print("PAIRED DIFFERENCES")
print("=" * 80)

print(
    paired.to_string(index=False)
)


print("\n" + "=" * 80)
print("PERCENT IMPROVEMENT VS CONVENTIONAL")
print("=" * 80)

for population_name in summary["population"].unique():

    s = summary[
        summary["population"] == population_name
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
        f"\n{population_name}"
    )

    for model in [
        "B_movement",
        "C_combined"
    ]:

        rmse = s.loc[
            model,
            "rmse_mean"
        ]

        mae = s.loc[
            model,
            "mae_mean"
        ]

        rmse_pct = (
            100 *
            (base_rmse - rmse) /
            base_rmse
        )

        mae_pct = (
            100 *
            (base_mae - mae) /
            base_mae
        )

        print(
            f"{model}: "
            f"RMSE {rmse_pct:+.2f}% | "
            f"MAE {mae_pct:+.2f}%"
        )


print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

for p in sorted(OUT.glob("*.csv")):
    print(p)

print("\nDONE")
