from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESEARCH_DATA = (
    PROJECT_ROOT
    / "research_outputs"
    / "wr_research_dataset.csv"
)

PLAYER_PLAY = (
    PROJECT_ROOT
    / "data"
    / "player_play.csv"
)

PLAYERS = (
    PROJECT_ROOT
    / "data"
    / "players.csv"
)

GAMES = (
    PROJECT_ROOT
    / "data"
    / "games.csv"
)

OUT = (
    PROJECT_ROOT
    / "research_outputs"
    / "model_comparison"
    / "separation_reliability_robustness"
)

OUT.mkdir(parents=True, exist_ok=True)


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

# Reliability thresholds.
# 0 = all players with an observed outcome.
# 10/20/30 = increasingly reliable separation estimates.
MIN_TARGET_ROWS = [0, 10, 20, 30]

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10


# ============================================================
# LOAD RESEARCH DATA
# ============================================================

print("=" * 78)
print("SEPARATION RELIABILITY ROBUSTNESS ANALYSIS")
print("=" * 78)

print()
print("Loading research dataset...")

df = pd.read_csv(RESEARCH_DATA)

print(f"Research dataset shape: {df.shape}")


required = [
    "nfl_id",
    "draft_year",
    "mean_separation",
    "targets",
    "forty",
    "vertical",
    "speed_cv_DAGGER_ROUTE_LEFT"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns in research dataset: "
        + ", ".join(missing)
    )


# ============================================================
# LOAD RAW OUTCOME DATA
# ============================================================

print()
print("Loading player_play.csv...")

pp = pd.read_csv(
    PLAYER_PLAY,
    usecols=[
        "game_id",
        "play_id",
        "nfl_id",
        "target",
        "separation_at_pass_forward"
    ]
)

print(f"player_play rows loaded: {len(pp):,}")


# ============================================================
# LOAD PLAYER POSITION / DRAFT INFORMATION
# ============================================================

players = pd.read_csv(
    PLAYERS,
    usecols=[
        "nfl_id",
        "nfl_position",
        "draft_year"
    ]
)

print(f"players rows loaded: {len(players):,}")


# ============================================================
# LOAD GAME INFORMATION
# ============================================================

games = pd.read_csv(
    GAMES,
    usecols=[
        "game_id",
        "season_type"
    ]
)

print(f"games rows loaded: {len(games):,}")


# ============================================================
# RECONSTRUCT OUTCOME POPULATION
# ============================================================

print()
print("Constructing separation observations...")

outcomes = pp.merge(
    players,
    on="nfl_id",
    how="inner"
)

outcomes = outcomes.merge(
    games,
    on="game_id",
    how="inner"
)

# Same conceptual population used in the existing research:
# WRs, post-draft, regular-season observations.
outcomes = outcomes[
    outcomes["nfl_position"].eq("WR")
    &
    outcomes["season_type"].eq("REG")
    &
    outcomes["draft_year"].notna()
].copy()

# Only actual target observations with observed separation.
outcomes = outcomes[
    outcomes["target"].eq(1)
    &
    outcomes["separation_at_pass_forward"].notna()
].copy()

print(
    "Observed WR separation rows:",
    f"{len(outcomes):,}"
)

print(
    "Unique players with observed separation:",
    outcomes["nfl_id"].nunique()
)


# ============================================================
# COMPUTE PLAYER-LEVEL MEDIAN SEPARATION
# ============================================================

median_outcomes = (
    outcomes
    .groupby("nfl_id")
    ["separation_at_pass_forward"]
    .agg(
        median_separation="median",
        observed_separation_rows="count"
    )
    .reset_index()
)

print()
print("Median separation table:")
print(
    median_outcomes.head().to_string(index=False)
)


# ============================================================
# MERGE MEDIAN INTO RESEARCH DATASET
# ============================================================

df = df.merge(
    median_outcomes[
        [
            "nfl_id",
            "median_separation",
            "observed_separation_rows"
        ]
    ],
    on="nfl_id",
    how="left"
)


# ============================================================
# VERIFY TARGET COUNT CONSISTENCY
# ============================================================

check = df[
    ["nfl_id", "targets", "observed_separation_rows"]
].dropna(
    subset=["targets", "observed_separation_rows"]
).copy()

check["target_count_difference"] = (
    check["targets"]
    -
    check["observed_separation_rows"]
)

print()
print("=" * 78)
print("TARGET COUNT CONSISTENCY CHECK")
print("=" * 78)

print(
    check["target_count_difference"]
    .describe()
    .to_string()
)

n_mismatch = (
    check["target_count_difference"]
    .ne(0)
    .sum()
)

print()
print(
    f"Players with target-count mismatch: "
    f"{n_mismatch}"
)

if n_mismatch > 0:

    print()
    print(
        "WARNING: targets and reconstructed observed "
        "separation counts are not identical."
    )

    print(
        "The robustness thresholds will use the "
        "reconstructed observed separation count."
    )


# ============================================================
# COMMON MODEL POPULATION
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
print("=" * 78)
print("COMMON MODEL POPULATION")
print("=" * 78)

print(
    f"Players with complete model features: {len(base)}"
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


# ============================================================
# RELIABILITY DISTRIBUTION
# ============================================================

print()
print("=" * 78)
print("OBSERVED SEPARATION ROW DISTRIBUTION")
print("=" * 78)

print(
    base["observed_separation_rows"]
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
# EXPERIMENT
# ============================================================

def run_experiment(
    data,
    target_column,
    population_name,
    minimum_rows
):

    eligible = data[
        data[target_column].notna()
        &
        (
            data["observed_separation_rows"]
            >= minimum_rows
        )
    ].copy()

    eligible = eligible.reset_index(drop=True)

    print()
    print("-" * 78)
    print(f"Population: {population_name}")
    print(f"Target: {target_column}")
    print(f"Minimum observed rows: {minimum_rows}")
    print(f"Eligible players: {len(eligible)}")

    if len(eligible) < 20:

        print(
            "SKIPPED: fewer than 20 players."
        )

        return None, None, None

    print(
        "Draft-year counts:"
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
        random_state=RANDOM_STATE
    )

    results = []

    for split_id, (
        train_idx,
        test_idx
    ) in enumerate(
        cv.split(eligible),
        start=1
    ):

        train = eligible.iloc[train_idx]
        test = eligible.iloc[test_idx]

        for model_name, feature_list in FEATURES.items():

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
                    "target": target_column,
                    "minimum_observed_rows": minimum_rows,
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
    # Paired fold comparisons
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
                "target": target_column,
                "minimum_observed_rows": minimum_rows,
                "comparison": "B_movement_vs_A_conventional",
                "rmse_difference_mean":
                    wide["rmse_B_minus_A"].mean(),
                "rmse_difference_std":
                    wide["rmse_B_minus_A"].std(),
                "mae_difference_mean":
                    wide["mae_B_minus_A"].mean(),
                "r2_difference_mean":
                    wide["r2_B_minus_A"].mean(),
                "folds_rmse_improved":
                    (wide["rmse_B_minus_A"] < 0).sum(),
                "folds_rmse_worse":
                    (wide["rmse_B_minus_A"] > 0).sum()
            },
            {
                "population": population_name,
                "target": target_column,
                "minimum_observed_rows": minimum_rows,
                "comparison": "C_combined_vs_A_conventional",
                "rmse_difference_mean":
                    wide["rmse_C_minus_A"].mean(),
                "rmse_difference_std":
                    wide["rmse_C_minus_A"].std(),
                "mae_difference_mean":
                    wide["mae_C_minus_A"].mean(),
                "r2_difference_mean":
                    wide["r2_C_minus_A"].mean(),
                "folds_rmse_improved":
                    (wide["rmse_C_minus_A"] < 0).sum(),
                "folds_rmse_worse":
                    (wide["rmse_C_minus_A"] > 0).sum()
            }
        ]
    )

    return results, summary, comparison


# ============================================================
# RUN MEAN-SEPARATION ROBUSTNESS
# ============================================================

mean_folds = []
mean_summaries = []
mean_comparisons = []

for minimum_rows in MIN_TARGET_ROWS:

    result = run_experiment(
        base,
        "mean_separation",
        f"mean_separation_min_{minimum_rows}",
        minimum_rows
    )

    folds, summary, comparison = result

    if folds is not None:

        mean_folds.append(folds)
        mean_summaries.append(summary)
        mean_comparisons.append(comparison)


# ============================================================
# SAVE MEAN RESULTS
# ============================================================

if mean_folds:

    mean_folds_df = pd.concat(
        mean_folds,
        ignore_index=True
    )

    mean_summary_df = pd.concat(
        mean_summaries,
        ignore_index=True
    )

    mean_comparison_df = pd.concat(
        mean_comparisons,
        ignore_index=True
    )

    mean_folds_df.to_csv(
        OUT / "mean_separation_robustness_folds.csv",
        index=False
    )

    mean_summary_df.to_csv(
        OUT / "mean_separation_robustness_summary.csv",
        index=False
    )

    mean_comparison_df.to_csv(
        OUT / "mean_separation_robustness_comparisons.csv",
        index=False
    )


# ============================================================
# RUN MEDIAN-SEPARATION ROBUSTNESS
# ============================================================

print()
print("=" * 78)
print("MEDIAN SEPARATION ROBUSTNESS")
print("=" * 78)

median_result = run_experiment(
    base,
    "median_separation",
    "median_separation",
    0
)

median_folds, median_summary, median_comparison = median_result

if median_folds is not None:

    median_folds.to_csv(
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
# POPULATION COUNT TABLE
# ============================================================

population_rows = []

for minimum_rows in MIN_TARGET_ROWS:

    eligible = base[
        base["mean_separation"].notna()
        &
        (
            base["observed_separation_rows"]
            >= minimum_rows
        )
    ]

    population_rows.append(
        {
            "minimum_observed_rows": minimum_rows,
            "eligible_players": len(eligible),
            "draft_2023":
                (eligible["draft_year"] == 2023).sum(),
            "draft_2024":
                (eligible["draft_year"] == 2024).sum(),
            "draft_2025":
                (eligible["draft_year"] == 2025).sum()
        }
    )

population_table = pd.DataFrame(
    population_rows
)

population_table.to_csv(
    OUT / "reliability_population_counts.csv",
    index=False
)


# ============================================================
# SAVE MERGED AUDIT TABLE
# ============================================================

audit_columns = [
    "nfl_id",
    "draft_year",
    "mean_separation",
    "median_separation",
    "targets",
    "observed_separation_rows",
    "forty",
    "vertical",
    "speed_cv_DAGGER_ROUTE_LEFT"
]

df[audit_columns].to_csv(
    OUT / "separation_reliability_player_table.csv",
    index=False
)


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print()
print("=" * 78)
print("FINAL ROBUSTNESS SUMMARY")
print("=" * 78)

if mean_summaries:

    final_summary = pd.concat(
        mean_summaries,
        ignore_index=True
    )

    print()
    print(
        final_summary[
            [
                "model",
                "n_players",
                "rmse_mean",
                "rmse_std",
                "mae_mean",
                "mae_std",
                "r2_mean",
                "r2_std"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


print()
print("=" * 78)
print("POPULATION SIZES")
print("=" * 78)

print(
    population_table.to_string(
        index=False
    )
)


if mean_comparisons:

    final_comparisons = pd.concat(
        mean_comparisons,
        ignore_index=True
    )

    print()
    print("=" * 78)
    print("PAIRED MODEL COMPARISONS")
    print("=" * 78)

    print(
        final_comparisons.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


if median_summary is not None:

    print()
    print("=" * 78)
    print("MEDIAN SEPARATION SUMMARY")
    print("=" * 78)

    print(
        median_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print("Median paired comparisons:")

    print(
        median_comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


print()
print("=" * 78)
print("DONE")
print("=" * 78)

print()
print("All robustness outputs saved to:")
print(OUT)
