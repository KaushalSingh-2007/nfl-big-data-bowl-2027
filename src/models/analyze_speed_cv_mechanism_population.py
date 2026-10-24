import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RESEARCH = os.path.join(
    ROOT,
    "research_outputs",
    "wr_research_dataset.csv"
)

MECH_DIR = os.path.join(
    ROOT,
    "research_outputs",
    "speed_cv_mechanism_exact"
)

MECH = os.path.join(
    MECH_DIR,
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
    MECH_DIR,
    "population_analysis"
)

os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("MECHANISM ANALYSIS — FEATURE-SPECIFIC POPULATIONS")
print("=" * 70)

# ---------------------------------------------------------------------
# Load research population and exact mechanism features.
# ---------------------------------------------------------------------
research = pd.read_csv(RESEARCH)
mechanism = pd.read_csv(MECH)

print(f"Research rows: {len(research)}")
print(f"Mechanism rows: {len(mechanism)}")

df = research.merge(
    mechanism,
    on="nfl_id",
    how="inner"
)

print(f"Merged rows: {len(df)}")

# ---------------------------------------------------------------------
# RECONSTRUCT RECEIVING EFFICIENCY.
#
# This is intentionally the same definition used by the locked
# receiving-efficiency validation:
#
#   WR
#   regular season
#   post-draft
#   target == 1
#   rec_yards numeric
#   missing rec_yards -> 0
#
# receiving_efficiency =
#     sum(rec_yards on target rows)
#     /
#     number of target rows
# ---------------------------------------------------------------------
print()
print("Reconstructing receiving efficiency from raw player-play data...")

pp_cols = [
    "game_id",
    "play_id",
    "nfl_id",
    "target",
    "rec_yards",
]

pp = pd.read_csv(
    PLAYER_PLAY,
    usecols=pp_cols
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

# Keep only WRs represented in the research population.
research_ids = set(
    research["nfl_id"].dropna().astype(int)
)

players["nfl_id"] = pd.to_numeric(
    players["nfl_id"],
    errors="coerce"
)

pp["nfl_id"] = pd.to_numeric(
    pp["nfl_id"],
    errors="coerce"
)

pp = pp[
    pp["nfl_id"].isin(research_ids)
].copy()

players_wr = players[
    (players["nfl_position"] == "WR") &
    (players["nfl_id"].isin(research_ids))
].copy()

pp = pp.merge(
    players_wr[["nfl_id", "draft_year"]],
    on="nfl_id",
    how="inner"
)

games["season"] = pd.to_numeric(
    games["season"],
    errors="coerce"
)

pp = pp.merge(
    games[["game_id", "season", "season_type"]],
    on="game_id",
    how="inner"
)

# Post-draft regular-season plays only.
pp = pp[
    (pp["season_type"] == "REG") &
    (pp["season"] >= pp["draft_year"])
].copy()

print(f"Post-draft regular-season player-play rows: {len(pp):,}")

# Target rows.
pp["target"] = pd.to_numeric(
    pp["target"],
    errors="coerce"
).fillna(0)

target_rows = pp[
    pp["target"] == 1
].copy()

target_rows["rec_yards"] = pd.to_numeric(
    target_rows["rec_yards"],
    errors="coerce"
).fillna(0)

outcome = (
    target_rows
    .groupby("nfl_id")
    .agg(
        observed_targets=("target", "size"),
        observed_receiving_yards=("rec_yards", "sum"),
    )
    .reset_index()
)

outcome["receiving_efficiency"] = (
    outcome["observed_receiving_yards"]
    / outcome["observed_targets"]
)

print(
    f"Players with observed targets: {len(outcome)}"
)

print(
    f"Receiving-efficiency mean: "
    f"{outcome['receiving_efficiency'].mean():.6f}"
)

# Merge reconstructed outcome.
df = df.drop(
    columns=[
        "observed_targets",
        "observed_receiving_yards",
        "receiving_efficiency",
    ],
    errors="ignore"
)

df = df.merge(
    outcome,
    on="nfl_id",
    how="left"
)

print(
    f"Players with receiving efficiency after merge: "
    f"{df['receiving_efficiency'].notna().sum()}"
)

# ---------------------------------------------------------------------
# Candidate mechanism variables.
# ---------------------------------------------------------------------
mechanism_cols = [
    "speed_cv",
    "speed_range",
    "early_speed_median",
    "late_speed_median",
    "late_early_speed_ratio",
    "mean_positive_dsdt",
    "mean_negative_dsdt",
    "max_positive_dsdt",
    "max_negative_dsdt",
    "mean_abs_dsdt",
    "frac_accelerating",
    "frac_decelerating",
    "frac_stable_speed",
    "speed_sign_transitions",
    "frac_above_median_speed",
    "frac_below_median_speed",
]

# ---------------------------------------------------------------------
# Locked baseline population.
# ---------------------------------------------------------------------
baseline = df.dropna(
    subset=[
        "forty",
        "vertical",
        "receiving_efficiency",
    ]
).copy()

print()
print(
    "Baseline population "
    "(40 yd + vertical + receiving efficiency): "
    f"{len(baseline)}"
)

# ---------------------------------------------------------------------
# Residualized Spearman:
# remove linear effects of 40-yard + vertical from both variables.
#
# Exploratory association only; NOT causal inference.
# ---------------------------------------------------------------------
def residualized_spearman(data, feature):

    tmp = data[
        [
            "forty",
            "vertical",
            feature,
            "receiving_efficiency",
        ]
    ].dropna()

    if len(tmp) < 10:
        return np.nan, np.nan, len(tmp)

    X = tmp[
        ["forty", "vertical"]
    ].to_numpy()

    feature_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    )

    outcome_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    )

    feature_model.fit(
        X,
        tmp[feature].to_numpy()
    )

    outcome_model.fit(
        X,
        tmp["receiving_efficiency"].to_numpy()
    )

    feature_resid = (
        tmp[feature].to_numpy()
        - feature_model.predict(X)
    )

    outcome_resid = (
        tmp["receiving_efficiency"].to_numpy()
        - outcome_model.predict(X)
    )

    r, p = spearmanr(
        feature_resid,
        outcome_resid
    )

    return r, p, len(tmp)

# ---------------------------------------------------------------------
# Analyze each feature independently.
# ---------------------------------------------------------------------
rows = []

for feature in mechanism_cols:

    if feature not in df.columns:
        continue

    tmp = df[
        [
            "nfl_id",
            "draft_year",
            "forty",
            "vertical",
            "receiving_efficiency",
            feature,
        ]
    ].dropna(
        subset=[
            feature,
            "receiving_efficiency",
        ]
    ).copy()

    n = len(tmp)

    if n < 10:
        continue

    pearson_r, pearson_p = pearsonr(
        tmp[feature],
        tmp["receiving_efficiency"]
    )

    spearman_r, spearman_p = spearmanr(
        tmp[feature],
        tmp["receiving_efficiency"]
    )

    residual_r, residual_p, residual_n = (
        residualized_spearman(
            df,
            feature
        )
    )

    rows.append({
        "feature": feature,
        "n": n,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "residualized_spearman_r": residual_r,
        "residualized_p": residual_p,
        "residualized_n": residual_n,
        "draft_2023_n": int(
            (tmp["draft_year"] == 2023).sum()
        ),
        "draft_2024_n": int(
            (tmp["draft_year"] == 2024).sum()
        ),
        "draft_2025_n": int(
            (tmp["draft_year"] == 2025).sum()
        ),
    })

results = pd.DataFrame(rows)

results["abs_spearman"] = (
    results["spearman_r"].abs()
)

results["abs_residualized"] = (
    results["residualized_spearman_r"].abs()
)

results = results.sort_values(
    "abs_residualized",
    ascending=False
)

print()
print("=" * 70)
print("ALL FEATURES — FEATURE-SPECIFIC POPULATIONS")
print("=" * 70)

print(
    results[
        [
            "feature",
            "n",
            "spearman_r",
            "spearman_p",
            "residualized_spearman_r",
            "residualized_p",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

results.to_csv(
    os.path.join(
        OUT,
        "mechanism_population_correlations.csv"
    ),
    index=False
)

# ---------------------------------------------------------------------
# Focus features from the previous n=39 exploratory analysis.
# ---------------------------------------------------------------------
focus = [
    "frac_stable_speed",
    "speed_sign_transitions",
    "frac_decelerating",
]

print()
print("=" * 70)
print("FOCUS FEATURES")
print("=" * 70)

print(
    results[
        results["feature"].isin(focus)
    ][
        [
            "feature",
            "n",
            "spearman_r",
            "spearman_p",
            "residualized_spearman_r",
            "residualized_p",
            "draft_2023_n",
            "draft_2024_n",
            "draft_2025_n",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# ---------------------------------------------------------------------
# Availability inside the exact locked 64-player population.
# ---------------------------------------------------------------------
locked = df.dropna(
    subset=[
        "forty",
        "vertical",
        "speed_cv_DAGGER_ROUTE_LEFT",
        "receiving_efficiency",
    ]
).copy()

print()
print("=" * 70)
print("LOCKED 64-PLAYER POPULATION")
print("=" * 70)

print(
    f"Locked population: {len(locked)}"
)

availability = []

for feature in mechanism_cols:

    if feature not in locked.columns:
        continue

    available = locked[feature].notna()

    availability.append({
        "feature": feature,
        "locked_population": len(locked),
        "available": int(available.sum()),
        "missing": int((~available).sum()),
        "coverage_pct": (
            100 * available.mean()
        ),
    })

availability = pd.DataFrame(
    availability
).sort_values(
    "coverage_pct",
    ascending=False
)

print(
    availability.round(2).to_string(
        index=False
    )
)

availability.to_csv(
    os.path.join(
        OUT,
        "mechanism_population_availability.csv"
    ),
    index=False
)

# ---------------------------------------------------------------------
# Save the actual merged player-level analysis table.
# ---------------------------------------------------------------------
df.to_csv(
    os.path.join(
        OUT,
        "mechanism_population_merged.csv"
    ),
    index=False
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "Correlation results:"
)
print(
    os.path.join(
        OUT,
        "mechanism_population_correlations.csv"
    )
)

print(
    "Availability results:"
)
print(
    os.path.join(
        OUT,
        "mechanism_population_availability.csv"
    )
)
