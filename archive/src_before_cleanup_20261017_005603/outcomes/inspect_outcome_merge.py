from pathlib import Path
import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)

print("Project directory:", ROOT.resolve())
print("Loading source data...")


# ============================================================
# 2. LOAD DATA
# ============================================================

combine = pd.read_csv(DATA_DIR / "combine_results.csv")
players = pd.read_csv(DATA_DIR / "players.csv")
career = pd.read_csv(DATA_DIR / "player_career_successes.csv")
player_play = pd.read_csv(DATA_DIR / "player_play.csv")
games = pd.read_csv(DATA_DIR / "games.csv")

route_features = pd.read_csv(
    OUT / "wr_player_route_features.csv"
)

print("Loaded source data.")


# ============================================================
# 3. FILTER POST-DRAFT REGULAR-SEASON PLAYS
# ============================================================

# Attach season and season type to each play.
player_play = player_play.merge(
    games[["game_id", "season", "season_type"]],
    on="game_id",
    how="left",
    validate="many_to_one",
)

# Attach player position and draft year.
player_play = player_play.merge(
    players[["nfl_id", "draft_year", "nfl_position"]],
    on="nfl_id",
    how="left",
    validate="many_to_one",
)

# Keep regular-season plays in or after the player's draft year.
player_play = player_play[
    (player_play["season_type"] == "REG")
    & player_play["season"].notna()
    & player_play["draft_year"].notna()
    & (player_play["season"] >= player_play["draft_year"])
].copy()

# Restrict the outcome analysis to wide receivers.
wr = player_play[
    player_play["nfl_position"].isin(["WR", "SLOT_WR"])
].copy()

# Ensure target values are handled safely.
wr["target"] = pd.to_numeric(wr["target"], errors="coerce")

targeted = wr[wr["target"] == 1].copy()

print("Post-draft WR play rows:", len(wr))
print("Targeted WR play rows:", len(targeted))


# ============================================================
# 4. AGGREGATE RECEIVING OUTCOMES
# ============================================================

# Count targets independently from completed receptions.
# rec_yards is only observed for a subset of plays, so its
# non-missing count is not a complete official reception count.
outcomes = targeted.groupby("nfl_id").agg(
    targets=("target", "size"),
    mean_separation=(
        "separation_at_pass_forward", "mean"
    ),
    observed_receiving_yard_rows=(
        "rec_yards", "count"
    ),
    receiving_yards_observed_sum=(
        "rec_yards", "sum"
    ),
    mean_yards_after_catch=(
        "yards_after_catch", "mean"
    ),
).reset_index()

# Keep these outcomes explicitly labelled as observed values.
# Do not interpret the observed receiving-yard sum as a complete
# career total until the source's missing-value semantics are checked.


# ============================================================
# 5. PREPARE CAREER OUTCOMES
# ============================================================

career_columns = [
    "nfl_id",
    "career_offensive_snaps",
    "career_games_active",
    "career_games_started",
    "ap_all_pro_1st_team",
    "ap_all_pro_2nd_team",
    "pro_bowl_original_ballot",
]

career_subset = career[career_columns].copy()

# ============================================================
# 6. PREPARE COMBINE BASE DATASET
# ============================================================

player_columns = [
    "nfl_id",
    "nfl_position",
    "draft_year",
    "draft_round",
    "draft_pick_within_round",
    "draft_overall_pick",
]

# Use player metadata to identify Combine WRs.
base = combine.merge(
    players[player_columns],
    on=["nfl_id", "draft_year"],
    how="left",
    validate="many_to_one",
)

base = base[
    base["nfl_position"].isin(["WR", "SLOT_WR"])
].copy()

# Attach career outcomes.
base = base.merge(
    career_subset,
    on="nfl_id",
    how="left",
    validate="one_to_one",
)


# ============================================================
# 7. PIVOT ROUTE FEATURES AND FLATTEN COLUMN NAMES
# ============================================================

feature_columns = [
    "straightness_ratio",
    "peak_speed",
    "mean_speed",
    "speed_cv",
    "late_early_speed_ratio",
]

route_wide = route_features.pivot(
    index="nfl_id",
    columns="drill_name",
    values=feature_columns,
)

# Flatten the two-level columns produced by pivot().
route_wide.columns = [
    f"{feature}_{route}"
    for feature, route in route_wide.columns
]

route_wide = route_wide.reset_index()

# Ensure there is at most one row per player in the pivoted table.
if route_wide["nfl_id"].duplicated().any():
    raise ValueError("Route feature pivot has duplicate player IDs.")


# ============================================================
# 8. MERGE RECEIVING OUTCOMES AND ROUTE FEATURES
# ============================================================

dataset = base.merge(
    outcomes,
    on="nfl_id",
    how="left",
    validate="one_to_one",
)

dataset = dataset.merge(
    route_wide,
    on="nfl_id",
    how="left",
    validate="one_to_one",
)

# Missing outcome rows mean no qualifying targeted plays were
# observed in the filtered play table. Keep them missing for now;
# we will investigate whether these represent true zero targets
# or incomplete tracking coverage before modelling.


# ============================================================
# 9. SAVE DATASET
# ============================================================

output_file = OUT / "wr_research_dataset.csv"

dataset.to_csv(output_file, index=False)


# ============================================================
# 10. PRINT VALIDATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("DATASET MERGE REPORT")
print("=" * 65)

print("\nDataset shape:", dataset.shape)
print("Unique players:", dataset["nfl_id"].nunique())
print("Duplicate player IDs:", dataset["nfl_id"].duplicated().sum())

print(
    "Players with observed target rows:",
    int(dataset["targets"].notna().sum()),
)

print("\nTarget count distribution among players with observed targets:")

if dataset["targets"].notna().any():
    print(
        dataset.loc[dataset["targets"].notna(), "targets"]
        .describe()
        .round(2)
        .to_string()
    )
else:
    print("No observed targets found.")

print("\nMissingness in important fields:")

important_fields = [
    "forty",
    "vertical",
    "short_shuttle",
    "targets",
    "mean_separation",
    "receiving_yards_observed_sum",
    "straightness_ratio_COMEBACK_ROUTE_RIGHT",
    "straightness_ratio_DAGGER_ROUTE_LEFT",
]

available_fields = [
    col for col in important_fields
    if col in dataset.columns
]

print(
    dataset[available_fields]
    .isna()
    .mean()
    .round(3)
    .to_string()
)

print("\nWRs by draft year:")
print(
    dataset.groupby("draft_year")["nfl_id"]
    .nunique()
    .to_string()
)

print("\nSaved dataset:", output_file.resolve())
print("\nMerge complete. No prediction model has been trained yet.")

