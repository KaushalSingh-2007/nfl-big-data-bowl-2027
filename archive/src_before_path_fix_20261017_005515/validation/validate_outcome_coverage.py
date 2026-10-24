from pathlib import Path
import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path("nfl-big-data-bowl-2027")

if not ROOT.exists():
    ROOT = Path(".")

OUT = ROOT / "research_outputs"

DATASET_FILE = OUT / "wr_research_dataset.csv"
PLAY_FILE = ROOT / "player_play.csv"
GAMES_FILE = ROOT / "games.csv"
PLAYERS_FILE = ROOT / "players.csv"

print("Project directory:", ROOT.resolve())
print("Loading files...")


# ============================================================
# 2. LOAD DATA
# ============================================================

dataset = pd.read_csv(DATASET_FILE)
plays = pd.read_csv(PLAY_FILE)
games = pd.read_csv(GAMES_FILE)
players = pd.read_csv(PLAYERS_FILE)

print("Files loaded successfully.")

# Use player names from players.csv instead of assuming the
# research dataset already contains display_name.
player_info = players[
    ["nfl_id", "draft_year", "nfl_position", "display_name"]
].drop_duplicates()

# Confirm the player metadata is unique for the join keys.
if player_info.duplicated(["nfl_id", "draft_year"]).any():
    player_info = player_info.drop_duplicates(
        ["nfl_id", "draft_year"]
    )


# ============================================================
# 3. BUILD POST-DRAFT REGULAR-SEASON PLAY UNIVERSE
# ============================================================

plays = plays.merge(
    games[["game_id", "season", "season_type"]],
    on="game_id",
    how="left",
    validate="many_to_one",
)

plays = plays.merge(
    player_info[["nfl_id", "draft_year", "nfl_position"]],
    on="nfl_id",
    how="left",
    validate="many_to_one",
)

plays = plays[
    (plays["season_type"] == "REG")
    & plays["season"].notna()
    & plays["draft_year"].notna()
    & (plays["season"] >= plays["draft_year"])
    & plays["nfl_position"].isin(["WR", "SLOT_WR"])
].copy()

plays["target"] = pd.to_numeric(
    plays["target"], errors="coerce"
)

print("Post-draft regular-season WR play rows:", len(plays))


# ============================================================
# 4. SUMMARIZE COVERAGE PER PLAYER
# ============================================================

coverage_counts = plays.groupby("nfl_id").agg(
    recorded_postdraft_plays=("game_id", "size"),
    observed_target_flags=("target", "count"),
    targets_from_plays=("target", lambda s: int((s == 1).sum())),
    seasons_observed=("season", "nunique"),
).reset_index()

# Keep every player from the Combine research dataset, including
# those with no matching rows in the play table.
coverage = dataset[
    ["nfl_id", "draft_year", "targets",
     "career_offensive_snaps", "career_games_active"]
].copy()

coverage = coverage.merge(
    player_info[["nfl_id", "display_name"]].drop_duplicates("nfl_id"),
    on="nfl_id",
    how="left",
    validate="many_to_one",
)

coverage = coverage.merge(
    coverage_counts,
    on="nfl_id",
    how="left",
    validate="one_to_one",
)

coverage["has_recorded_plays"] = (
    coverage["recorded_postdraft_plays"].fillna(0) > 0
)

coverage["has_observed_targets"] = (
    coverage["targets_from_plays"].fillna(0) > 0
)


# ============================================================
# 5. REPORT COVERAGE BY DRAFT YEAR
# ============================================================

print("\n" + "=" * 70)
print("COVERAGE BY DRAFT YEAR")
print("=" * 70)

year_report = coverage.groupby("draft_year").agg(
    players=("nfl_id", "nunique"),
    players_with_plays=("has_recorded_plays", "sum"),
    players_with_targets=("has_observed_targets", "sum"),
    median_recorded_plays=("recorded_postdraft_plays", "median"),
    median_seasons=("seasons_observed", "median"),
)

print(year_report.round(2).to_string())


# ============================================================
# 6. PLAYERS WITHOUT OBSERVED TARGETS
# ============================================================

print("\n" + "=" * 70)
print("PLAYERS WITHOUT OBSERVED TARGETS")
print("=" * 70)

no_target_cols = [
    "nfl_id",
    "display_name",
    "draft_year",
    "recorded_postdraft_plays",
    "observed_target_flags",
    "seasons_observed",
    "career_offensive_snaps",
    "career_games_active",
]

no_target_players = coverage.loc[
    ~coverage["has_observed_targets"],
    no_target_cols,
].sort_values(["draft_year", "nfl_id"])

print(no_target_players.to_string(index=False))


# ============================================================
# 7. TARGET DISTRIBUTION BY DRAFT YEAR
# ============================================================

print("\n" + "=" * 70)
print("TARGET COUNTS BY DRAFT YEAR")
print("=" * 70)

players_with_targets = coverage.loc[
    coverage["has_observed_targets"]
].copy()

if not players_with_targets.empty:
    print(
        players_with_targets.groupby("draft_year")[
            "targets_from_plays"
        ]
        .describe()
        .round(2)
        .to_string()
    )
else:
    print("No observed targets found.")


# ============================================================
# 8. SAVE REPORT
# ============================================================

output_file = OUT / "wr_outcome_coverage.csv"

coverage.to_csv(output_file, index=False)

print("\nSaved coverage report:")
print(output_file.resolve())

print("\nImportant:")
print(
    "- No matching play rows does not prove zero NFL opportunity."
)
print(
    "- A recorded play with no target does not mean a target occurred."
)
print(
    "- We will check these distinctions before modelling."
)