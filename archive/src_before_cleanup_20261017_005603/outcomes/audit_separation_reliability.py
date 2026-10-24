from pathlib import Path
import pandas as pd

# Locate the project folder from either the parent or inner directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "research_outputs"

print("Project folder:", ROOT.resolve())
print("Loading existing files...")

# Load only the columns needed for this audit.
plays = pd.read_csv(
    DATA_DIR / "player_play.csv",
    usecols=[
        "game_id",
        "nfl_id",
        "lined_up_position",
        "target",
        "separation_at_pass_forward",
    ],
)

games = pd.read_csv(
    DATA_DIR / "games.csv",
    usecols=["game_id", "season", "season_type"],
)

players = pd.read_csv(
    DATA_DIR / "players.csv",
    usecols=["nfl_id", "draft_year", "nfl_position"],
)

# Attach season information and player draft-year/position information.
plays = plays.merge(
    games,
    on="game_id",
    how="left",
    validate="many_to_one",
)

plays = plays.merge(
    players,
    on="nfl_id",
    how="left",
    validate="many_to_one",
)

# Apply the documented research filters.
plays = plays[
    (plays["season_type"] == "REG")
    & (plays["season"] >= plays["draft_year"])
    & (plays["lined_up_position"].isin(["WR", "SLOT_WR"]))
].copy()

# Restrict to target rows.
target_rows = plays[
    pd.to_numeric(plays["target"], errors="coerce") == 1
].copy()

# Count how many target plays have an actual observed separation value.
target_rows["separation_at_pass_forward"] = pd.to_numeric(
    target_rows["separation_at_pass_forward"],
    errors="coerce",
)

audit = target_rows.groupby("nfl_id").agg(
    targets_recounted=("target", "size"),
    separation_count=("separation_at_pass_forward", "count"),
    mean_separation_recomputed=(
        "separation_at_pass_forward",
        "mean",
    ),
).reset_index()

# Merge the audit into the existing dataset without changing the original.
dataset_path = OUT / "wr_research_dataset.csv"
dataset = pd.read_csv(dataset_path)

result = dataset.merge(
    audit,
    on="nfl_id",
    how="left",
    validate="one_to_one",
)

count_cols = ["targets_recounted", "separation_count"]
result[count_cols] = result[count_cols].fillna(0).astype(int)

result["target_count_matches"] = (
    result["targets"].fillna(0).astype(int)
    == result["targets_recounted"]
)

result["separation_missing_fraction"] = (
    1 - result["separation_count"]
    / result["targets_recounted"].replace(0, float("nan"))
)

# Save a separate audit file. The original dataset is untouched.
output_path = OUT / "wr_separation_reliability.csv"
result.to_csv(output_path, index=False)

print("\n=== AUDIT SUMMARY ===")
print("Research dataset rows:", len(result))
print("Players with at least one observed separation:",
      int((result["separation_count"] > 0).sum()))
print("Players with zero observed separation:",
      int((result["separation_count"] == 0).sum()))
print("Target counts matching existing dataset:",
      int(result["target_count_matches"].sum()), "/", len(result))

print("\n=== SEPARATION OBSERVATION COUNTS ===")
print(result["separation_count"].describe().round(2).to_string())

print("\n=== PLAYERS BY OBSERVED SEPARATION COUNT ===")
for threshold in [1, 10, 20, 30, 50, 100]:
    count = int((result["separation_count"] >= threshold).sum())
    print(f"At least {threshold:>3} observations: {count:>3} players")

print("\n=== TARGET COUNT MISMATCHES ===")
mismatches = result.loc[
    ~result["target_count_matches"],
    ["nfl_id", "draft_year", "targets", "targets_recounted"],
]
if mismatches.empty:
    print("None")
else:
    print(mismatches.to_string(index=False))

print("\n=== LOWEST SEPARATION COUNTS ===")
cols = [
    "nfl_id",
    "draft_year",
    "targets",
    "targets_recounted",
    "separation_count",
    "mean_separation",
    "mean_separation_recomputed",
]
print(
    result.sort_values("separation_count")[cols]
    .head(15)
    .to_string(index=False)
)

print("\nSaved audit to:", output_path.resolve())

