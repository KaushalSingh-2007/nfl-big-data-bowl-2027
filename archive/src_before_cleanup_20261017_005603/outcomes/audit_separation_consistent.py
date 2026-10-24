from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "research_outputs"

# Load only the necessary columns.
plays = pd.read_csv(
    DATA_DIR / "player_play.csv",
    usecols=[
        "game_id", "nfl_id", "target",
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

# Join season and player information.
plays = plays.merge(
    games, on="game_id", how="left", validate="many_to_one"
)
plays = plays.merge(
    players, on="nfl_id", how="left", validate="many_to_one"
)

# Match the NFL-position definition used for the original WR outcome dataset.
plays = plays[
    (plays["season_type"] == "REG")
    & (plays["season"] >= plays["draft_year"])
    & (plays["nfl_position"] == "WR")
].copy()

plays["target"] = pd.to_numeric(plays["target"], errors="coerce")
plays["separation_at_pass_forward"] = pd.to_numeric(
    plays["separation_at_pass_forward"], errors="coerce"
)

targets = plays[plays["target"] == 1]

audit = targets.groupby("nfl_id").agg(
    targets_recounted=("target", "size"),
    separation_count=("separation_at_pass_forward", "count"),
    mean_separation_recomputed=(
        "separation_at_pass_forward", "mean"
    ),
).reset_index()

dataset = pd.read_csv(OUT / "wr_research_dataset.csv")

result = dataset.merge(
    audit, on="nfl_id", how="left", validate="one_to_one"
)

result["targets_recounted"] = (
    result["targets_recounted"].fillna(0).astype(int)
)
result["separation_count"] = (
    result["separation_count"].fillna(0).astype(int)
)

result["target_count_matches"] = (
    result["targets"].fillna(0).astype(int)
    == result["targets_recounted"]
)

result.to_csv(
    OUT / "wr_separation_reliability_consistent.csv",
    index=False,
)

print("=== TARGET COUNT VALIDATION ===")
print("Matching target counts:",
      int(result["target_count_matches"].sum()), "/", len(result))

print("\n=== SEPARATION OBSERVATIONS ===")
print(result["separation_count"].describe().round(2).to_string())

for threshold in [1, 10, 20, 30, 50, 100]:
    n = int((result["separation_count"] >= threshold).sum())
    print(f"At least {threshold} observations: {n} players")

print("\n=== TARGET COUNT MISMATCHES ===")
mismatches = result.loc[
    ~result["target_count_matches"],
    ["nfl_id", "draft_year", "targets", "targets_recounted"],
]
print(mismatches.to_string(index=False) if len(mismatches) else "None")

print("\n=== TOP PLAYERS BY SEPARATION COUNT ===")
cols = [
    "nfl_id", "draft_year", "targets",
    "targets_recounted", "separation_count",
    "mean_separation", "mean_separation_recomputed",
]
print(
    result.sort_values("separation_count", ascending=False)[cols]
    .head(15).to_string(index=False)
)

print("\nSaved:",
      (OUT / "wr_separation_reliability_consistent.csv").resolve())

