from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
OUT = PROJECT_ROOT / "research_outputs"

# Load only required columns.
plays = pd.read_csv(
    DATA_DIR / "player_play.csv",
    usecols=[
        "game_id",
        "nfl_id",
        "lined_up_position",
        "target",
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

print("Raw play rows:", len(plays))
print("Duplicate game IDs:", games["game_id"].duplicated().sum())
print("Duplicate player IDs:", players["nfl_id"].duplicated().sum())

# Join the data.
plays = plays.merge(
    games, on="game_id", how="left", validate="many_to_one"
)
plays = plays.merge(
    players, on="nfl_id", how="left", validate="many_to_one"
)

# Apply the documented time filters.
plays = plays[
    (plays["season_type"] == "REG")
    & (plays["season"] >= plays["draft_year"])
].copy()

plays["target_numeric"] = pd.to_numeric(
    plays["target"], errors="coerce"
)

print("\n=== POSITION COUNTS ===")
print(
    plays["lined_up_position"]
    .value_counts(dropna=False)
    .head(15)
    .to_string()
)

print("\n=== TARGET COUNTS UNDER DIFFERENT POSITION FILTERS ===")

filters = {
    "WR or SLOT_WR lineup": plays["lined_up_position"].isin(
        ["WR", "SLOT_WR"]
    ),
    "NFL position WR": plays["nfl_position"].eq("WR"),
    "Either definition": (
        plays["lined_up_position"].isin(["WR", "SLOT_WR"])
        | plays["nfl_position"].eq("WR")
    ),
}

for name, mask in filters.items():
    subset = plays.loc[mask]
    targets = subset[subset["target_numeric"].eq(1)]
    counts = targets.groupby("nfl_id").size()

    print(f"\n{name}")
    print("Qualifying play rows:", len(subset))
    print("Target rows:", len(targets))
    print("Players with targets:", len(counts))
    print("Total targets:", int(counts.sum()))

# Compare the direct recount with both saved sources.
dataset = pd.read_csv(OUT / "wr_research_dataset.csv")
coverage = pd.read_csv(OUT / "wr_outcome_coverage.csv")

targets = plays[
    plays["lined_up_position"].isin(["WR", "SLOT_WR"])
    & plays["target_numeric"].eq(1)
]

recount = (
    targets.groupby("nfl_id")
    .size()
    .rename("recounted_targets")
    .reset_index()
)

comparison = dataset[
    ["nfl_id", "draft_year", "targets"]
].merge(
    coverage[["nfl_id", "targets_from_plays"]],
    on="nfl_id",
    how="left",
    validate="one_to_one",
).merge(
    recount,
    on="nfl_id",
    how="left",
    validate="one_to_one",
)

comparison["recounted_targets"] = (
    comparison["recounted_targets"].fillna(0).astype(int)
)

comparison["dataset_minus_recount"] = (
    pd.to_numeric(comparison["targets"], errors="coerce").fillna(0)
    - comparison["recounted_targets"]
)

comparison["coverage_minus_recount"] = (
    pd.to_numeric(comparison["targets_from_plays"], errors="coerce")
    - comparison["recounted_targets"]
)

print("\n=== SAVED COUNTS VS DIRECT RECOUNT ===")
print(
    "Dataset mismatches:",
    int((comparison["dataset_minus_recount"] != 0).sum()),
    "/",
    len(comparison),
)
print(
    "Coverage mismatches:",
    int((comparison["coverage_minus_recount"] != 0).sum()),
    "/",
    len(comparison),
)

print("\nLargest discrepancies:")
comparison["absolute_difference"] = (
    comparison["dataset_minus_recount"].abs()
)
print(
    comparison.sort_values(
        "absolute_difference", ascending=False
    ).head(20).to_string(index=False)
)

output = OUT / "target_count_discrepancy_audit.csv"
comparison.to_csv(output, index=False)

print("\nSaved:", output.resolve())


