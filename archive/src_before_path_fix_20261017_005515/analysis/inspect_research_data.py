
from pathlib import Path
import pandas as pd

ROOT = Path("nfl-big-data-bowl-2027")
if not ROOT.exists():
    ROOT = Path(".")

def read_csv(name, **kwargs):
    return pd.read_csv(ROOT / name, low_memory=False, **kwargs)

# 1. Player-level joins
combine = read_csv("combine_results.csv")
players = read_csv("players.csv")
career = read_csv("player_career_successes.csv")

combine_ids = set(combine["nfl_id"].dropna())
player_ids = set(players["nfl_id"].dropna())
career_ids = set(career["nfl_id"].dropna())

print("\n=== PLAYER ID OVERLAP ===")
print("Combine prospects:", len(combine_ids))
print("Present in players.csv:", len(combine_ids & player_ids))
print("Present in career outcomes:", len(combine_ids & career_ids))
print("Combine positions:")
print(combine["combine_position"].value_counts().to_string())
print("\nDraft years:")
print(combine["draft_year"].value_counts().sort_index().to_string())

# 2. Exact drill coverage and trajectory completeness
tracking = read_csv("combine_tracking.csv")

print("\n=== COMBINE TRACKING ===")
print("Rows:", len(tracking))
print("Unique players:", tracking["nfl_id"].nunique())
print("Unique events:", tracking["event_id"].nunique())
print("\nDrill types:")
print(tracking["drill_type"].value_counts().to_string())
print("\nMost frequent drill names:")
print(tracking["drill_name"].value_counts().head(40).to_string())
print("\nEntity types:")
print(tracking["entity_type"].value_counts().to_string())

# A trajectory is an event + player + drill attempt.
keys = ["nfl_id", "event_id", "drill_name", "attempt"]
print("\nUnique player-drill attempts:", tracking[keys].drop_duplicates().shape[0])

# 3. Traditional baseline coverage
print("\n=== TRADITIONAL TEST COVERAGE ===")
metrics = [
    "forty", "ten_yd_split", "three_cone",
    "short_shuttle", "vertical", "broad_jump"
]
for col in metrics:
    print(f"{col}: {combine[col].notna().sum()} / {len(combine)}")

# 4. Game-tracking player overlap.
# Read only identifiers to limit memory usage.
print("\n=== GAME TRACKING OVERLAP ===")
all_game_ids = set()

for year in [2023, 2024, 2025]:
    name = f"game_tracking_{year}.csv"
    ids = set()

    for chunk in pd.read_csv(
        ROOT / name,
        usecols=["nfl_id"],
        chunksize=500_000
    ):
        ids.update(chunk["nfl_id"].dropna().astype(int).unique())

    overlap = combine_ids & ids
    all_game_ids.update(ids)
    print(f"{year}: {len(ids)} unique players; "
          f"{len(overlap)} also appear in Combine results")

print("\nTotal distinct game-tracking players:", len(all_game_ids))
print("Combine players found in any game tracking:",
      len(combine_ids & all_game_ids))

# 5. Career outcome distribution among Combine prospects
matched = career[career["nfl_id"].isin(combine_ids)]
print("\n=== CAREER OUTCOMES ===")
print(matched[[
    "career_offensive_snaps",
    "career_defensive_snaps",
    "career_games_active",
    "career_games_started",
    "ap_all_pro_1st_team",
    "pro_bowl_original_ballot"
]].describe().round(2).to_string())

print("\nINSPECTION COMPLETE")
