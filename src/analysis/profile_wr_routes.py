
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
# Load the relatively small prospect and outcome files.
combine = pd.read_csv(DATA_DIR / "combine_results.csv")
players = pd.read_csv(DATA_DIR / "players.csv")
career = pd.read_csv(DATA_DIR / "player_career_successes.csv")

# Read Combine tracking and keep only actual players.
ct = pd.read_csv(DATA_DIR / "combine_tracking.csv")
ct = ct[ct["entity_type"].eq("PLAYER")].copy()

wr_ids = set(
    combine.loc[combine["combine_position"].eq("WR"), "nfl_id"]
)

wr = ct[ct["nfl_id"].isin(wr_ids)].copy()

print("\n=== WR COMBINE TRACKING ===")
print("Unique WR prospects:", len(wr_ids))
print("WR prospects with tracking:", wr["nfl_id"].nunique())
print("Tracking rows:", len(wr))

# Routes and attempts available to WRs.
route_rows = wr[
    wr["drill_name"].str.contains("ROUTE", case=False, na=False)
].copy()

route_summary = (
    route_rows.groupby("drill_name")
    .agg(
        rows=("nfl_id", "size"),
        players=("nfl_id", "nunique"),
        attempts=("event_id", "nunique"),
    )
    .sort_values("players", ascending=False)
)

print("\n=== ROUTE COVERAGE ===")
print(route_summary.to_string())

# Check trajectory lengths and tracking completeness.
keys = ["nfl_id", "event_id", "drill_name", "attempt"]
attempts = (
    route_rows.groupby(keys)
    .agg(
        points=("time", "size"),
        unique_times=("time", "nunique"),
        speed_missing=("s", lambda x: x.isna().mean()),
        accel_missing=("a", lambda x: x.isna().mean()),
        x_missing=("x", lambda x: x.isna().mean()),
        y_missing=("y", lambda x: x.isna().mean()),
    )
    .reset_index()
)

print("\n=== ROUTE ATTEMPT LENGTHS ===")
print(attempts["points"].describe().round(2).to_string())
print("\nShortest 15 attempts:")
print(attempts.nsmallest(15, "points").to_string(index=False))

# Position and separation coverage in player_play.
pp = pd.read_csv(DATA_DIR / "player_play.csv", low_memory=False)

wr_game = pp[pp["nfl_id"].isin(wr_ids)].copy()

print("\n=== WR GAME OUTCOME COVERAGE ===")
print("Rows for Combine WRs:", len(wr_game))
print("Distinct WRs in player_play:", wr_game["nfl_id"].nunique())

if "lined_up_position" in wr_game:
    print("\nPositions:")
    print(wr_game["lined_up_position"].value_counts().head(20))

for col in [
    "target", "route_ran", "separation_at_pass_forward",
    "rec_yards", "yards_after_catch"
]:
    if col in wr_game:
        print(
            f"{col}: nonmissing={wr_game[col].notna().sum()}, "
            f"unique players={wr_game.loc[wr_game[col].notna(), 'nfl_id'].nunique()}"
        )

# Summarize meaningful target-level outcomes.
if "target" in wr_game:
    targeted = wr_game[wr_game["target"].eq(True)]
    print("\nTargeted rows:", len(targeted))
    print("WRs targeted:", targeted["nfl_id"].nunique())

    for col in ["separation_at_pass_forward", "rec_yards",
                "yards_after_catch"]:
        if col in targeted:
            print(
                f"Targeted {col}: "
                f"{targeted[col].notna().sum()} nonmissing observations"
            )

# Draft year and position metadata for future stratification.
print("\n=== WR DRAFT CLASSES ===")
print(
    combine.loc[combine["combine_position"].eq("WR")]
    .groupby("draft_year")["nfl_id"]
    .nunique()
    .to_string()
)

print("\n=== PROFILE COMPLETE ===")

