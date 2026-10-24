
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "research_outputs"

attempts = pd.read_csv(OUT / "wr_attempt_features.csv")
features = pd.read_csv(OUT / "wr_combine_movement_features.csv")
outcomes = pd.read_csv(OUT / "wr_post_draft_outcomes.csv")

print("\n=== 1. ATTEMPT-LEVEL FEATURE DISTRIBUTIONS ===")
cols = [
    "duration_s",
    "peak_speed",
    "mean_speed",
    "max_braking",
    "max_accel",
    "candidate_turn_deg",
    "speed_before_turn",
    "speed_after_turn",
    "speed_retention",
]

print(attempts[cols].describe(
    percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
).round(3).to_string())

print("\n=== 2. ROUTE-SPECIFIC DISTRIBUTIONS ===")
print(
    attempts.groupby("drill_name")[cols]
    .median()
    .round(3)
    .to_string()
)

print("\n=== 3. POTENTIAL WARNING FLAGS ===")
checks = {
    "duration_under_2s": attempts["duration_s"] < 2,
    "duration_over_12s": attempts["duration_s"] > 12,
    "peak_speed_zero_or_negative": attempts["peak_speed"] <= 0,
    "turn_angle_over_180": attempts["candidate_turn_deg"] > 180,
    "retention_negative": attempts["speed_retention"] < 0,
    "retention_over_2": attempts["speed_retention"] > 2,
    "extreme_braking_below_minus_10": attempts["max_braking"] < -10,
    "extreme_acceleration_over_10": attempts["max_accel"] > 10,
}

for label, mask in checks.items():
    print(f"{label}: {int(mask.sum())} / {len(attempts)}")

print("\n=== 4. PLAYER FEATURE COMPLETENESS ===")
movement_cols = [
    c for c in features.columns
    if c not in ["nfl_id", "draft_year", "combine_position"]
]

print("Player rows:", len(features))
print("Players with any missing movement feature:",
      int(features[movement_cols].isna().any(axis=1).sum()))

print("\nMissing percentage by feature:")
print(
    (features[movement_cols].isna().mean() * 100)
    .sort_values(ascending=False)
    .round(1)
    .to_string()
)

print("\n=== 5. NFL OUTCOME RELIABILITY ===")
print(outcomes[[
    "nfl_id",
    "targeted_plays",
    "mean_separation",
    "mean_rec_yards",
    "observed_play_rows",
    "seasons_observed",
]].sort_values("targeted_plays").head(15).to_string(index=False))

print("\nTargeted-play count distribution:")
print(
    outcomes["targeted_plays"]
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    .round(2)
    .to_string()
)

# Illustrative reliability thresholds, not automatic exclusions.
for threshold in [10, 20, 30, 50]:
    count = (outcomes["targeted_plays"] >= threshold).sum()
    print(
        f"Players with at least {threshold} targets: "
        f"{count} / {len(outcomes)}"
    )

print("\n=== 6. PLAYER-LEVEL FEATURE/OUTCOME MATCH ===")
merged = features.merge(
    outcomes, on="nfl_id", how="left", validate="one_to_one"
)
print("Total WR prospects:", len(merged))
print("Prospects with target outcomes:",
      int(merged["targeted_plays"].notna().sum()))
print("Prospects with at least 20 targets:",
      int((merged["targeted_plays"] >= 20).sum()))

print("\nVALIDATION AUDIT COMPLETE")

