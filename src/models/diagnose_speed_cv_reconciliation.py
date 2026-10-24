import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"C:\Users\singh\Downloads\nfl-big-data-bowl-2027\nfl-big-data-bowl-2027")

original_path = ROOT / "research_outputs" / "wr_whole_route_features.csv"
mech_path = ROOT / "research_outputs" / "receiving_efficiency_mechanism" / "dagger_speed_mechanism_attempts.csv"

print("=" * 80)
print("ATTEMPT-LEVEL SPEED-CV RECONCILIATION")
print("=" * 80)

orig = pd.read_csv(original_path)
mech = pd.read_csv(mech_path)

orig = orig[orig["drill_name"] == "DAGGER_ROUTE_LEFT"].copy()

print(f"\nOriginal Dagger attempts: {len(orig)}")
print(f"Mechanism Dagger attempts: {len(mech)}")

print("\nOriginal columns:")
print(orig.columns.tolist())

print("\nMechanism columns:")
print(mech.columns.tolist())

# The mechanism file is already Dagger-only, so drill_name is unnecessary.
keys = ["nfl_id", "event_id", "attempt"]

missing_orig = [c for c in keys if c not in orig.columns]
missing_mech = [c for c in keys if c not in mech.columns]

if missing_orig or missing_mech:
    print("\nERROR: Required merge columns are missing.")
    print("Missing from original:", missing_orig)
    print("Missing from mechanism:", missing_mech)
    raise SystemExit(1)

merged = orig.merge(
    mech,
    on=keys,
    how="outer",
    suffixes=("_orig", "_mech"),
    indicator=True
)

print("\nMerge status:")
print(merged["_merge"].value_counts().to_string())

common = merged[merged["_merge"] == "both"].copy()

print(f"\nCommon attempts: {len(common)}")

compare_cols = [
    "speed_cv",
    "speed_std",
    "mean_speed",
    "median_speed",
    "peak_speed",
    "early_median_speed",
    "late_median_speed",
    "late_early_speed_ratio",
]

print("\n" + "=" * 80)
print("ATTEMPT-LEVEL DIFFERENCES")
print("=" * 80)

for col in compare_cols:
    left = f"{col}_orig"
    right = f"{col}_mech"

    if left not in common.columns or right not in common.columns:
        print(f"{col:30s} NOT AVAILABLE")
        continue

    a = pd.to_numeric(common[left], errors="coerce")
    b = pd.to_numeric(common[right], errors="coerce")

    diff = (a - b).abs()

    print(
        f"{col:30s} "
        f"mean_abs={diff.mean():.8f}  "
        f"median_abs={diff.median():.8f}  "
        f"max_abs={diff.max():.8f}"
    )

print("\n" + "=" * 80)
print("TOP SPEED-CV ATTEMPT DISCREPANCIES")
print("=" * 80)

common["speed_cv_abs_diff"] = (
    pd.to_numeric(common["speed_cv_orig"], errors="coerce")
    - pd.to_numeric(common["speed_cv_mech"], errors="coerce")
).abs()

cols = [
    "nfl_id",
    "event_id",
    "attempt",
    "speed_cv_orig",
    "speed_cv_mech",
    "speed_cv_abs_diff",
    "speed_std_orig",
    "speed_std_mech",
    "mean_speed_orig",
    "mean_speed_mech",
]

print(
    common.sort_values("speed_cv_abs_diff", ascending=False)
    .head(15)[cols]
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("PLAYER-LEVEL AGGREGATION CHECK")
print("=" * 80)

orig_player = (
    orig.groupby(["nfl_id", "drill_name"])["speed_cv"]
    .median()
    .reset_index(name="speed_cv_orig_player")
)

mech_player = (
    mech.groupby(["nfl_id"])["speed_cv"]
    .median()
    .reset_index(name="speed_cv_mech_player")
)

pm = orig_player.merge(
    mech_player,
    on=["nfl_id"],
    how="inner"
)

pm["abs_diff"] = (
    pm["speed_cv_orig_player"] - pm["speed_cv_mech_player"]
).abs()

print(f"Players compared: {len(pm)}")
print(f"Mean abs difference: {pm['abs_diff'].mean():.8f}")
print(f"Median abs difference: {pm['abs_diff'].median():.8f}")
print(f"Maximum abs difference: {pm['abs_diff'].max():.8f}")

print("\nTop player discrepancies:")
print(
    pm.sort_values("abs_diff", ascending=False)
    .head(15)
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

attempt_mean = common["speed_cv_abs_diff"].mean()

player_mean = pm["abs_diff"].mean()

print(f"Attempt-level mean absolute speed-CV difference: {attempt_mean:.8f}")
print(f"Player-level mean absolute speed-CV difference:  {player_mean:.8f}")

if attempt_mean < 1e-6:
    print("\n=> Attempt-level calculation matches.")
    print("=> Remaining difference is almost certainly player-level aggregation.")
elif attempt_mean < 1e-3:
    print("\n=> Attempt-level calculation is extremely close.")
    print("=> Small numerical/preprocessing difference remains.")
else:
    print("\n=> The discrepancy already exists at the attempt level.")
    print("=> We need to inspect the exact speed preprocessing next.")

print("\nDONE — NOTHING WAS MODIFIED")
