from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "research_outputs"
DIAG_OUT = (
    OUT_DIR
    / "model_comparison"
    / "speed_cv_signal"
    / "outlier_diagnostic"
)

DIAG_OUT.mkdir(parents=True, exist_ok=True)

RESEARCH_DATA = OUT_DIR / "wr_research_dataset.csv"
TRACKING_DATA = DATA_DIR / "combine_tracking.csv"

NFL_ID = 56248

print("=" * 80)
print("SPEED CV OUTLIER INVESTIGATION - FINAL")
print("=" * 80)

# ================================================================
# 1. Load research dataset
# ================================================================

df = pd.read_csv(RESEARCH_DATA)

required = [
    "nfl_id",
    "draft_year",
    "speed_cv_DAGGER_ROUTE_LEFT",
    "mean_separation",
    "forty",
    "vertical",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

common = df[required].dropna().copy()

outlier_row = common.loc[
    common["nfl_id"] == NFL_ID
]

if outlier_row.empty:
    raise ValueError(
        f"nfl_id {NFL_ID} not found in research dataset."
    )

outlier_row = outlier_row.iloc[0]

research_speed_cv = float(
    outlier_row[
        "speed_cv_DAGGER_ROUTE_LEFT"
    ]
)

print("\n--- EXTREME PLAYER ---")

print(
    f"nfl_id:          {NFL_ID}"
)

print(
    f"draft_year:      "
    f"{int(outlier_row['draft_year'])}"
)

print(
    f"research speed CV: "
    f"{research_speed_cv:.6f}"
)

print(
    f"40 yard:         "
    f"{outlier_row['forty']}"
)

print(
    f"vertical:        "
    f"{outlier_row['vertical']}"
)

print(
    f"mean separation: "
    f"{outlier_row['mean_separation']:.6f}"
)

# ================================================================
# 2. Load raw tracking
# ================================================================

print("\n--- LOADING RAW TRACKING ---")

cols = [
    "draft_year",
    "event_id",
    "nfl_id",
    "entity_type",
    "time",
    "drill_type",
    "drill_name",
    "attempt",
    "x",
    "y",
    "s",
    "a",
    "dis",
    "dir",
]

tracking = pd.read_csv(
    TRACKING_DATA,
    usecols=cols,
    low_memory=False
)

print(
    f"Raw rows loaded: "
    f"{len(tracking):,}"
)

# ================================================================
# 3. Parse time correctly
# ================================================================

tracking["time_datetime"] = pd.to_datetime(
    tracking["time"],
    errors="coerce"
)

print(
    f"Time values successfully parsed: "
    f"{tracking['time_datetime'].notna().sum():,} / "
    f"{len(tracking):,}"
)

# ================================================================
# 4. Find player's Dagger route
# ================================================================

player = tracking[
    (tracking["nfl_id"] == NFL_ID)
    &
    (
        tracking["drill_name"]
        .astype(str)
        .str.upper()
        .str.contains(
            "DAGGER_ROUTE_LEFT",
            na=False
        )
    )
].copy()

if player.empty:
    raise ValueError(
        "No Dagger route found."
    )

player = player.sort_values(
    "time_datetime"
).copy()

# elapsed seconds from beginning of route
player["elapsed_time"] = (
    player["time_datetime"]
    -
    player["time_datetime"].iloc[0]
).dt.total_seconds()

print("\n--- PLAYER ROUTE ---")

print(
    f"Rows:       {len(player)}"
)

print(
    f"Attempts:   "
    f"{player['attempt'].unique()}"
)

print(
    f"Event IDs:  "
    f"{player['event_id'].unique()}"
)

print(
    f"Start time: "
    f"{player['time_datetime'].iloc[0]}"
)

print(
    f"End time:   "
    f"{player['time_datetime'].iloc[-1]}"
)

print(
    f"Duration:   "
    f"{player['elapsed_time'].iloc[-1]:.3f} seconds"
)

# ================================================================
# 5. Calculate speed CV
# ================================================================

speed = player["s"].astype(float)

mean_speed = speed.mean()
std_speed_population = speed.std(ddof=0)
std_speed_sample = speed.std(ddof=1)

calculated_cv_population = (
    std_speed_population / mean_speed
)

calculated_cv_sample = (
    std_speed_sample / mean_speed
)

print("\n--- SPEED CV CALCULATION ---")

print(
    f"Mean speed:              "
    f"{mean_speed:.6f}"
)

print(
    f"Population std:           "
    f"{std_speed_population:.6f}"
)

print(
    f"Sample std:               "
    f"{std_speed_sample:.6f}"
)

print(
    f"Population CV:            "
    f"{calculated_cv_population:.6f}"
)

print(
    f"Sample CV:                "
    f"{calculated_cv_sample:.6f}"
)

print(
    f"Research dataset CV:      "
    f"{research_speed_cv:.6f}"
)

print(
    f"Difference population:    "
    f"{abs(calculated_cv_population - research_speed_cv):.8f}"
)

print(
    f"Difference sample:        "
    f"{abs(calculated_cv_sample - research_speed_cv):.8f}"
)

# ================================================================
# 6. Full speed statistics
# ================================================================

print("\n--- SPEED STATISTICS ---")

print(
    speed.describe().to_string()
)

# ================================================================
# 7. Acceleration / speed change diagnostics
# ================================================================

player["speed_change"] = player["s"].diff()

player["speed_change_abs"] = (
    player["speed_change"].abs()
)

print(
    "\nLargest frame-to-frame speed changes:"
)

largest_changes = (
    player[
        [
            "elapsed_time",
            "s",
            "speed_change",
            "a",
        ]
    ]
    .sort_values(
        "speed_change_abs",
        ascending=False
    )
    .head(15)
)

print(
    largest_changes.to_string(
        index=False
    )
)

largest_changes.to_csv(
    DIAG_OUT
    / "largest_speed_changes.csv",
    index=False
)

# ================================================================
# 8. Route movement statistics
# ================================================================

dx = player["x"].diff()
dy = player["y"].diff()

step_distance = np.sqrt(
    dx ** 2 + dy ** 2
)

path_length = (
    step_distance.sum()
)

direct_distance = np.sqrt(
    (
        player["x"].iloc[-1]
        -
        player["x"].iloc[0]
    ) ** 2
    +
    (
        player["y"].iloc[-1]
        -
        player["y"].iloc[0]
    ) ** 2
)

straightness = (
    direct_distance / path_length
    if path_length != 0
    else np.nan
)

print("\n--- ROUTE GEOMETRY ---")

print(
    f"Path length:       "
    f"{path_length:.6f}"
)

print(
    f"Direct distance:   "
    f"{direct_distance:.6f}"
)

print(
    f"Straightness:      "
    f"{straightness:.6f}"
)

# ================================================================
# 9. Save complete raw trajectory
# ================================================================

raw_output = (
    DIAG_OUT
    / "outlier_56248_full_trajectory.csv"
)

player.to_csv(
    raw_output,
    index=False
)

print(
    f"\nRaw trajectory saved:"
)

print(raw_output)

# ================================================================
# 10. Speed profile by route percentage
# ================================================================

route_duration = (
    player["elapsed_time"].max()
)

if route_duration > 0:

    player["route_percent"] = (
        player["elapsed_time"]
        /
        route_duration
        *
        100
    )

else:

    player["route_percent"] = 0

player["route_bin"] = pd.cut(
    player["route_percent"],
    bins=[
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    ],
    include_lowest=True
)

profile = (
    player
    .groupby(
        "route_bin",
        observed=False
    )
    .agg(
        mean_speed=("s", "mean"),
        median_speed=("s", "median"),
        min_speed=("s", "min"),
        max_speed=("s", "max"),
        mean_acceleration=("a", "mean"),
    )
    .reset_index()
)

print("\n--- SPEED THROUGH ROUTE ---")

print(
    profile.to_string(index=False)
)

profile.to_csv(
    DIAG_OUT
    / "outlier_speed_profile.csv",
    index=False
)

# ================================================================
# 11. Plot speed vs time
# ================================================================

plt.figure(figsize=(11, 5))

plt.plot(
    player["elapsed_time"],
    player["s"],
    linewidth=1.5
)

plt.xlabel(
    "Elapsed time (seconds)"
)

plt.ylabel(
    "Speed"
)

plt.title(
    f"Dagger Route Speed Profile - nfl_id {NFL_ID}"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

speed_plot = (
    DIAG_OUT
    / "outlier_speed_vs_time.png"
)

plt.savefig(
    speed_plot,
    dpi=200
)

plt.close()

# ================================================================
# 12. Plot trajectory
# ================================================================

plt.figure(figsize=(7, 7))

plt.plot(
    player["x"],
    player["y"],
    linewidth=1.5
)

plt.scatter(
    player["x"].iloc[0],
    player["y"].iloc[0],
    s=70,
    label="Start"
)

plt.scatter(
    player["x"].iloc[-1],
    player["y"].iloc[-1],
    s=70,
    label="End"
)

plt.xlabel("X")
plt.ylabel("Y")

plt.title(
    f"Dagger Route Trajectory - nfl_id {NFL_ID}"
)

plt.legend()
plt.grid(
    True,
    alpha=0.3
)

plt.axis("equal")
plt.tight_layout()

trajectory_plot = (
    DIAG_OUT
    / "outlier_trajectory_xy.png"
)

plt.savefig(
    trajectory_plot,
    dpi=200
)

plt.close()

# ================================================================
# 13. Compare against other players
# ================================================================

others = common[
    common["nfl_id"] != NFL_ID
]

print("\n--- OUTLIER VS OTHER PLAYERS ---")

print(
    f"Outlier speed CV: "
    f"{research_speed_cv:.6f}"
)

print(
    f"Other players mean CV: "
    f"{others['speed_cv_DAGGER_ROUTE_LEFT'].mean():.6f}"
)

print(
    f"Other players median CV: "
    f"{others['speed_cv_DAGGER_ROUTE_LEFT'].median():.6f}"
)

print(
    f"Other players 95th percentile: "
    f"{others['speed_cv_DAGGER_ROUTE_LEFT'].quantile(0.95):.6f}"
)

# ================================================================
# 14. Final conclusion flags
# ================================================================

print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

if (
    abs(
        calculated_cv_population
        -
        research_speed_cv
    )
    < 0.01
):
    print(
        "\n✓ Research speed CV is reproduced "
        "by raw tracking data."
    )
else:
    print(
        "\n⚠ Research speed CV does NOT closely "
        "match the direct raw calculation."
    )

if (
    player["s"].min() >= 0
    and
    player["s"].max() < 15
):
    print(
        "✓ Speed values are within a "
        "plausible tracking range."
    )
else:
    print(
        "⚠ Speed values contain potentially "
        "extreme observations."
    )

print(
    "\nImportant:"
)

print(
    "We are NOT removing this player."
)

print(
    "The purpose of this diagnostic is to "
    "determine whether the high CV is a "
    "legitimate route-level movement pattern "
    "or a data artifact."
)

print("\nOutput folder:")
print(DIAG_OUT)

print("\nFiles created:")

for file in sorted(DIAG_OUT.glob("*")):
    print(
        " - " + file.name
    )

print(
    "\nOUTLIER INVESTIGATION COMPLETE."
)
