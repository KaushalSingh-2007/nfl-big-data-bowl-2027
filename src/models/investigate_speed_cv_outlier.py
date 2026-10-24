from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

OUT = (
    PROJECT_ROOT
    / "research_outputs"
    / "model_comparison"
    / "speed_cv_signal"
    / "outlier_diagnostic"
)

OUT.mkdir(parents=True, exist_ok=True)

NFL_ID = 56248
DRAFT_YEAR = 2023
DRILL_NAME = "DAGGER_ROUTE_LEFT"

tracking_path = DATA_DIR / "combine_tracking.csv"

print("Loading combine tracking...")

tracking = pd.read_csv(
    tracking_path,
    usecols=[
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
        "dir"
    ]
)

player = tracking[
    (tracking["nfl_id"] == NFL_ID) &
    (tracking["draft_year"] == DRAFT_YEAR) &
    (tracking["drill_name"] == DRILL_NAME)
].copy()

if player.empty:
    raise ValueError("No rows found for requested player/route.")

print(f"Rows found: {len(player)}")

player["time_datetime"] = pd.to_datetime(
    player["time"],
    errors="coerce"
)

if player["time_datetime"].isna().any():
    bad = int(player["time_datetime"].isna().sum())
    raise ValueError(
        f"{bad} timestamps could not be parsed."
    )

player = (
    player
    .sort_values("time_datetime")
    .reset_index(drop=True)
)

player["elapsed_time"] = (
    player["time_datetime"]
    - player["time_datetime"].iloc[0]
).dt.total_seconds()

player["speed_change"] = player["s"].diff()

player["speed_change_abs"] = (
    player["speed_change"].abs()
)

player["dt"] = player["elapsed_time"].diff()

player["dx"] = player["x"].diff()
player["dy"] = player["y"].diff()

player["frame_distance"] = np.sqrt(
    player["dx"] ** 2 +
    player["dy"] ** 2
)

player["derived_speed"] = (
    player["frame_distance"] /
    player["dt"]
)

speed = player["s"].astype(float)

population_cv = (
    speed.std(ddof=0) /
    speed.mean()
)

sample_cv = (
    speed.std(ddof=1) /
    speed.mean()
)

print()
print("========== OUTLIER DIAGNOSTIC ==========")
print(f"NFL ID: {NFL_ID}")
print(f"Draft year: {DRAFT_YEAR}")
print(f"Route: {DRILL_NAME}")
print(f"Rows: {len(player)}")
print(
    f"Duration: "
    f"{player['elapsed_time'].iloc[-1]:.3f} s"
)
print(f"Mean speed: {speed.mean():.6f}")
print(
    "Population speed std: "
    f"{speed.std(ddof=0):.6f}"
)
print(
    "Sample speed std: "
    f"{speed.std(ddof=1):.6f}"
)
print(f"Population CV: {population_cv:.6f}")
print(f"Sample CV: {sample_cv:.6f}")
print(f"Min speed: {speed.min():.6f}")
print(f"Max speed: {speed.max():.6f}")

print()
print("Speed quantiles:")

print(
    speed.quantile(
        [
            0,
            .01,
            .05,
            .10,
            .25,
            .50,
            .75,
            .90,
            .95,
            .99,
            1.0
        ]
    )
)

print()
print("Largest frame-to-frame speed changes:")

largest_changes = (
    player[
        [
            "time_datetime",
            "elapsed_time",
            "s",
            "speed_change",
            "speed_change_abs",
            "a",
            "dt"
        ]
    ]
    .dropna(
        subset=["speed_change_abs"]
    )
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

print()
print(
    "Largest frame-to-frame positional distances:"
)

largest_distances = (
    player[
        [
            "time_datetime",
            "elapsed_time",
            "x",
            "y",
            "dx",
            "dy",
            "frame_distance",
            "dt",
            "derived_speed",
            "s"
        ]
    ]
    .dropna(
        subset=["frame_distance"]
    )
    .sort_values(
        "frame_distance",
        ascending=False
    )
    .head(15)
)

print(
    largest_distances.to_string(
        index=False
    )
)

print()
print(
    "Speed supplied by tracking vs "
    "speed derived from x/y:"
)

speed_compare = (
    player[
        [
            "elapsed_time",
            "s",
            "derived_speed"
        ]
    ]
    .dropna()
)

print(
    speed_compare[
        ["s", "derived_speed"]
    ].describe()
)

if len(speed_compare) >= 3:

    corr = speed_compare["s"].corr(
        speed_compare["derived_speed"]
    )

    mae = (
        speed_compare["s"] -
        speed_compare["derived_speed"]
    ).abs().mean()

    print()
    print(
        f"Correlation: {corr:.6f}"
    )

    print(
        f"Mean absolute difference: {mae:.6f}"
    )

else:

    corr = np.nan
    mae = np.nan

raw_output = (
    OUT /
    f"outlier_{NFL_ID}_diagnostic.csv"
)

player.to_csv(
    raw_output,
    index=False
)

changes_output = (
    OUT /
    f"outlier_{NFL_ID}_largest_speed_changes.csv"
)

largest_changes.to_csv(
    changes_output,
    index=False
)

summary = pd.DataFrame(
    [
        {
            "nfl_id": NFL_ID,
            "draft_year": DRAFT_YEAR,
            "drill_name": DRILL_NAME,
            "n_rows": len(player),
            "duration_s": (
                player["elapsed_time"].iloc[-1]
            ),
            "mean_speed": speed.mean(),
            "speed_std_population": (
                speed.std(ddof=0)
            ),
            "speed_std_sample": (
                speed.std(ddof=1)
            ),
            "speed_cv_population": (
                population_cv
            ),
            "speed_cv_sample": (
                sample_cv
            ),
            "min_speed": speed.min(),
            "max_speed": speed.max(),
            "speed_xy_correlation": corr,
            "speed_xy_mae": mae
        }
    ]
)

summary_output = (
    OUT /
    f"outlier_{NFL_ID}_summary.csv"
)

summary.to_csv(
    summary_output,
    index=False
)

plt.figure(figsize=(11, 5))

plt.plot(
    player["elapsed_time"],
    player["s"],
    linewidth=1.5
)

plt.xlabel("Elapsed time (s)")
plt.ylabel("Tracking speed")

plt.title(
    f"Speed trajectory — "
    f"NFL ID {NFL_ID}, {DRILL_NAME}"
)

plt.grid(alpha=0.25)
plt.tight_layout()

speed_plot = (
    OUT /
    f"outlier_{NFL_ID}_speed_trajectory.png"
)

plt.savefig(
    speed_plot,
    dpi=180
)

plt.close()

plt.figure(figsize=(11, 5))

plt.plot(
    player["elapsed_time"],
    player["speed_change"],
    linewidth=1.2
)

plt.axhline(
    0,
    linewidth=0.8
)

plt.xlabel("Elapsed time (s)")
plt.ylabel("Frame-to-frame speed change")

plt.title(
    f"Frame-to-frame speed changes — "
    f"NFL ID {NFL_ID}"
)

plt.grid(alpha=0.25)
plt.tight_layout()

change_plot = (
    OUT /
    f"outlier_{NFL_ID}_speed_changes.png"
)

plt.savefig(
    change_plot,
    dpi=180
)

plt.close()

print()
print("Saved:")
print(raw_output)
print(changes_output)
print(summary_output)
print(speed_plot)
print(change_plot)
