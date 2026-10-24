from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("nfl-big-data-bowl-2027")
if not ROOT.exists():
    ROOT = Path(".")

OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)

INPUT = ROOT / "combine_tracking.csv"
OUTPUT = OUT / "wr_whole_route_features.csv"

ROUTES = [
    "COMEBACK_ROUTE_RIGHT",
    "DAGGER_ROUTE_LEFT",
]

print("Loading tracking data...")

df = pd.read_csv(INPUT)

df = df[
    (df["entity_type"] == "PLAYER")
    & df["drill_name"].isin(ROUTES)
].copy()

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["nfl_id", "event_id", "time", "x", "y", "s"])

keys = ["nfl_id", "event_id", "drill_name", "attempt"]
df = df.sort_values(keys + ["time"])

def extract(g):
    # Aggregate duplicate timestamps.
    g = (
        g.groupby("time", as_index=False)
        .agg(x=("x", "mean"), y=("y", "mean"), s=("s", "mean"))
        .sort_values("time")
    )

    if len(g) < 20:
        return None

    t = g["time"].astype("int64").to_numpy() / 1e9
    t -= t[0]

    dt = np.diff(t)
    if np.any(dt <= 0):
        return None

    x = g["x"].to_numpy(float)
    y = g["y"].to_numpy(float)
    s = g["s"].to_numpy(float)

    # Smooth coordinates slightly before estimating path length.
    xs = pd.Series(x).rolling(
        3, center=True, min_periods=1
    ).mean().to_numpy()

    ys = pd.Series(y).rolling(
        3, center=True, min_periods=1
    ).mean().to_numpy()

    # Total distance travelled along the observed trajectory.
    step_distance = np.hypot(np.diff(xs), np.diff(ys))
    path_length = float(step_distance.sum())

    # Direct distance between the start and end positions.
    direct_distance = float(
        np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    )

    # Straightness: 1 means a straight path; lower means more winding.
    straightness = (
        direct_distance / path_length
        if path_length > 0
        else np.nan
    )

    # Robust speed summaries.
    smooth_s = pd.Series(s).rolling(
        5, center=True, min_periods=1
    ).median().to_numpy()

    # Split the route into early and late halves by elapsed time.
    midpoint = t[-1] / 2
    early = t <= midpoint
    late = t > midpoint

    early_speed = float(np.median(smooth_s[early]))
    late_speed = float(np.median(smooth_s[late]))

    return {
        "duration_s": float(t[-1]),
        "n_samples": int(len(g)),
        "path_length": path_length,
        "direct_distance": direct_distance,
        "straightness_ratio": straightness,
        "peak_speed": float(np.max(s)),
        "mean_speed": float(np.mean(s)),
        "median_speed": float(np.median(smooth_s)),
        "speed_std": float(np.std(smooth_s)),
        "speed_cv": (
            float(np.std(smooth_s) / np.mean(smooth_s))
            if np.mean(smooth_s) > 0
            else np.nan
        ),
        "early_median_speed": early_speed,
        "late_median_speed": late_speed,
        "late_early_speed_ratio": (
            late_speed / early_speed
            if early_speed > 0.5
            else np.nan
        ),
    }

rows = []

for key, g in df.groupby(keys, sort=False):
    result = extract(g)

    if result is None:
        continue

    nfl_id, event_id, drill_name, attempt = key

    rows.append({
        "nfl_id": int(nfl_id),
        "event_id": event_id,
        "drill_name": drill_name,
        "attempt": attempt,
        **result,
    })

attempts = pd.DataFrame(rows)

if attempts.empty:
    raise RuntimeError("No valid attempts found.")

attempts.to_csv(OUTPUT, index=False)

# Aggregate attempts to one row per player and route.
feature_columns = [
    "duration_s",
    "path_length",
    "direct_distance",
    "straightness_ratio",
    "peak_speed",
    "mean_speed",
    "median_speed",
    "speed_std",
    "speed_cv",
    "early_median_speed",
    "late_median_speed",
    "late_early_speed_ratio",
]

player_features = (
    attempts.groupby(["nfl_id", "drill_name"])[feature_columns]
    .median()
    .reset_index()
)

player_features.to_csv(
    OUT / "wr_player_route_features.csv",
    index=False,
)

print("\nAttempt-level results")
print("Valid attempts:", len(attempts))
print("Players:", attempts["nfl_id"].nunique())
print("\nAttempts by route:")
print(attempts.groupby("drill_name").size().to_string())

print("\nPlayer-route feature rows:", len(player_features))
print("\nFeature summary:")
print(attempts[feature_columns].describe().round(3).to_string())

print("\nSaved:")
print(OUTPUT)
print(OUT / "wr_player_route_features.csv")
