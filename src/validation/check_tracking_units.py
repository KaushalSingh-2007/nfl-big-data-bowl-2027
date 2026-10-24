
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA_DIR / "combine_tracking.csv")
df = df[
    (df["entity_type"] == "PLAYER")
    & df["drill_name"].isin([
        "COMEBACK_ROUTE_RIGHT",
        "DAGGER_ROUTE_LEFT",
    ])
].copy()

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time", "x", "y", "s", "a"])
df = df.sort_values(
    ["nfl_id", "event_id", "drill_name", "attempt", "time"]
)

keys = ["nfl_id", "event_id", "drill_name", "attempt"]
results = []

for key, g in df.groupby(keys, sort=False):
    g = g.drop_duplicates("time").copy()
    if len(g) < 10:
        continue

    t = g["time"].astype("int64").to_numpy() / 1e9
    dt = np.diff(t)

    if np.any(dt <= 0):
        continue

    x = g["x"].to_numpy(float)
    y = g["y"].to_numpy(float)
    s = g["s"].to_numpy(float)
    a = g["a"].to_numpy(float)

    # Distance-based speed, in coordinate units per second.
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    speed_xy = np.hypot(vx, vy)

    # Derivative of the provided speed, in speed units per second.
    ds_dt = np.gradient(s, t)

    # Ignore the endpoints, where derivative estimates are less stable.
    sl = slice(2, -2)

    results.append({
        "nfl_id": key[0],
        "drill_name": key[2],
        "n_points": len(g),
        "median_dt": np.median(dt),
        "median_supplied_speed": np.median(s[sl]),
        "median_xy_speed": np.median(speed_xy[sl]),
        "median_supplied_accel": np.median(a[sl]),
        "median_speed_derivative": np.median(ds_dt[sl]),
        "corr_speed": np.corrcoef(s[sl], speed_xy[sl])[0, 1],
        "corr_accel": np.corrcoef(a[sl], ds_dt[sl])[0, 1],
    })

out = pd.DataFrame(results)

print("=== SAMPLING INTERVAL ===")
print(out["median_dt"].describe().round(4))

print("\n=== SPEED COMPARISON ===")
print(out[[
    "median_supplied_speed",
    "median_xy_speed",
    "corr_speed"
]].describe().round(3))

print("\n=== ACCELERATION COMPARISON ===")
print(out[[
    "median_supplied_accel",
    "median_speed_derivative",
    "corr_accel"
]].describe().round(3))

print("\n=== ROUTE-SPECIFIC CORRELATIONS ===")
print(
    out.groupby("drill_name")[
        ["corr_speed", "corr_accel"]
    ].median().round(3)
)

out.to_csv(
    OUT / "tracking_unit_checks.csv",
    index=False
)

print("\nSaved tracking_unit_checks.csv")


