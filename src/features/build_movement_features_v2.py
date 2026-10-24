from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
OUT = PROJECT_ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "combine_tracking.csv"
OUTPUT_FILE = OUT / "wr_attempt_features_v2.csv"

ROUTES = [
    "COMEBACK_ROUTE_RIGHT",
    "DAGGER_ROUTE_LEFT",
]

MIN_POINTS = 20
SMOOTHING_WINDOW = 5
MIN_MOVEMENT_SPEED = 1.0

# Main analysis window and sensitivity windows, in seconds.
MAIN_WINDOW = 0.3
SENSITIVITY_WINDOWS = [0.2, 0.3, 0.5]


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading Combine tracking data...")

df = pd.read_csv(INPUT_FILE)

df = df[
    (df["entity_type"] == "PLAYER")
    & df["drill_name"].isin(ROUTES)
].copy()

df["time"] = pd.to_datetime(df["time"], errors="coerce")

df = df.dropna(
    subset=["nfl_id", "event_id", "drill_name", "attempt",
            "time", "x", "y", "s"]
)

df = df.sort_values(
    ["nfl_id", "event_id", "drill_name", "attempt", "time"]
)

KEYS = ["nfl_id", "event_id", "drill_name", "attempt"]

print(f"Tracking rows: {len(df):,}")
print(f"Unique players: {df['nfl_id'].nunique():,}")


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def circular_difference(a, b):
    """Shortest signed angular difference, in radians."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def angle_change_series(heading, k):
    """
    Calculate absolute heading change between points k samples
    before and after each candidate.

    Returns radians. Boundary points remain NaN.
    """
    n = len(heading)
    delta = np.full(n, np.nan)

    for i in range(k, n - k):
        h_before = heading[i - k]
        h_after = heading[i + k]

        if np.isfinite(h_before) and np.isfinite(h_after):
            delta[i] = abs(
                circular_difference(h_after, h_before)
            )

    return delta


def find_candidate(t, heading, smooth_s, window_seconds):
    """
    Find the point with the largest heading change for a given
    comparison window.

    This is a candidate detector, not verified ground truth.
    """
    median_dt = float(np.median(np.diff(t)))

    if not np.isfinite(median_dt) or median_dt <= 0:
        return None, np.full(len(t), np.nan)

    k = max(2, round(window_seconds / median_dt))

    delta = angle_change_series(heading, k)

    indices = np.arange(len(t))

    valid = (
        np.isfinite(delta)
        & (indices >= k)
        & (indices < len(t) - k)
        & np.isfinite(smooth_s)
        & (smooth_s > MIN_MOVEMENT_SPEED)
    )

    candidates = np.flatnonzero(valid)

    if len(candidates) == 0:
        return None, delta

    idx = candidates[np.argmax(delta[candidates])]

    return int(idx), delta


def median_in_window(values, mask):
    """Return median for a window, or NaN if it has no observations."""
    selected = values[mask]

    if len(selected) == 0:
        return np.nan

    return float(np.median(selected))


# ============================================================
# 4. EXTRACT FEATURES FOR ONE ATTEMPT
# ============================================================

def extract(g):
    # Aggregate duplicate timestamps before computing derivatives.
    g = (
        g.groupby("time", as_index=False)
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            s=("s", "mean"),
            a=("a", "mean"),
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    if len(g) < MIN_POINTS:
        return None

    # Convert timestamps to elapsed seconds.
    t = g["time"].astype("int64").to_numpy(dtype=float) / 1e9
    t -= t[0]

    dt = np.diff(t)

    if (
        not np.all(np.isfinite(dt))
        or np.any(dt <= 0)
    ):
        return None

    x = g["x"].to_numpy(dtype=float)
    y = g["y"].to_numpy(dtype=float)
    speed = g["s"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Smooth supplied speed and calculate its time derivative.
    # --------------------------------------------------------

    smooth_s = (
        pd.Series(speed)
        .rolling(
            SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .median()
        .to_numpy()
    )

    ds_dt = np.gradient(smooth_s, t)

    # --------------------------------------------------------
    # Smooth coordinates and estimate heading.
    # --------------------------------------------------------

    xs = (
        pd.Series(x)
        .rolling(
            SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    ys = (
        pd.Series(y)
        .rolling(
            SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    dx_dt = np.gradient(xs, t)
    dy_dt = np.gradient(ys, t)

    estimated_speed = np.hypot(dx_dt, dy_dt)
    heading = np.arctan2(dy_dt, dx_dt)

    # Exclude unreliable heading estimates when movement is slow.
    heading[estimated_speed < 0.5] = np.nan

    # --------------------------------------------------------
    # Main candidate and sensitivity analysis.
    # --------------------------------------------------------

    candidate_indices = {}
    candidate_deltas = {}

    for window in SENSITIVITY_WINDOWS:
        idx, delta = find_candidate(
            t=t,
            heading=heading,
            smooth_s=smooth_s,
            window_seconds=window,
        )

        candidate_indices[window] = idx
        candidate_deltas[window] = delta

    cut = candidate_indices[MAIN_WINDOW]

    if cut is None:
        return None

    delta = candidate_deltas[MAIN_WINDOW]
    candidate_angle_deg = float(np.degrees(delta[cut]))

    # --------------------------------------------------------
    # Speed before and after the candidate.
    # --------------------------------------------------------

    before = (
        (t >= t[cut] - 0.5)
        & (t < t[cut])
    )

    after = (
        (t > t[cut])
        & (t <= t[cut] + 0.5)
    )

    if before.sum() < 3 or after.sum() < 3:
        speed_before = np.nan
        speed_after = np.nan
        speed_retention = np.nan
    else:
        speed_before = median_in_window(smooth_s, before)
        speed_after = median_in_window(smooth_s, after)

        speed_retention = (
            speed_after / speed_before
            if np.isfinite(speed_before) and speed_before > 0.5
            else np.nan
        )

    # --------------------------------------------------------
    # Deceleration proxy around the candidate.
    # Negative values mean supplied speed is decreasing.
    # --------------------------------------------------------

    braking_window = (
        (t >= t[cut] - 0.5)
        & (t <= t[cut] + 0.5)
    )

    local_ds_dt = ds_dt[braking_window]
    local_ds_dt = local_ds_dt[np.isfinite(local_ds_dt)]

    min_smoothed_dspeed_dt = (
        float(np.min(local_ds_dt))
        if len(local_ds_dt)
        else np.nan
    )

    # --------------------------------------------------------
    # Candidate stability across window sizes.
    # --------------------------------------------------------

    sensitivity_times = {}

    for window in SENSITIVITY_WINDOWS:
        idx = candidate_indices[window]

        sensitivity_times[window] = (
            float(t[idx]) if idx is not None else np.nan
        )

    t02 = sensitivity_times[0.2]
    t03 = sensitivity_times[0.3]
    t05 = sensitivity_times[0.5]

    valid_sensitivity_times = [
        value
        for value in [t02, t03, t05]
        if np.isfinite(value)
    ]

    sensitivity_time_range = (
        max(valid_sensitivity_times) - min(valid_sensitivity_times)
        if len(valid_sensitivity_times) >= 2
        else np.nan
    )

    return {
        "duration_s": float(t[-1]),
        "n_samples": int(len(t)),

        "peak_speed_supplied": float(np.max(speed)),
        "mean_speed_supplied": float(np.mean(speed)),

        "candidate_turn_deg": candidate_angle_deg,
        "turn_time_s": float(t[cut]),

        "speed_before_turn": speed_before,
        "speed_after_turn": speed_after,
        "speed_retention": speed_retention,

        "min_smoothed_dspeed_dt": min_smoothed_dspeed_dt,

        "candidate_time_02_s": t02,
        "candidate_time_03_s": t03,
        "candidate_time_05_s": t05,
        "candidate_time_range_s": sensitivity_time_range,

        "candidate_stable_05s": (
            bool(sensitivity_time_range <= 0.5)
            if np.isfinite(sensitivity_time_range)
            else False
        ),
    }


# ============================================================
# 5. PROCESS ALL ATTEMPTS
# ============================================================

rows = []

groups = df.groupby(KEYS, sort=False)

print(f"Processing {len(groups):,} attempts...")

for count, (key, group) in enumerate(groups, start=1):
    result = extract(group)

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

    if count % 500 == 0:
        print(f"Processed {count:,} attempts...")


# ============================================================
# 6. SAVE RESULTS
# ============================================================

attempts = pd.DataFrame(rows)

if attempts.empty:
    raise RuntimeError(
        "No valid attempts were extracted. Check the input data."
    )

attempts.to_csv(OUTPUT_FILE, index=False)

# Save a compact stability report separately.
stability_file = OUT / "wr_candidate_stability_report.csv"

stability_report = (
    attempts.groupby("drill_name")
    .agg(
        attempts=("nfl_id", "size"),
        players=("nfl_id", "nunique"),
        median_candidate_angle_deg=("candidate_turn_deg", "median"),
        median_candidate_time_s=("turn_time_s", "median"),
        median_time_range_s=("candidate_time_range_s", "median"),
        unstable_candidates=(
            "candidate_stable_05s",
            lambda values: int((~values).sum()),
        ),
    )
    .reset_index()
)

stability_report.to_csv(stability_file, index=False)


# ============================================================
# 7. PRINT DIAGNOSTICS
# ============================================================

print("\n" + "=" * 65)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 65)

print(f"Valid attempts: {len(attempts):,}")
print(f"Unique players: {attempts['nfl_id'].nunique():,}")

print("\nFeature distributions:")
print(
    attempts.select_dtypes(include="number")
    .describe()
    .round(3)
    .to_string()
)

print("\nRoute summaries:")
print(
    attempts.groupby("drill_name")
    .median(numeric_only=True)
    .round(3)
    .to_string()
)

print("\nCandidate stability by route:")
print(stability_report.round(3).to_string(index=False))

print("\nQuality checks:")
print(
    "Angles outside [0, 180]:",
    int(
        (
            (attempts["candidate_turn_deg"] < 0)
            | (attempts["candidate_turn_deg"] > 180)
        ).sum()
    ),
)

print(
    "Speed retention above 2:",
    int((attempts["speed_retention"] > 2).sum()),
)

print(
    "Candidates shifting more than 0.5 seconds:",
    int((attempts["candidate_time_range_s"] > 0.5).sum()),
)

print(
    "Missing acceleration derivatives:",
    int(attempts["min_smoothed_dspeed_dt"].isna().sum()),
)

print("\nSaved files:")
print(OUTPUT_FILE)
print(stability_file)

print(
    "\nImportant: candidate stability measures sensitivity to the "
    "heading window, not whether the detected point is the true route break."
)


