
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
OUT = PROJECT_ROOT / "research_outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 1. Load Combine tracking and select two WR route families
# ---------------------------------------------------------
tracking = pd.read_csv(DATA_DIR / "combine_tracking.csv")
tracking = tracking[tracking["entity_type"].eq("PLAYER")].copy()

selected_routes = [
    "COMEBACK_ROUTE_RIGHT",
    "DAGGER_ROUTE_LEFT",
]

tracking = tracking[
    tracking["drill_name"].isin(selected_routes)
].copy()

tracking["time"] = pd.to_datetime(tracking["time"], errors="coerce")
tracking = tracking.dropna(
    subset=["nfl_id", "event_id", "time", "x", "y"]
)

keys = ["nfl_id", "event_id", "drill_name", "attempt"]
tracking = tracking.sort_values(keys + ["time"])

# ---------------------------------------------------------
# 2. Extract attempt-level movement features
# ---------------------------------------------------------
def extract_features(group):
    group = group.sort_values("time").copy()

    # Collapse duplicate timestamps if present.
    group = group.groupby("time", as_index=False).agg(
        x=("x", "mean"),
        y=("y", "mean"),
        supplied_speed=("s", "mean"),
        supplied_accel=("a", "mean"),
    )

    if len(group) < 15:
        return None

    t = (
        group["time"].astype("int64").to_numpy() / 1e9
    )
    t = t - t[0]

    # Exclude irregular/duplicate time sequences.
    dt = np.diff(t)
    if np.any(dt <= 0):
        return None

    x = group["x"].to_numpy(dtype=float)
    y = group["y"].to_numpy(dtype=float)

    # Light smoothing before differentiating positions.
    # Three samples correspond to roughly 0.3 seconds.
    xs = pd.Series(x).rolling(
        3, center=True, min_periods=1
    ).mean().to_numpy()

    ys = pd.Series(y).rolling(
        3, center=True, min_periods=1
    ).mean().to_numpy()

    vx = np.gradient(xs, t)
    vy = np.gradient(ys, t)
    speed = np.hypot(vx, vy)

    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)

    # Signed tangential acceleration:
    # negative values indicate slowing along the direction of travel.
    tangential_accel = np.gradient(speed, t)

    # Change in movement heading over a 0.5-second window.
    # Angle wrapping avoids treating 359 -> 1 degrees as a 358-degree turn.
    heading = np.unwrap(np.arctan2(vy, vx))
    window = max(2, round(0.5 / np.median(dt)))

    angle_change = np.full(len(t), np.nan)
    for i in range(window, len(t) - window):
        angle_change[i] = abs(
            heading[i + window] - heading[i - window]
        )

    valid = np.where(np.isfinite(angle_change))[0]
    if len(valid) == 0:
        return None

    # Candidate turning point: largest heading change over the window.
    cut_idx = valid[np.argmax(angle_change[valid])]

    # Compare speeds in windows before and after the candidate cut.
    pre = (t >= t[cut_idx] - 0.5) & (t < t[cut_idx])
    post = (t > t[cut_idx]) & (t <= t[cut_idx] + 0.5)

    if pre.sum() < 2 or post.sum() < 2:
        return None

    speed_before = float(np.median(speed[pre]))
    speed_after = float(np.median(speed[post]))

    return {
        "n_points": len(t),
        "duration_s": float(t[-1]),
        "peak_speed": float(np.max(speed)),
        "mean_speed": float(np.mean(speed)),
        "max_braking": float(np.min(tangential_accel)),
        "max_accel": float(np.max(tangential_accel)),
        "candidate_turn_deg": float(
            np.degrees(angle_change[cut_idx])
        ),
        "speed_before_turn": speed_before,
        "speed_after_turn": speed_after,
        "speed_retention": (
            speed_after / speed_before
            if speed_before > 0.1 else np.nan
        ),
        "turn_time_s": float(t[cut_idx]),
    }


records = []

for key, group in tracking.groupby(keys, sort=False):
    features = extract_features(group)
    if features is None:
        continue

    nfl_id, event_id, drill_name, attempt = key
    records.append({
        "nfl_id": int(nfl_id),
        "event_id": event_id,
        "drill_name": drill_name,
        "attempt": attempt,
        **features,
    })

attempt_features = pd.DataFrame(records)

if attempt_features.empty:
    raise RuntimeError(
        "No valid trajectories extracted. Check tracking columns."
    )

attempt_features.to_csv(
    OUT / "wr_attempt_features.csv", index=False
)

print("\n=== ATTEMPT FEATURE EXTRACTION ===")
print("Valid attempts:", len(attempt_features))
print("Players:", attempt_features["nfl_id"].nunique())
print(attempt_features.groupby("drill_name").agg(
    attempts=("nfl_id", "size"),
    players=("nfl_id", "nunique"),
).to_string())

# ---------------------------------------------------------
# 3. Aggregate repeated attempts into player-level features
# ---------------------------------------------------------
feature_cols = [
    "peak_speed",
    "mean_speed",
    "max_braking",
    "max_accel",
    "candidate_turn_deg",
    "speed_before_turn",
    "speed_after_turn",
    "speed_retention",
]

player_features = (
    attempt_features.groupby(["nfl_id", "drill_name"])[feature_cols]
    .median()
    .unstack("drill_name")
)

player_features.columns = [
    f"{metric}__{route}"
    for metric, route in player_features.columns
]

player_features = player_features.reset_index()

combine = pd.read_csv(DATA_DIR / "combine_results.csv")
combine = combine[combine["combine_position"].eq("WR")]

model_data = combine.merge(
    player_features, on="nfl_id", how="left",
    validate="one_to_one"
)

model_data.to_csv(
    OUT / "wr_combine_movement_features.csv", index=False
)

print("\n=== PLAYER FEATURE DATASET ===")
print("Players:", len(model_data))
print("Players with extracted features:",
      model_data.filter(regex="peak_speed").notna().any(axis=1).sum())

# ---------------------------------------------------------
# 4. Build post-draft NFL receiving outcomes
# ---------------------------------------------------------
players = pd.read_csv(DATA_DIR / "players.csv")
games = pd.read_csv(DATA_DIR / "games.csv")
plays = pd.read_csv(DATA_DIR / "player_play.csv", low_memory=False)

# Keep only the Combine WRs and merge draft years.
plays = plays[plays["nfl_id"].isin(set(combine["nfl_id"]))].copy()

plays = plays.merge(
    players[["nfl_id", "draft_year"]],
    on="nfl_id", how="inner", validate="many_to_one"
)

plays = plays.merge(
    games[["game_id", "season", "season_type"]],
    on="game_id", how="left", validate="many_to_one"
)

# Regular-season outcomes only, from the draft year onward.
plays = plays[
    plays["season_type"].eq("REG")
    & (plays["season"] >= plays["draft_year"])
].copy()

# Targeted pass plays are the primary separation sample.
targeted = plays[plays["target"].eq(True)].copy()

outcomes = targeted.groupby("nfl_id").agg(
    targeted_plays=("target", "size"),
    mean_separation=("separation_at_pass_forward", "mean"),
    median_separation=("separation_at_pass_forward", "median"),
    mean_rec_yards=("rec_yards", "mean"),
    mean_yac=("yards_after_catch", "mean"),
).reset_index()

# Attach counts of all observed regular-season play rows as context.
exposure = plays.groupby("nfl_id").agg(
    observed_play_rows=("play_id", "size"),
    seasons_observed=("season", "nunique"),
).reset_index()

outcomes = outcomes.merge(
    exposure, on="nfl_id", how="left", validate="one_to_one"
)

outcomes.to_csv(
    OUT / "wr_post_draft_outcomes.csv", index=False
)

print("\n=== POST-DRAFT OUTCOMES ===")
print("Players with targeted plays:", len(outcomes))
print(outcomes[[
    "targeted_plays", "mean_separation", "mean_rec_yards"
]].describe().round(3).to_string())

print("\nOutputs saved in:", OUT.resolve())


