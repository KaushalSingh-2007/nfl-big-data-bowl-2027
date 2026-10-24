from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)

ROUTES = [
    "COMEBACK_ROUTE_RIGHT",
    "DAGGER_ROUTE_LEFT",
]

raw = pd.read_csv(
    DATA_DIR / "combine_tracking.csv",
    usecols=[
        "nfl_id", "event_id", "entity_type", "time",
        "drill_name", "attempt", "x", "y", "s", "dir",
    ],
)

raw = raw[
    (raw["entity_type"] == "PLAYER")
    & raw["drill_name"].isin(ROUTES)
].copy()

raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
raw = raw.dropna(subset=["time", "x", "y"])
raw = raw.sort_values(
    ["nfl_id", "event_id", "drill_name", "attempt", "time"]
)

features = pd.read_csv(OUT / "wr_attempt_features_v2.csv")

# Select representative examples, including suspicious extremes.
chosen = []

for route in ROUTES:
    subset = features[features["drill_name"] == route]

    if subset.empty:
        continue

    chosen.extend(
        subset.nlargest(2, "candidate_turn_deg").to_dict("records")
    )
    chosen.extend(
        subset.nsmallest(2, "candidate_turn_deg").to_dict("records")
    )

chosen.extend(
    features.nlargest(3, "speed_retention").to_dict("records")
)

unique = []
seen = set()

for row in chosen:
    key = (
        row["nfl_id"], row["event_id"],
        row["drill_name"], row["attempt"],
    )

    if key not in seen:
        seen.add(key)
        unique.append(row)

for i, row in enumerate(unique):
    mask = (
        (raw["nfl_id"] == row["nfl_id"])
        & (raw["event_id"] == row["event_id"])
        & (raw["drill_name"] == row["drill_name"])
        & (raw["attempt"] == row["attempt"])
    )

    g = raw.loc[mask].sort_values("time").copy()

    if len(g) < 10:
        continue

    t = (
        g["time"].astype("int64").to_numpy() / 1e9
    )
    t -= t[0]

    x = g["x"].to_numpy(dtype=float)
    y = g["y"].to_numpy(dtype=float)
    speed = g["s"].to_numpy(dtype=float)

    # Smooth positions and avoid differentiating raw noisy coordinates.
    xs = (
        pd.Series(x)
        .rolling(5, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    ys = (
        pd.Series(y)
        .rolling(5, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    # Estimate movement direction from the smoothed trajectory.
    dx = np.gradient(xs, t)
    dy = np.gradient(ys, t)
    estimated_speed = np.hypot(dx, dy)

    # Ignore direction estimates when movement is nearly stationary.
    heading = np.arctan2(dy, dx)
    heading[estimated_speed < 0.5] = np.nan

    k = max(2, round(0.3 / np.median(np.diff(t))))
    angle_change = np.full(len(t), np.nan)

    for j in range(k, len(t) - k):
        h1 = heading[j - k]
        h2 = heading[j + k]

        if np.isfinite(h1) and np.isfinite(h2):
            angle_change[j] = abs(
                np.degrees(
                    np.arctan2(
                        np.sin(h2 - h1),
                        np.cos(h2 - h1),
                    )
                )
            )

    valid = (
        np.isfinite(angle_change)
        & (speed > 1.0)
        & (np.arange(len(t)) >= k)
        & (np.arange(len(t)) < len(t) - k)
    )

    candidates = np.flatnonzero(valid)

    if len(candidates) == 0:
        continue

    detected_idx = candidates[np.argmax(angle_change[candidates])]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Time-coloured trajectory: early samples and late samples differ.
    sc = ax.scatter(
        xs, ys,
        c=t,
        cmap="viridis",
        s=17,
        label="Trajectory (colour = time)",
    )
    ax.plot(xs, ys, alpha=0.45, linewidth=1)

    # Start and end markers.
    ax.scatter(
        xs[0], ys[0],
        marker="o", s=100,
        color="green", label="Start",
        zorder=5,
    )
    ax.scatter(
        xs[-1], ys[-1],
        marker="s", s=100,
        color="red", label="End",
        zorder=5,
    )

    # Show local movement directions around the candidate.
    for offset, colour, label in [
        (-k, "orange", "Before candidate"),
        (k, "red", "After candidate"),
    ]:
        j = detected_idx + offset

        if 0 <= j < len(t) and np.isfinite(heading[j]):
            arrow_len = 1.2
            ax.annotate(
                "",
                xy=(
                    xs[j] + arrow_len * np.cos(heading[j]),
                    ys[j] + arrow_len * np.sin(heading[j]),
                ),
                xytext=(xs[j], ys[j]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=colour,
                    lw=2,
                ),
            )
            ax.scatter(
                xs[j], ys[j],
                color=colour, s=55,
                label=label,
                zorder=6,
            )

    ax.scatter(
        xs[detected_idx], ys[detected_idx],
        marker="X", s=180,
        color="black",
        label="Current candidate",
        zorder=7,
    )

    plt.colorbar(sc, ax=ax, label="Seconds since start")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(
        f"{row['drill_name']} | player {row['nfl_id']} "
        f"| stored angle={row['candidate_turn_deg']:.1f}Â° "
        f"| recalculated angle={angle_change[detected_idx]:.1f}Â° "
        f"| time={t[detected_idx]:.2f}s"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    filename = f"turn_diagnostic_{i:02d}.png"
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)

print("Saved diagnostic plots to:", OUT.resolve())
print("Open turn_diagnostic_*.png")


