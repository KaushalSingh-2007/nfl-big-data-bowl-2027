import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "research_outputs", "receiving_efficiency_mechanism")
os.makedirs(OUT, exist_ok=True)

print("=" * 80)
print("SPEED-CV MECHANISM ANALYSIS")
print("=" * 80)

# ---------------------------------------------------------------------
# 1. Load research population
# ---------------------------------------------------------------------
research_path = os.path.join(ROOT, "research_outputs", "wr_research_dataset.csv")
research = pd.read_csv(research_path)

research["nfl_id"] = pd.to_numeric(research["nfl_id"], errors="coerce")

print(f"Research population: {len(research)}")

# ---------------------------------------------------------------------
# 2. Load raw Combine tracking
# ---------------------------------------------------------------------
tracking_path = os.path.join(DATA, "combine_tracking.csv")

usecols = [
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

print("\nLoading Combine tracking...")
tracking = pd.read_csv(tracking_path, usecols=usecols)
tracking = tracking[tracking["entity_type"] == "PLAYER"].copy()

tracking["nfl_id"] = pd.to_numeric(tracking["nfl_id"], errors="coerce")
tracking = tracking[tracking["drill_name"].eq("DAGGER_ROUTE_LEFT")].copy()

tracking["time_dt"] = pd.to_datetime(
    tracking["time"],
    errors="coerce",
    utc=True
)

tracking["s"] = pd.to_numeric(tracking["s"], errors="coerce")
tracking["a"] = pd.to_numeric(tracking["a"], errors="coerce")
tracking["x"] = pd.to_numeric(tracking["x"], errors="coerce")
tracking["y"] = pd.to_numeric(tracking["y"], errors="coerce")

tracking = tracking.dropna(
    subset=["nfl_id", "event_id", "attempt", "time_dt", "s"]
).copy()

tracking = tracking.sort_values(
    ["nfl_id", "event_id", "attempt", "time_dt"]
).reset_index(drop=True)

print(f"Dagger tracking rows: {len(tracking)}")
print(f"Dagger attempts: {tracking[['nfl_id','event_id','attempt']].drop_duplicates().shape[0]}")
print(f"Dagger players: {tracking['nfl_id'].nunique()}")

# ---------------------------------------------------------------------
# 3. Derive within-route speed dynamics
# ---------------------------------------------------------------------
rows = []

group_cols = ["nfl_id", "event_id", "attempt"]

for key, g in tracking.groupby(group_cols, sort=False):
    g = g.sort_values("time_dt").copy()

    if len(g) < 5:
        continue

    t = (g["time_dt"] - g["time_dt"].iloc[0]).dt.total_seconds().to_numpy()
    s = g["s"].to_numpy(dtype=float)

    dt = np.diff(t)
    ds = np.diff(s)

    valid = np.isfinite(dt) & np.isfinite(ds) & (dt > 0)

    if valid.sum() < 3:
        continue

    dtv = dt[valid]
    dsv = ds[valid]

    dsdt = dsv / dtv

    # Ignore extreme derivative artifacts from tiny timestamp gaps.
    dsdt = dsdt[np.isfinite(dsdt)]

    if len(dsdt) < 3:
        continue

    positive = dsdt[dsdt > 0]
    negative = dsdt[dsdt < 0]

    # Acceleration/deceleration statistics derived directly from speed.
    mean_acc = np.mean(positive) if len(positive) else 0.0
    mean_dec = np.mean(negative) if len(negative) else 0.0

    max_acc = np.max(positive) if len(positive) else 0.0
    max_dec = np.min(negative) if len(negative) else 0.0

    mean_abs_change = np.mean(np.abs(dsdt))

    frac_acc = np.mean(dsdt > 0)
    frac_dec = np.mean(dsdt < 0)
    frac_stable = np.mean(dsdt == 0)

    # Count sign changes in speed derivative.
    signs = np.sign(dsdt)
    signs = signs[signs != 0]

    if len(signs) >= 2:
        transitions = np.sum(signs[1:] != signs[:-1])
    else:
        transitions = 0

    # Speed variability.
    speed_mean = np.mean(s)
    speed_std = np.std(s, ddof=0)
    speed_cv = speed_std / speed_mean if speed_mean != 0 else np.nan

    # Speed range.
    speed_range = np.max(s) - np.min(s)

    # Route phases.
    n = len(s)
    split = max(1, n // 3)

    early = s[:split]
    middle = s[split:2 * split]
    late = s[-split:]

    early_median = np.median(early)
    middle_median = np.median(middle)
    late_median = np.median(late)

    late_early_ratio = (
        late_median / early_median
        if early_median != 0 else np.nan
    )

    # Fraction of route spent above/below the route median speed.
    route_median = np.median(s)
    frac_above_median = np.mean(s > route_median)
    frac_below_median = np.mean(s < route_median)

    rows.append({
        "nfl_id": key[0],
        "event_id": key[1],
        "attempt": key[2],

        "n_samples": len(g),
        "duration_s": t[-1] - t[0],

        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "speed_cv": speed_cv,
        "speed_range": speed_range,

        "early_speed_median": early_median,
        "middle_speed_median": middle_median,
        "late_speed_median": late_median,
        "late_early_speed_ratio": late_early_ratio,

        "mean_positive_dsdt": mean_acc,
        "mean_negative_dsdt": mean_dec,
        "max_positive_dsdt": max_acc,
        "max_negative_dsdt": max_dec,
        "mean_abs_dsdt": mean_abs_change,

        "frac_accelerating": frac_acc,
        "frac_decelerating": frac_dec,
        "frac_stable_speed": frac_stable,

        "speed_sign_transitions": transitions,

        "frac_above_median_speed": frac_above_median,
        "frac_below_median_speed": frac_below_median,
    })

route = pd.DataFrame(rows)

route_path = os.path.join(OUT, "dagger_speed_mechanism_attempts.csv")
route.to_csv(route_path, index=False)

print(f"\nDerived attempt-level mechanism features: {route.shape}")
print(f"Saved: {route_path}")

# ---------------------------------------------------------------------
# 4. Aggregate to player level
# ---------------------------------------------------------------------
numeric_cols = [
    c for c in route.columns
    if c not in ["nfl_id", "event_id", "attempt"]
]

player_route = (
    route.groupby("nfl_id")[numeric_cols]
    .median()
    .reset_index()
)

# Rename to make the route explicit.
player_route = player_route.rename(
    columns={
        c: f"{c}_DAGGER_ROUTE_LEFT"
        for c in numeric_cols
    }
)

player_path = os.path.join(OUT, "dagger_speed_mechanism_player.csv")
player_route.to_csv(player_path, index=False)

print(f"Player-level mechanism table: {player_route.shape}")
print(f"Saved: {player_path}")

# ---------------------------------------------------------------------
# 5. Reconstruct receiving-efficiency outcome exactly as locked model
# ---------------------------------------------------------------------
players = pd.read_csv(
    os.path.join(DATA, "players.csv"),
    usecols=["nfl_id", "nfl_position", "draft_year"]
)

games = pd.read_csv(
    os.path.join(DATA, "games.csv"),
    usecols=["game_id", "season", "season_type"]
)

pp = pd.read_csv(
    os.path.join(DATA, "player_play.csv"),
    usecols=[
        "game_id",
        "play_id",
        "nfl_id",
        "target",
        "rec_yards",
    ]
)

players["nfl_id"] = pd.to_numeric(players["nfl_id"], errors="coerce")
pp["nfl_id"] = pd.to_numeric(pp["nfl_id"], errors="coerce")

pp = pp.merge(
    players[["nfl_id", "nfl_position", "draft_year"]],
    on="nfl_id",
    how="inner"
)

pp = pp.merge(
    games,
    on="game_id",
    how="inner"
)

pp["target"] = pd.to_numeric(pp["target"], errors="coerce")
pp["rec_yards"] = pd.to_numeric(pp["rec_yards"], errors="coerce")

pp = pp[
    pp["nfl_position"].eq("WR")
    & pp["season_type"].eq("REG")
    & (pp["season"] >= pp["draft_year"])
    & pp["target"].eq(1)
].copy()

# This is the exact definition used in the locked receiving-efficiency
# analysis: sum of target-row receiving yards, with missing rec_yards
# treated as zero, divided by number of target rows.
pp["rec_yards_filled"] = pp["rec_yards"].fillna(0)

outcome = (
    pp.groupby("nfl_id")
    .agg(
        observed_targets=("target", "size"),
        receiving_yards=("rec_yards_filled", "sum"),
        nonmissing_rec_yards=("rec_yards", "count"),
    )
    .reset_index()
)

outcome["receiving_efficiency"] = (
    outcome["receiving_yards"] /
    outcome["observed_targets"]
)

# ---------------------------------------------------------------------
# 6. Merge research + outcome + mechanism
# ---------------------------------------------------------------------
df = research.merge(
    outcome[[
        "nfl_id",
        "observed_targets",
        "receiving_yards",
        "nonmissing_rec_yards",
        "receiving_efficiency",
    ]],
    on="nfl_id",
    how="left"
)

# The research dataset already contains the original locked
# speed_cv_DAGGER_ROUTE_LEFT feature. The mechanism table independently
# reconstructs speed CV from raw tracking. Keep both so we can verify
# that the reconstruction agrees with the locked feature.
df = df.merge(
    player_route,
    on="nfl_id",
    how="left",
    suffixes=("_research", "_mechanism")
)

# Explicitly preserve the original locked feature for all predictive
# comparisons. The mechanism-derived version is retained separately.
if "speed_cv_DAGGER_ROUTE_LEFT_research" in df.columns:
    df["speed_cv_DAGGER_ROUTE_LEFT"] = df[
        "speed_cv_DAGGER_ROUTE_LEFT_research"
    ]

# Compare original locked speed CV against independently reconstructed
# mechanism speed CV.
if "speed_cv_DAGGER_ROUTE_LEFT_mechanism" in df.columns:
    check = df[
        [
            "nfl_id",
            "speed_cv_DAGGER_ROUTE_LEFT",
            "speed_cv_DAGGER_ROUTE_LEFT_mechanism",
        ]
    ].dropna()

    if len(check):
        check["absolute_difference"] = (
            check["speed_cv_DAGGER_ROUTE_LEFT"]
            - check["speed_cv_DAGGER_ROUTE_LEFT_mechanism"]
        ).abs()

        check_path = os.path.join(
            OUT,
            "speed_cv_reconstruction_check.csv"
        )
        check.to_csv(check_path, index=False)

        print("\nSpeed-CV reconstruction check:")
        print(f"  Players compared: {len(check)}")
        print(
            f"  Mean absolute difference: "
            f"{check['absolute_difference'].mean():.8f}"
        )
        print(
            f"  Median absolute difference: "
            f"{check['absolute_difference'].median():.8f}"
        )
        print(
            f"  Maximum absolute difference: "
            f"{check['absolute_difference'].max():.8f}"
        )

# Keep only players in the exact common locked population:
# 40-yard + vertical + original locked speed-CV + receiving efficiency.
common_cols = [
    "forty",
    "vertical",
    "speed_cv_DAGGER_ROUTE_LEFT",
    "receiving_efficiency",
]

common = df.dropna(subset=common_cols).copy()

print("\n" + "=" * 80)
print("COMMON LOCKED POPULATION")
print("=" * 80)

print(f"n = {len(common)}")
print(common["draft_year"].value_counts().sort_index().to_string())

# ---------------------------------------------------------------------
# 7. Correlation table
# ---------------------------------------------------------------------
# Resolve mechanism feature names from the actual generated player table.
# This avoids failures if an existing project convention uses a slightly
# different name for the late/early speed ratio.
mechanism_base_names = [
    "speed_cv",
    "speed_range",
    "early_speed_median",
    "middle_speed_median",
    "late_speed_median",
    "late_early_speed_ratio",
    "late_early_speed_ratio_DAGGER_ROUTE_LEFT",
    "mean_positive_dsdt",
    "mean_negative_dsdt",
    "max_positive_dsdt",
    "max_negative_dsdt",
    "mean_abs_dsdt",
    "frac_accelerating",
    "frac_decelerating",
    "speed_sign_transitions",
    "frac_above_median_speed",
]

mechanism_cols = []

for base in mechanism_base_names:
    candidates = [
        f"{base}_DAGGER_ROUTE_LEFT",
        base,
    ]

    found = next(
        (c for c in candidates if c in common.columns),
        None
    )

    if found is not None and found not in mechanism_cols:
        mechanism_cols.append(found)

print("\nResolved mechanism features:")
for c in mechanism_cols:
    print(f"  {c}")

results = []

for feature in mechanism_cols:
    tmp = common[[feature, "receiving_efficiency"]].dropna()

    if len(tmp) < 5:
        continue

    pearson_r, pearson_p = pearsonr(
        tmp[feature],
        tmp["receiving_efficiency"]
    )

    spearman_r, spearman_p = spearmanr(
        tmp[feature],
        tmp["receiving_efficiency"]
    )

    results.append({
        "feature": feature,
        "n": len(tmp),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    })

corr = pd.DataFrame(results)

corr["abs_spearman"] = corr["spearman_r"].abs()
corr = corr.sort_values("abs_spearman", ascending=False)

corr_path = os.path.join(OUT, "mechanism_correlations.csv")
corr.to_csv(corr_path, index=False)

print("\nMechanism correlations:")
print(
    corr[
        [
            "feature",
            "n",
            "pearson_r",
            "pearson_p",
            "spearman_r",
            "spearman_p",
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------------------
# 8. Correlation with the original speed-CV signal
# ---------------------------------------------------------------------
signal_rows = []

for feature in mechanism_cols:
    if feature == "speed_cv_DAGGER_ROUTE_LEFT":
        continue

    tmp = common[
        ["speed_cv_DAGGER_ROUTE_LEFT", feature]
    ].dropna()

    if len(tmp) < 5:
        continue

    r, p = pearsonr(
        tmp["speed_cv_DAGGER_ROUTE_LEFT"],
        tmp[feature]
    )

    sr, sp = spearmanr(
        tmp["speed_cv_DAGGER_ROUTE_LEFT"],
        tmp[feature]
    )

    signal_rows.append({
        "feature": feature,
        "n": len(tmp),
        "pearson_with_speed_cv": r,
        "pearson_p": p,
        "spearman_with_speed_cv": sr,
        "spearman_p": sp,
    })

signal_corr = pd.DataFrame(signal_rows)
signal_corr["abs_spearman"] = signal_corr["spearman_with_speed_cv"].abs()
signal_corr = signal_corr.sort_values("abs_spearman", ascending=False)

signal_path = os.path.join(
    OUT,
    "mechanism_correlations_with_speed_cv.csv"
)

signal_corr.to_csv(signal_path, index=False)

print("\nMechanism relationship with original speed CV:")
print(signal_corr.to_string(index=False))

# ---------------------------------------------------------------------
# 9. Residualized outcome relationships
#
# Remove conventional Combine effects (40 yd + vertical), then inspect
# whether each mechanism feature still relates to receiving efficiency.
# ---------------------------------------------------------------------
X_conv = common[["forty", "vertical"]].to_numpy()
y = common["receiving_efficiency"].to_numpy()

conv_model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

conv_model.fit(X_conv, y)
y_pred_conv = conv_model.predict(X_conv)
y_resid = y - y_pred_conv

partial_rows = []

for feature in mechanism_cols:
    tmp = common[[feature]].copy()
    mask = tmp[feature].notna().to_numpy()

    if mask.sum() < 5:
        continue

    r, p = pearsonr(
        tmp.loc[mask, feature],
        y_resid[mask]
    )

    sr, sp = spearmanr(
        tmp.loc[mask, feature],
        y_resid[mask]
    )

    partial_rows.append({
        "feature": feature,
        "n": int(mask.sum()),
        "pearson_residual_r": r,
        "pearson_p": p,
        "spearman_residual_r": sr,
        "spearman_p": sp,
    })

partial = pd.DataFrame(partial_rows)
partial["abs_spearman"] = partial["spearman_residual_r"].abs()
partial = partial.sort_values("abs_spearman", ascending=False)

partial_path = os.path.join(
    OUT,
    "mechanism_residual_correlations.csv"
)

partial.to_csv(partial_path, index=False)

print("\nMechanism vs receiving-efficiency residual after 40 yd + vertical:")
print(partial.to_string(index=False))

# ---------------------------------------------------------------------
# 10. Compact descriptive summary
# ---------------------------------------------------------------------
summary = []

for feature in mechanism_cols:
    if feature not in common.columns:
        continue

    s = common[feature].dropna()

    if len(s) == 0:
        continue

    summary.append({
        "feature": feature,
        "n": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "min": s.min(),
        "max": s.max(),
    })

summary_df = pd.DataFrame(summary)

summary_path = os.path.join(
    OUT,
    "mechanism_feature_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

# ---------------------------------------------------------------------
# 11. Save merged analysis population
# ---------------------------------------------------------------------
merged_path = os.path.join(
    OUT,
    "mechanism_analysis_population.csv"
)

common.to_csv(merged_path, index=False)

print("\n" + "=" * 80)
print("FILES WRITTEN")
print("=" * 80)

for p in [
    route_path,
    player_path,
    corr_path,
    signal_path,
    partial_path,
    summary_path,
    merged_path,
]:
    print(p)

print("\nDone.")


