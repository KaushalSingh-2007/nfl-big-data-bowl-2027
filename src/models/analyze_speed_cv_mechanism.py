import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "research_outputs", "speed_cv_mechanism_exact")
os.makedirs(OUT, exist_ok=True)

TRACKING = os.path.join(DATA, "combine_tracking.csv")
RESEARCH = os.path.join(OUT.replace("speed_cv_mechanism_exact", ""), "wr_research_dataset.csv")

# Fallback to the normal research dataset location.
if not os.path.exists(RESEARCH):
    RESEARCH = os.path.join(ROOT, "research_outputs", "wr_research_dataset.csv")

ROUTE = "DAGGER_ROUTE_LEFT"

print("=" * 70)
print("EXACT SPEED-CV MECHANISM ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------------------
# 1. Load tracking
# ---------------------------------------------------------------------
tracking = pd.read_csv(TRACKING, low_memory=False)

tracking = tracking[
    (tracking["entity_type"] == "PLAYER") &
    (tracking["drill_name"] == ROUTE)
].copy()

tracking["time_dt"] = pd.to_datetime(tracking["time"], errors="coerce")

tracking = tracking.dropna(
    subset=["nfl_id", "event_id", "attempt", "time_dt", "x", "y", "s"]
).copy()

tracking["nfl_id"] = pd.to_numeric(tracking["nfl_id"], errors="coerce")
tracking["attempt"] = pd.to_numeric(tracking["attempt"], errors="coerce")

tracking = tracking.dropna(
    subset=["nfl_id", "attempt"]
).copy()

tracking["nfl_id"] = tracking["nfl_id"].astype(int)
tracking["attempt"] = tracking["attempt"].astype(int)

print(f"PLAYER rows: {len(tracking):,}")
print(f"Players: {tracking['nfl_id'].nunique():,}")

# Exact original grouping keys.
group_cols = ["nfl_id", "event_id", "drill_name", "attempt"]

# ---------------------------------------------------------------------
# 2. EXACT reconstruction of original speed preprocessing
# ---------------------------------------------------------------------
def extract_exact(g):
    # EXACTLY as build_whole_route_features.py:
    # aggregate duplicate timestamps first.
    g = (
        g.groupby("time", as_index=False)
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            s=("s", "mean")
        )
        .sort_values("time")
    )

    if len(g) < 20:
        return None

    # Parse elapsed time using the same sorted observations.
    t_dt = pd.to_datetime(g["time"], errors="coerce")

    if t_dt.isna().any():
        return None

    t = (
        t_dt - t_dt.iloc[0]
    ).dt.total_seconds().to_numpy(float)

    x = g["x"].to_numpy(float)
    y = g["y"].to_numpy(float)
    s = g["s"].to_numpy(float)

    if len(s) < 20:
        return None

    # EXACT original speed smoothing.
    smooth_s = (
        pd.Series(s)
        .rolling(5, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    # EXACT original speed statistics.
    mean_smooth = np.mean(smooth_s)
    std_smooth = np.std(smooth_s)

    speed_cv = (
        float(std_smooth / mean_smooth)
        if mean_smooth > 0
        else np.nan
    )

    # -----------------------------------------------------------------
    # Original route phase definition:
    # first half / second half by ELAPSED TIME.
    # -----------------------------------------------------------------
    midpoint = t[-1] / 2

    early_mask = t <= midpoint
    late_mask = t > midpoint

    early_speed = float(np.median(smooth_s[early_mask]))
    late_speed = float(np.median(smooth_s[late_mask]))

    # -----------------------------------------------------------------
    # Mechanism features, all derived from EXACT smoothed speed.
    # -----------------------------------------------------------------
    ds = np.diff(smooth_s)
    dt = np.diff(t)

    valid_dt = dt > 0

    if valid_dt.any():
        dsdt = ds[valid_dt] / dt[valid_dt]

        positive = dsdt[dsdt > 0]
        negative = dsdt[dsdt < 0]

        mean_positive = float(np.mean(positive)) if len(positive) else 0.0
        mean_negative = float(np.mean(negative)) if len(negative) else 0.0
        max_positive = float(np.max(positive)) if len(positive) else 0.0
        max_negative = float(np.min(negative)) if len(negative) else 0.0
        mean_abs = float(np.mean(np.abs(dsdt)))

        frac_acc = float(np.mean(dsdt > 0))
        frac_dec = float(np.mean(dsdt < 0))
        frac_stable = float(np.mean(dsdt == 0))

        signs = np.sign(dsdt)
        nonzero = signs[signs != 0]

        if len(nonzero) >= 2:
            transitions = int(np.sum(nonzero[1:] != nonzero[:-1]))
        else:
            transitions = 0
    else:
        mean_positive = np.nan
        mean_negative = np.nan
        max_positive = np.nan
        max_negative = np.nan
        mean_abs = np.nan
        frac_acc = np.nan
        frac_dec = np.nan
        frac_stable = np.nan
        transitions = np.nan

    route_median = float(np.median(smooth_s))

    return {
        "duration_s": float(t[-1]),
        "n_samples": int(len(g)),

        # Original feature, reproduced exactly.
        "speed_cv": speed_cv,
        "speed_std": float(std_smooth),
        "speed_mean": float(mean_smooth),

        # Additional mechanism features.
        "speed_range": float(np.max(smooth_s) - np.min(smooth_s)),
        "early_speed_median": early_speed,
        "late_speed_median": late_speed,
        "late_early_speed_ratio": (
            late_speed / early_speed
            if early_speed > 0.5
            else np.nan
        ),

        "mean_positive_dsdt": mean_positive,
        "mean_negative_dsdt": mean_negative,
        "max_positive_dsdt": max_positive,
        "max_negative_dsdt": max_negative,
        "mean_abs_dsdt": mean_abs,

        "frac_accelerating": frac_acc,
        "frac_decelerating": frac_dec,
        "frac_stable_speed": frac_stable,
        "speed_sign_transitions": transitions,

        "frac_above_median_speed": float(
            np.mean(smooth_s > route_median)
        ),
        "frac_below_median_speed": float(
            np.mean(smooth_s < route_median)
        ),
    }


rows = []

for key, g in tracking.groupby(group_cols, sort=False):
    result = extract_exact(g)

    if result is None:
        continue

    nfl_id, event_id, drill_name, attempt = key

    result.update({
        "nfl_id": int(nfl_id),
        "event_id": event_id,
        "drill_name": drill_name,
        "attempt": int(attempt),
    })

    rows.append(result)

attempts = pd.DataFrame(rows)

print()
print("Exact reconstructed attempts:", len(attempts))
print("Exact reconstructed players:", attempts["nfl_id"].nunique())

# ---------------------------------------------------------------------
# 3. Median across attempts exactly as original pipeline.
# ---------------------------------------------------------------------
feature_cols = [
    "duration_s",
    "n_samples",
    "speed_cv",
    "speed_std",
    "speed_mean",
    "speed_range",
    "early_speed_median",
    "late_speed_median",
    "late_early_speed_ratio",
    "mean_positive_dsdt",
    "mean_negative_dsdt",
    "max_positive_dsdt",
    "max_negative_dsdt",
    "mean_abs_dsdt",
    "frac_accelerating",
    "frac_decelerating",
    "frac_stable_speed",
    "speed_sign_transitions",
    "frac_above_median_speed",
    "frac_below_median_speed",
]

player_features = (
    attempts
    .groupby("nfl_id")[feature_cols]
    .median()
    .reset_index()
)

player_features.to_csv(
    os.path.join(OUT, "exact_mechanism_player_features.csv"),
    index=False
)

attempts.to_csv(
    os.path.join(OUT, "exact_mechanism_attempts.csv"),
    index=False
)

# ---------------------------------------------------------------------
# 4. Compare EXACT reconstruction against locked research feature.
# ---------------------------------------------------------------------
research = pd.read_csv(RESEARCH)

check = research[
    ["nfl_id", "speed_cv_DAGGER_ROUTE_LEFT"]
].merge(
    player_features[["nfl_id", "speed_cv"]],
    on="nfl_id",
    how="inner"
).dropna()

check["absolute_difference"] = (
    check["speed_cv_DAGGER_ROUTE_LEFT"] - check["speed_cv"]
).abs()

print()
print("=" * 70)
print("EXACT SPEED-CV RECONSTRUCTION CHECK")
print("=" * 70)
print(f"Players compared: {len(check)}")
print(f"Mean absolute difference:   {check['absolute_difference'].mean():.10f}")
print(f"Median absolute difference: {check['absolute_difference'].median():.10f}")
print(f"Maximum absolute difference:{check['absolute_difference'].max():.10f}")

print()
print("Top 10 absolute differences:")
print(
    check.sort_values("absolute_difference", ascending=False)
    .head(10)
    .to_string(index=False)
)

check.to_csv(
    os.path.join(OUT, "exact_speed_cv_reconstruction_check.csv"),
    index=False
)

# ---------------------------------------------------------------------
# 5. Build outcome exactly as current receiving-efficiency analysis.
# ---------------------------------------------------------------------
pp = pd.read_csv(
    os.path.join(DATA, "player_play.csv"),
    low_memory=False
)

players = pd.read_csv(
    os.path.join(DATA, "players.csv"),
    low_memory=False
)

games = pd.read_csv(
    os.path.join(DATA, "games.csv"),
    low_memory=False
)

pp = pp.merge(
    players[["nfl_id", "nfl_position", "draft_year"]],
    on="nfl_id",
    how="left"
)

pp = pp.merge(
    games[["game_id", "season_type"]],
    on="game_id",
    how="left"
)

pp["draft_year"] = pd.to_numeric(pp["draft_year"], errors="coerce")
pp["season"] = pd.to_numeric(
    pp.get("season", pd.Series(np.nan, index=pp.index)),
    errors="coerce"
)

# Use game season from games if available through player_play merge.
if "season" not in pp.columns or pp["season"].isna().all():
    # games.csv contains season, so merge it explicitly.
    pp = pp.drop(columns=["season"], errors="ignore")
    pp = pp.merge(
        games[["game_id", "season"]],
        on="game_id",
        how="left"
    )

pp["target"] = pd.to_numeric(pp["target"], errors="coerce")
pp["rec_yards"] = pd.to_numeric(pp["rec_yards"], errors="coerce")

pp = pp[
    (pp["nfl_position"] == "WR") &
    (pp["season_type"] == "REG") &
    (pp["season"] >= pp["draft_year"]) &
    (pp["target"] == 1)
].copy()

# Strict post-draft condition.
pp = pp[pp["season"] > pp["draft_year"]].copy()

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

df = research.merge(
    outcome[
        ["nfl_id", "observed_targets", "receiving_efficiency"]
    ],
    on="nfl_id",
    how="left"
)

df = df.merge(
    player_features,
    on="nfl_id",
    how="left",
    suffixes=("_research", "_mechanism")
)

# ---------------------------------------------------------------------
# 6. Mechanism correlations with receiving efficiency.
# ---------------------------------------------------------------------
common = df.dropna(
    subset=[
        "forty",
        "vertical",
        "speed_cv_DAGGER_ROUTE_LEFT",
        "receiving_efficiency",
    ]
).copy()

print()
print("=" * 70)
print("MECHANISM SIGNALS vs RECEIVING EFFICIENCY")
print("=" * 70)
print("Common population:", len(common))

mechanism_cols = [
    "speed_range",
    "early_speed_median",
    "late_speed_median",
    "late_early_speed_ratio",
    "mean_positive_dsdt",
    "mean_negative_dsdt",
    "max_positive_dsdt",
    "max_negative_dsdt",
    "mean_abs_dsdt",
    "frac_accelerating",
    "frac_decelerating",
    "frac_stable_speed",
    "speed_sign_transitions",
    "frac_above_median_speed",
    "frac_below_median_speed",
]

results = []

for feature in mechanism_cols:
    col = f"{feature}"

    if col not in common.columns:
        continue

    tmp = common[
        ["receiving_efficiency", col]
    ].dropna()

    if len(tmp) < 10:
        continue

    pr, ppval = pearsonr(
        tmp["receiving_efficiency"],
        tmp[col]
    )

    sr, spval = spearmanr(
        tmp["receiving_efficiency"],
        tmp[col]
    )

    results.append({
        "feature": feature,
        "n": len(tmp),
        "pearson_r": pr,
        "pearson_p": ppval,
        "spearman_r": sr,
        "spearman_p": spval,
    })

results = pd.DataFrame(results)

if len(results):
    results["abs_spearman"] = results["spearman_r"].abs()
    results = results.sort_values(
        "abs_spearman",
        ascending=False
    )

    print(results.round(4).to_string(index=False))

    results.to_csv(
        os.path.join(
            OUT,
            "mechanism_vs_receiving_efficiency.csv"
        ),
        index=False
    )

# ---------------------------------------------------------------------
# 7. Mechanism features vs the ORIGINAL speed CV.
# ---------------------------------------------------------------------
print()
print("=" * 70)
print("MECHANISM RELATIONSHIP WITH ORIGINAL SPEED CV")
print("=" * 70)

speed_results = []

for feature in mechanism_cols:
    if feature not in common.columns:
        continue

    tmp = common[
        ["speed_cv_DAGGER_ROUTE_LEFT", feature]
    ].dropna()

    if len(tmp) < 10:
        continue

    sr, spval = spearmanr(
        tmp["speed_cv_DAGGER_ROUTE_LEFT"],
        tmp[feature]
    )

    speed_results.append({
        "feature": feature,
        "n": len(tmp),
        "spearman_with_original_speed_cv": sr,
        "p": spval,
    })

speed_results = pd.DataFrame(speed_results)

if len(speed_results):
    speed_results["abs_spearman"] = (
        speed_results[
            "spearman_with_original_speed_cv"
        ].abs()
    )

    speed_results = speed_results.sort_values(
        "abs_spearman",
        ascending=False
    )

    print(
        speed_results.round(4).to_string(index=False)
    )

    speed_results.to_csv(
        os.path.join(
            OUT,
            "mechanism_vs_original_speed_cv.csv"
        ),
        index=False
    )

# ---------------------------------------------------------------------
# 8. Residualized receiving-efficiency relationships.
# ---------------------------------------------------------------------
print()
print("=" * 70)
print("RESIDUALIZED MECHANISM SIGNAL")
print("=" * 70)

X = common[["forty", "vertical"]].to_numpy()
y = common["receiving_efficiency"].to_numpy()

model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

model.fit(X, y)
residual = y - model.predict(X)

residual_results = []

for feature in mechanism_cols:
    if feature not in common.columns:
        continue

    tmp = common[[feature]].copy()
    tmp["residual"] = residual
    tmp = tmp.dropna()

    if len(tmp) < 10:
        continue

    sr, spval = spearmanr(
        tmp[feature],
        tmp["residual"]
    )

    residual_results.append({
        "feature": feature,
        "n": len(tmp),
        "spearman_r": sr,
        "p": spval,
    })

residual_results = pd.DataFrame(residual_results)

if len(residual_results):
    residual_results["abs_spearman"] = (
        residual_results["spearman_r"].abs()
    )

    residual_results = residual_results.sort_values(
        "abs_spearman",
        ascending=False
    )

    print(
        residual_results.round(4).to_string(index=False)
    )

    residual_results.to_csv(
        os.path.join(
            OUT,
            "mechanism_vs_receiving_efficiency_residual.csv"
        ),
        index=False
    )

print()
print("=" * 70)
print("DONE")
print("=" * 70)
print("Outputs:")
print(OUT)
