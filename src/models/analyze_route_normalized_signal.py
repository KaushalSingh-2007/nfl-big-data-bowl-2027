from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "research_outputs" / "model_comparison" / "route_normalized_signal"
OUT.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROJECT_ROOT / "research_outputs" / "wr_research_dataset.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

df = pd.read_csv(DATA_PATH)

TARGET = "mean_separation"

routes = {
    "DAGGER": "DAGGER_ROUTE_LEFT",
    "COMEBACK": "COMEBACK_ROUTE_RIGHT",
}

metrics = [
    "speed_cv",
    "peak_speed",
    "mean_speed",
    "straightness_ratio",
    "late_early_speed_ratio",
]

required = ["mean_separation", "forty", "vertical"]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Same 64-player conventional population used in locked validation.
base = df.dropna(subset=required).copy()

print("=" * 80)
print("ROUTE-NORMALIZED MOVEMENT SIGNAL ANALYSIS")
print("=" * 80)
print(f"Dataset: {DATA_PATH}")
print(f"Base population: {len(base)}")

# ---------------------------------------------------------------------
# Create route-specific z-scores.
#
# Important:
# Each route is standardized independently because raw feature scales
# differ substantially between Dagger and Comeback.
# ---------------------------------------------------------------------

z_columns = {}

for route_name, suffix in routes.items():

    for metric in metrics:

        col = f"{metric}_{suffix}"

        if col not in base.columns:
            continue

        values = base[col]

        mean = values.mean()
        std = values.std(ddof=0)

        zcol = f"{metric}_{route_name.lower()}_z"

        if std == 0 or pd.isna(std):
            base[zcol] = np.nan
        else:
            base[zcol] = (values - mean) / std

        z_columns[(route_name, metric)] = zcol

# ---------------------------------------------------------------------
# Correlation of route-normalized features with separation
# ---------------------------------------------------------------------

rows = []

for (route_name, metric), zcol in z_columns.items():

    temp = base[[zcol, TARGET]].dropna()

    if len(temp) < 4:
        continue

    pr, pp = pearsonr(temp[zcol], temp[TARGET])
    sr, sp = spearmanr(temp[zcol], temp[TARGET])

    rows.append({
        "route": route_name,
        "metric": metric,
        "feature": zcol,
        "n": len(temp),
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_r": sr,
        "spearman_p": sp,
    })

corr = pd.DataFrame(rows)

corr.to_csv(
    OUT / "route_normalized_correlations.csv",
    index=False,
)

print("\n" + "=" * 80)
print("ROUTE-NORMALIZED FEATURES vs MEAN SEPARATION")
print("=" * 80)
print(corr.to_string(index=False))

# ---------------------------------------------------------------------
# Difference between Dagger and Comeback normalized movement.
#
# This asks whether a player's relative behavior on one route differs
# systematically from their relative behavior on another route.
# ---------------------------------------------------------------------

difference_rows = []

for metric in metrics:

    dagger_col = z_columns.get(("DAGGER", metric))
    comeback_col = z_columns.get(("COMEBACK", metric))

    if dagger_col is None or comeback_col is None:
        continue

    temp = base[
        [dagger_col, comeback_col, TARGET]
    ].dropna().copy()

    temp["route_z_difference"] = (
        temp[dagger_col] - temp[comeback_col]
    )

    if len(temp) >= 4:
        pr, pp = pearsonr(
            temp["route_z_difference"],
            temp[TARGET],
        )

        sr, sp = spearmanr(
            temp["route_z_difference"],
            temp[TARGET],
        )
    else:
        pr = pp = sr = sp = np.nan

    difference_rows.append({
        "metric": metric,
        "n": len(temp),
        "mean_difference": temp["route_z_difference"].mean(),
        "median_difference": temp["route_z_difference"].median(),
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_r": sr,
        "spearman_p": sp,
    })

difference = pd.DataFrame(difference_rows)

difference.to_csv(
    OUT / "route_normalized_differences.csv",
    index=False,
)

print("\n" + "=" * 80)
print("DAGGER Z-SCORE - COMEBACK Z-SCORE")
print("=" * 80)
print(difference.to_string(index=False))

# ---------------------------------------------------------------------
# Partial correlation after removing conventional Combine metrics.
#
# Residualize both the normalized movement feature and separation
# against 40-yard dash + vertical.
# ---------------------------------------------------------------------

def residualize(y, X):

    Xmat = np.column_stack([
        np.ones(len(X)),
        X,
    ])

    coef = np.linalg.lstsq(
        Xmat,
        y,
        rcond=None,
    )[0]

    return y - Xmat @ coef


partial_rows = []

for (route_name, metric), zcol in z_columns.items():

    temp = base[
        [TARGET, "forty", "vertical", zcol]
    ].dropna()

    if len(temp) < 8:
        continue

    X = temp[["forty", "vertical"]].to_numpy()

    y_resid = residualize(
        temp[TARGET].to_numpy(),
        X,
    )

    feature_resid = residualize(
        temp[zcol].to_numpy(),
        X,
    )

    pr, pp = pearsonr(
        feature_resid,
        y_resid,
    )

    sr, sp = spearmanr(
        feature_resid,
        y_resid,
    )

    partial_rows.append({
        "route": route_name,
        "metric": metric,
        "feature": zcol,
        "n": len(temp),
        "partial_pearson_r": pr,
        "partial_pearson_p": pp,
        "partial_spearman_r": sr,
        "partial_spearman_p": sp,
    })

partial = pd.DataFrame(partial_rows)

partial.to_csv(
    OUT / "route_normalized_partial_correlations.csv",
    index=False,
)

print("\n" + "=" * 80)
print("ROUTE-NORMALIZED SIGNAL AFTER CONTROLLING FOR 40-YD + VERTICAL")
print("=" * 80)
print(partial.to_string(index=False))

# ---------------------------------------------------------------------
# Save the complete analysis population.
# ---------------------------------------------------------------------

save_columns = [
    "nfl_id",
    "draft_year",
    "mean_separation",
    "forty",
    "vertical",
]

for zcol in z_columns.values():
    save_columns.append(zcol)

save_columns = [
    c for c in save_columns
    if c in base.columns
]

base[save_columns].to_csv(
    OUT / "route_normalized_population.csv",
    index=False,
)

# ---------------------------------------------------------------------
# Speed-CV focused summary.
# ---------------------------------------------------------------------

speed_summary = []

for route_name in routes:

    zcol = z_columns.get((route_name, "speed_cv"))

    if zcol is None:
        continue

    temp = base[[zcol, TARGET]].dropna()

    pr, pp = pearsonr(temp[zcol], temp[TARGET])
    sr, sp = spearmanr(temp[zcol], temp[TARGET])

    speed_summary.append({
        "route": route_name,
        "n": len(temp),
        "z_mean": temp[zcol].mean(),
        "z_std": temp[zcol].std(ddof=0),
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_r": sr,
        "spearman_p": sp,
    })

speed_summary = pd.DataFrame(speed_summary)

speed_summary.to_csv(
    OUT / "speed_cv_normalized_summary.csv",
    index=False,
)

print("\n" + "=" * 80)
print("SPEED CV NORMALIZED SUMMARY")
print("=" * 80)
print(speed_summary.to_string(index=False))

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

for path in sorted(OUT.glob("*")):
    if path.is_file():
        print(path)

print("\nDONE")
