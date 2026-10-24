from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "research_outputs" / "model_comparison" / "route_specific_signal"
OUT.mkdir(parents=True, exist_ok=True)

DATA_PATH = OUT.parent.parent.parent / "wr_research_dataset.csv"

if not DATA_PATH.exists():
    DATA_PATH = PROJECT_ROOT / "research_outputs" / "wr_research_dataset.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find wr_research_dataset.csv under {PROJECT_ROOT}")

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("ROUTE-SPECIFIC MOVEMENT SIGNAL ANALYSIS")
print("=" * 80)
print(f"Dataset: {DATA_PATH}")
print(f"Shape: {df.shape}")

target = "mean_separation"

routes = {
    "DAGGER": "DAGGER_ROUTE_LEFT",
    "COMEBACK": "COMEBACK_ROUTE_RIGHT",
}

movement_metrics = [
    "speed_cv",
    "peak_speed",
    "mean_speed",
    "straightness_ratio",
    "late_early_speed_ratio",
]

# ---------------------------------------------------------------------
# Identify available route/metric columns
# ---------------------------------------------------------------------

route_cols = {}

for route_name, route_suffix in routes.items():
    route_cols[route_name] = {}

    for metric in movement_metrics:
        col = f"{metric}_{route_suffix}"
        if col in df.columns:
            route_cols[route_name][metric] = col

print("\nAvailable movement features:")
for route_name, cols in route_cols.items():
    print(f"\n{route_name}:")
    for metric, col in cols.items():
        print(f"  {metric}: {col}")

# ---------------------------------------------------------------------
# Common population
#
# Keep this deliberately fixed:
# - observed mean separation
# - forty
# - vertical
# - route-specific feature
#
# The 40 + vertical restriction gives us the same conventional
# population used in the locked validation.
# ---------------------------------------------------------------------

required_base = ["mean_separation", "forty", "vertical"]

missing_base = [c for c in required_base if c not in df.columns]
if missing_base:
    raise ValueError(f"Missing required columns: {missing_base}")

base = df.dropna(subset=required_base).copy()

print(f"\nCommon conventional population: {len(base)}")

# ---------------------------------------------------------------------
# Correlation helper
# ---------------------------------------------------------------------

def correlation_row(feature, outcome, data):
    d = data[[feature, outcome]].dropna()

    if len(d) < 4:
        return {
            "feature": feature,
            "n": len(d),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }

    pr, pp = pearsonr(d[feature], d[outcome])
    sr, sp = spearmanr(d[feature], d[outcome])

    return {
        "feature": feature,
        "n": len(d),
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_r": sr,
        "spearman_p": sp,
    }

# ---------------------------------------------------------------------
# 1. Route-specific correlations with mean separation
# ---------------------------------------------------------------------

rows = []

for route_name, cols in route_cols.items():
    for metric, col in cols.items():
        row = correlation_row(col, target, base)
        row["route"] = route_name
        row["metric"] = metric
        rows.append(row)

corr_df = pd.DataFrame(rows)

corr_df = corr_df[
    [
        "route",
        "metric",
        "feature",
        "n",
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
    ]
]

corr_df.to_csv(OUT / "route_feature_correlations.csv", index=False)

print("\n" + "=" * 80)
print("ROUTE-SPECIFIC CORRELATIONS WITH MEAN SEPARATION")
print("=" * 80)
print(
    corr_df[
        [
            "route",
            "metric",
            "n",
            "pearson_r",
            "pearson_p",
            "spearman_r",
            "spearman_p",
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------------------
# 2. Direct Dagger vs Comeback comparison for each metric
# ---------------------------------------------------------------------

paired_rows = []

for metric in movement_metrics:
    dagger_col = route_cols["DAGGER"].get(metric)
    comeback_col = route_cols["COMEBACK"].get(metric)

    if dagger_col is None or comeback_col is None:
        continue

    temp = base[
        [dagger_col, comeback_col, target]
    ].dropna().copy()

    temp["route_difference"] = temp[dagger_col] - temp[comeback_col]

    row = {
        "metric": metric,
        "n": len(temp),
        "dagger_mean": temp[dagger_col].mean(),
        "comeback_mean": temp[comeback_col].mean(),
        "dagger_median": temp[dagger_col].median(),
        "comeback_median": temp[comeback_col].median(),
        "difference_mean": temp["route_difference"].mean(),
        "difference_median": temp["route_difference"].median(),
    }

    if len(temp) >= 4:
        pr, pp = pearsonr(temp["route_difference"], temp[target])
        sr, sp = spearmanr(temp["route_difference"], temp[target])

        row["difference_pearson_r"] = pr
        row["difference_pearson_p"] = pp
        row["difference_spearman_r"] = sr
        row["difference_spearman_p"] = sp
    else:
        row["difference_pearson_r"] = np.nan
        row["difference_pearson_p"] = np.nan
        row["difference_spearman_r"] = np.nan
        row["difference_spearman_p"] = np.nan

    paired_rows.append(row)

paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv(OUT / "route_feature_differences.csv", index=False)

print("\n" + "=" * 80)
print("DAGGER - COMEBACK DIFFERENCE")
print("=" * 80)
print(paired_df.to_string(index=False))

# ---------------------------------------------------------------------
# 3. Correlation between Dagger and Comeback versions
# ---------------------------------------------------------------------

cross_rows = []

for metric in movement_metrics:
    dagger_col = route_cols["DAGGER"].get(metric)
    comeback_col = route_cols["COMEBACK"].get(metric)

    if dagger_col is None or comeback_col is None:
        continue

    temp = base[[dagger_col, comeback_col]].dropna()

    if len(temp) >= 4:
        pr, pp = pearsonr(temp[dagger_col], temp[comeback_col])
        sr, sp = spearmanr(temp[dagger_col], temp[comeback_col])
    else:
        pr = pp = sr = sp = np.nan

    cross_rows.append(
        {
            "metric": metric,
            "n": len(temp),
            "pearson_r_dagger_vs_comeback": pr,
            "pearson_p": pp,
            "spearman_r_dagger_vs_comeback": sr,
            "spearman_p": sp,
        }
    )

cross_df = pd.DataFrame(cross_rows)
cross_df.to_csv(OUT / "cross_route_feature_correlations.csv", index=False)

print("\n" + "=" * 80)
print("DAGGER VS COMEBACK FEATURE CORRELATIONS")
print("=" * 80)
print(cross_df.to_string(index=False))

# ---------------------------------------------------------------------
# 4. Residualized relationship:
#
# Remove the linear contribution of conventional Combine metrics
# (40 yd + vertical) from BOTH separation and each movement feature.
#
# This is descriptive, not a predictive model-selection exercise.
# ---------------------------------------------------------------------

def residualize(y, X):
    Xmat = np.column_stack([np.ones(len(X)), X])
    coef = np.linalg.lstsq(Xmat, y, rcond=None)[0]
    return y - Xmat @ coef

partial_rows = []

for route_name, cols in route_cols.items():
    for metric, col in cols.items():

        temp = base[[target, "forty", "vertical", col]].dropna().copy()

        if len(temp) < 8:
            continue

        X = temp[["forty", "vertical"]].to_numpy()

        y_resid = residualize(
            temp[target].to_numpy(),
            X,
        )

        feature_resid = residualize(
            temp[col].to_numpy(),
            X,
        )

        pr, pp = pearsonr(feature_resid, y_resid)
        sr, sp = spearmanr(feature_resid, y_resid)

        partial_rows.append(
            {
                "route": route_name,
                "metric": metric,
                "feature": col,
                "n": len(temp),
                "partial_pearson_r": pr,
                "partial_pearson_p": pp,
                "partial_spearman_r": sr,
                "partial_spearman_p": sp,
            }
        )

partial_df = pd.DataFrame(partial_rows)
partial_df.to_csv(OUT / "route_feature_partial_correlations.csv", index=False)

print("\n" + "=" * 80)
print("RELATIONSHIP AFTER REMOVING 40-YD + VERTICAL")
print("=" * 80)
print(partial_df.to_string(index=False))

# ---------------------------------------------------------------------
# 5. Speed-CV focused diagnostic
# ---------------------------------------------------------------------

speed_rows = []

dagger_cv = route_cols["DAGGER"].get("speed_cv")
comeback_cv = route_cols["COMEBACK"].get("speed_cv")

if dagger_cv and comeback_cv:

    temp = base[
        [target, "forty", "vertical", dagger_cv, comeback_cv]
    ].dropna().copy()

    temp["dagger_minus_comeback_speed_cv"] = (
        temp[dagger_cv] - temp[comeback_cv]
    )

    temp["absolute_route_speed_cv_difference"] = (
        temp[dagger_cv] - temp[comeback_cv]
    ).abs()

    temp.to_csv(
        OUT / "speed_cv_route_comparison_population.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("SPEED-CV ROUTE COMPARISON")
    print("=" * 80)

    print(f"Population: {len(temp)}")
    print(
        f"Dagger speed CV mean:   {temp[dagger_cv].mean():.6f}"
    )
    print(
        f"Comeback speed CV mean: {temp[comeback_cv].mean():.6f}"
    )
    print(
        f"Mean Dagger - Comeback:  "
        f"{temp['dagger_minus_comeback_speed_cv'].mean():.6f}"
    )
    print(
        f"Median Dagger - Comeback: "
        f"{temp['dagger_minus_comeback_speed_cv'].median():.6f}"
    )

# ---------------------------------------------------------------------
# 6. Scatter plots
# ---------------------------------------------------------------------

if dagger_cv:
    temp = base[[dagger_cv, target]].dropna()

    plt.figure(figsize=(7, 5))
    plt.scatter(temp[dagger_cv], temp[target], alpha=0.7)
    plt.xlabel("Dagger route speed CV")
    plt.ylabel("Mean separation")
    plt.title("Dagger Speed CV vs Mean Separation")
    plt.tight_layout()
    plt.savefig(OUT / "dagger_speed_cv_vs_separation.png", dpi=160)
    plt.close()

if comeback_cv:
    temp = base[[comeback_cv, target]].dropna()

    plt.figure(figsize=(7, 5))
    plt.scatter(temp[comeback_cv], temp[target], alpha=0.7)
    plt.xlabel("Comeback route speed CV")
    plt.ylabel("Mean separation")
    plt.title("Comeback Speed CV vs Mean Separation")
    plt.tight_layout()
    plt.savefig(OUT / "comeback_speed_cv_vs_separation.png", dpi=160)
    plt.close()

if dagger_cv and comeback_cv:
    temp = base[[dagger_cv, comeback_cv]].dropna()

    plt.figure(figsize=(7, 5))
    plt.scatter(temp[dagger_cv], temp[comeback_cv], alpha=0.7)
    plt.xlabel("Dagger route speed CV")
    plt.ylabel("Comeback route speed CV")
    plt.title("Route-to-Route Speed CV Relationship")
    plt.tight_layout()
    plt.savefig(OUT / "dagger_vs_comeback_speed_cv.png", dpi=160)
    plt.close()

print("\n" + "=" * 80)
print("FILES WRITTEN")
print("=" * 80)

for path in sorted(OUT.glob("*")):
    if path.is_file():
        print(path)

print("\nDONE")
