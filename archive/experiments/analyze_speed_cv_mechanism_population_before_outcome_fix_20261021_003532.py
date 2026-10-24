import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RESEARCH = os.path.join(
    ROOT,
    "research_outputs",
    "wr_research_dataset.csv"
)

MECH_DIR = os.path.join(
    ROOT,
    "research_outputs",
    "speed_cv_mechanism_exact"
)

MECH = os.path.join(
    MECH_DIR,
    "exact_mechanism_player_features.csv"
)

OUT = os.path.join(
    MECH_DIR,
    "population_analysis"
)

os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("MECHANISM ANALYSIS — FEATURE-SPECIFIC POPULATIONS")
print("=" * 70)

research = pd.read_csv(RESEARCH)
mechanism = pd.read_csv(MECH)

# ---------------------------------------------------------------------
# Merge.
# ---------------------------------------------------------------------
df = research.merge(
    mechanism,
    on="nfl_id",
    how="inner"
)

print(f"Research rows: {len(research)}")
print(f"Mechanism rows: {len(mechanism)}")
print(f"Merged rows: {len(df)}")

# ---------------------------------------------------------------------
# Reconstruct the same receiving-efficiency definition already used
# by the locked receiving-efficiency validation.
#
# IMPORTANT:
# The research dataset contains the locked outcome for the model
# population, so we use that directly here.
# ---------------------------------------------------------------------
outcome_col = "receiving_efficiency"

if outcome_col not in df.columns:
    raise RuntimeError(
        f"Missing {outcome_col} in research dataset."
    )

# ---------------------------------------------------------------------
# Candidate mechanism variables.
# ---------------------------------------------------------------------
mechanism_cols = [
    "speed_cv",
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

# ---------------------------------------------------------------------
# Common baseline population:
# 40-yard + vertical + receiving efficiency.
# ---------------------------------------------------------------------
baseline = df.dropna(
    subset=[
        "forty",
        "vertical",
        outcome_col
    ]
).copy()

print()
print("Baseline population (40 yd + vertical + outcome):", len(baseline))

# ---------------------------------------------------------------------
# Function for residualized correlation.
# Remove linear effects of 40-yard + vertical from both:
#   mechanism feature
#   receiving efficiency
#
# This is exploratory, not a causal estimate.
# ---------------------------------------------------------------------
def residualized_spearman(data, feature):

    tmp = data[
        [
            "forty",
            "vertical",
            feature,
            outcome_col
        ]
    ].dropna()

    if len(tmp) < 10:
        return np.nan, np.nan, len(tmp)

    X = tmp[["forty", "vertical"]].to_numpy()

    model_feature = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    )

    model_outcome = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    )

    model_feature.fit(X, tmp[feature].to_numpy())
    model_outcome.fit(X, tmp[outcome_col].to_numpy())

    feature_resid = (
        tmp[feature].to_numpy()
        - model_feature.predict(X)
    )

    outcome_resid = (
        tmp[outcome_col].to_numpy()
        - model_outcome.predict(X)
    )

    r, p = spearmanr(
        feature_resid,
        outcome_resid
    )

    return r, p, len(tmp)


# ---------------------------------------------------------------------
# Analyze every feature independently.
# ---------------------------------------------------------------------
rows = []

for feature in mechanism_cols:

    if feature not in df.columns:
        continue

    tmp = df[
        [
            "nfl_id",
            "draft_year",
            "forty",
            "vertical",
            outcome_col,
            feature
        ]
    ].dropna(
        subset=[feature, outcome_col]
    ).copy()

    n = len(tmp)

    if n < 10:
        continue

    pearson_r, pearson_p = pearsonr(
        tmp[feature],
        tmp[outcome_col]
    )

    spearman_r, spearman_p = spearmanr(
        tmp[feature],
        tmp[outcome_col]
    )

    residual_r, residual_p, residual_n = (
        residualized_spearman(df, feature)
    )

    rows.append({
        "feature": feature,
        "n": n,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "residualized_spearman_r": residual_r,
        "residualized_p": residual_p,
        "residualized_n": residual_n,
        "draft_2023_n": int((tmp["draft_year"] == 2023).sum()),
        "draft_2024_n": int((tmp["draft_year"] == 2024).sum()),
        "draft_2025_n": int((tmp["draft_year"] == 2025).sum()),
    })

results = pd.DataFrame(rows)

results["abs_spearman"] = results["spearman_r"].abs()
results["abs_residualized"] = (
    results["residualized_spearman_r"].abs()
)

results = results.sort_values(
    "abs_residualized",
    ascending=False
)

print()
print("=" * 70)
print("ALL FEATURES — LARGEST AVAILABLE POPULATION")
print("=" * 70)

print(
    results[
        [
            "feature",
            "n",
            "spearman_r",
            "spearman_p",
            "residualized_spearman_r",
            "residualized_p"
        ]
    ]
    .round(4)
    .to_string(index=False)
)

results.to_csv(
    os.path.join(
        OUT,
        "mechanism_population_correlations.csv"
    ),
    index=False
)

# ---------------------------------------------------------------------
# Also print specifically the three features that looked most
# interesting in the complete-case n=39 analysis.
# ---------------------------------------------------------------------
focus = [
    "frac_stable_speed",
    "speed_sign_transitions",
    "frac_decelerating",
]

print()
print("=" * 70)
print("FOCUS FEATURES")
print("=" * 70)

print(
    results[
        results["feature"].isin(focus)
    ][
        [
            "feature",
            "n",
            "spearman_r",
            "spearman_p",
            "residualized_spearman_r",
            "residualized_p",
            "draft_2023_n",
            "draft_2024_n",
            "draft_2025_n"
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# ---------------------------------------------------------------------
# Test whether mechanism features themselves are available across
# the locked 64-player population.
# ---------------------------------------------------------------------
locked = df.dropna(
    subset=[
        "forty",
        "vertical",
        "speed_cv_DAGGER_ROUTE_LEFT",
        outcome_col
    ]
).copy()

print()
print("=" * 70)
print("LOCKED MODEL POPULATION")
print("=" * 70)
print("Locked population:", len(locked))

availability = []

for feature in mechanism_cols:

    if feature not in locked.columns:
        continue

    available = locked[feature].notna()

    availability.append({
        "feature": feature,
        "locked_population": len(locked),
        "available": int(available.sum()),
        "missing": int((~available).sum()),
        "coverage_pct": 100 * available.mean()
    })

availability = pd.DataFrame(availability)

print(
    availability
    .sort_values("coverage_pct", ascending=False)
    .round(2)
    .to_string(index=False)
)

availability.to_csv(
    os.path.join(
        OUT,
        "mechanism_population_availability.csv"
    ),
    index=False
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
print("Output directory:")
print(OUT)
