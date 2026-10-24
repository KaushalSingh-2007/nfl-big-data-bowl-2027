import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "research_outputs", "receiving_efficiency_mechanism")

print("=" * 90)
print("SPEED-CV DISCREPANCY AUDIT")
print("=" * 90)

# ---------------------------------------------------------------------
# 1. Load original player-level feature
# ---------------------------------------------------------------------
research = pd.read_csv(
    os.path.join(ROOT, "research_outputs", "wr_research_dataset.csv")
)

research["nfl_id"] = pd.to_numeric(research["nfl_id"], errors="coerce")

original_col = "speed_cv_DAGGER_ROUTE_LEFT"

if original_col not in research.columns:
    raise RuntimeError(
        f"Could not find {original_col} in wr_research_dataset.csv"
    )

original = research[
    ["nfl_id", "draft_year", original_col]
].copy()

original = original.rename(
    columns={original_col: "original_speed_cv"}
)

# ---------------------------------------------------------------------
# 2. Load reconstructed attempt-level features
# ---------------------------------------------------------------------
attempt_path = os.path.join(
    OUT,
    "dagger_speed_mechanism_attempts.csv"
)

attempts = pd.read_csv(attempt_path)

attempts["nfl_id"] = pd.to_numeric(
    attempts["nfl_id"],
    errors="coerce"
)

print(f"Original research players: {len(original)}")
print(f"Reconstructed attempts: {len(attempts)}")
print(f"Reconstructed players: {attempts['nfl_id'].nunique()}")

# ---------------------------------------------------------------------
# 3. Compute several possible player-level definitions
# ---------------------------------------------------------------------
# Definition A:
# median of attempt-level population-CV values.
a = (
    attempts.groupby("nfl_id")
    .agg(
        reconstructed_median_cv=("speed_cv", "median"),
        reconstructed_mean_cv=("speed_cv", "mean"),
        attempt_count=("speed_cv", "size"),
    )
    .reset_index()
)

# Definition B:
# pool all tracking rows belonging to a player, then compute CV.
pooled_rows = []

for nfl_id, g in attempts.groupby("nfl_id"):
    s = g["speed_mean"].to_numpy(dtype=float)
    # This is NOT the true pooled route CV; it is retained only as a
    # diagnostic showing what happens when attempt-level means are pooled.
    if len(s):
        pooled_rows.append({
            "nfl_id": nfl_id,
            "mean_of_attempt_means": np.mean(s),
            "std_of_attempt_means_population": np.std(s, ddof=0),
            "std_of_attempt_means_sample": np.std(s, ddof=1) if len(s) > 1 else np.nan,
        })

pooled = pd.DataFrame(pooled_rows)

# ---------------------------------------------------------------------
# 4. Merge everything
# ---------------------------------------------------------------------
comparison = original.merge(
    a,
    on="nfl_id",
    how="left"
)

comparison = comparison.merge(
    pooled,
    on="nfl_id",
    how="left"
)

comparison["diff_original_vs_median"] = (
    comparison["original_speed_cv"]
    - comparison["reconstructed_median_cv"]
)

comparison["abs_diff_original_vs_median"] = (
    comparison["diff_original_vs_median"].abs()
)

comparison["diff_original_vs_mean"] = (
    comparison["original_speed_cv"]
    - comparison["reconstructed_mean_cv"]
)

comparison["abs_diff_original_vs_mean"] = (
    comparison["diff_original_vs_mean"].abs()
)

comparison = comparison.sort_values(
    "abs_diff_original_vs_median",
    ascending=False
)

comparison_path = os.path.join(
    OUT,
    "speed_cv_discrepancy_full.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

# ---------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------
valid = comparison.dropna(
    subset=[
        "original_speed_cv",
        "reconstructed_median_cv"
    ]
)

print("\n" + "=" * 90)
print("ORIGINAL VS RECONSTRUCTED")
print("=" * 90)

print(f"Players compared: {len(valid)}")

print(
    f"Median absolute difference: "
    f"{valid['abs_diff_original_vs_median'].median():.8f}"
)

print(
    f"Mean absolute difference: "
    f"{valid['abs_diff_original_vs_median'].mean():.8f}"
)

print(
    f"90th percentile absolute difference: "
    f"{valid['abs_diff_original_vs_median'].quantile(.90):.8f}"
)

print(
    f"95th percentile absolute difference: "
    f"{valid['abs_diff_original_vs_median'].quantile(.95):.8f}"
)

print(
    f"Maximum absolute difference: "
    f"{valid['abs_diff_original_vs_median'].max():.8f}"
)

# ---------------------------------------------------------------------
# 6. Difference thresholds
# ---------------------------------------------------------------------
print("\nDifference thresholds:")

for threshold in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
    n = (
        valid["abs_diff_original_vs_median"]
        > threshold
    ).sum()

    pct = 100 * n / len(valid)

    print(
        f"  > {threshold:>5.3f}: "
        f"{n:>3} players ({pct:6.2f}%)"
    )

# ---------------------------------------------------------------------
# 7. Is discrepancy associated with number of attempts?
# ---------------------------------------------------------------------
if len(valid) >= 5:
    corr = valid[
        [
            "attempt_count",
            "abs_diff_original_vs_median"
        ]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    print(
        f"\nSpearman correlation between attempt count "
        f"and discrepancy: {corr:.6f}"
    )

# ---------------------------------------------------------------------
# 8. Largest discrepancies
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("TOP 20 DISCREPANCIES")
print("=" * 90)

cols = [
    "nfl_id",
    "draft_year",
    "attempt_count",
    "original_speed_cv",
    "reconstructed_median_cv",
    "reconstructed_mean_cv",
    "diff_original_vs_median",
    "abs_diff_original_vs_median",
]

print(
    valid[cols]
    .head(20)
    .to_string(index=False)
)

# ---------------------------------------------------------------------
# 9. Check whether sample/population SD explains differences
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("SINGLE-ATTEMPT SAMPLE VS POPULATION CV CHECK")
print("=" * 90)

single = attempts[
    attempts["nfl_id"].isin(
        valid.loc[
            valid["attempt_count"] == 1,
            "nfl_id"
        ]
    )
].copy()

# For each single attempt, recompute CV using both ddof=0 and ddof=1
# directly from the raw tracking file.
tracking_path = os.path.join(
    DATA,
    "combine_tracking.csv"
)

usecols = [
    "nfl_id",
    "event_id",
    "attempt",
    "time",
    "drill_name",
    "s",
]

tracking = pd.read_csv(
    tracking_path,
    usecols=usecols
)

tracking["nfl_id"] = pd.to_numeric(
    tracking["nfl_id"],
    errors="coerce"
)

tracking = tracking[
    tracking["drill_name"].eq("DAGGER_ROUTE_LEFT")
].copy()

tracking["s"] = pd.to_numeric(
    tracking["s"],
    errors="coerce"
)

tracking = tracking.dropna(
    subset=["nfl_id", "event_id", "attempt", "s"]
)

single_results = []

for key, g in tracking.groupby(
    ["nfl_id", "event_id", "attempt"]
):
    if g["nfl_id"].iloc[0] not in set(
        single["nfl_id"].unique()
    ):
        continue

    s = g["s"].to_numpy(dtype=float)

    if len(s) < 2:
        continue

    mean_s = np.mean(s)

    if mean_s == 0:
        continue

    pop_cv = np.std(s, ddof=0) / mean_s
    sample_cv = np.std(s, ddof=1) / mean_s

    single_results.append({
        "nfl_id": key[0],
        "event_id": key[1],
        "attempt": key[2],
        "n_samples": len(s),
        "population_cv": pop_cv,
        "sample_cv": sample_cv,
    })

single_results = pd.DataFrame(single_results)

if len(single_results):
    single_compare = single_results.merge(
        original,
        on="nfl_id",
        how="left"
    )

    single_compare["diff_population"] = (
        single_compare["original_speed_cv"]
        - single_compare["population_cv"]
    ).abs()

    single_compare["diff_sample"] = (
        single_compare["original_speed_cv"]
        - single_compare["sample_cv"]
    ).abs()

    print(
        f"Single-attempt route/player comparisons: "
        f"{len(single_compare)}"
    )

    print(
        f"Median abs diff using population SD: "
        f"{single_compare['diff_population'].median():.8f}"
    )

    print(
        f"Median abs diff using sample SD: "
        f"{single_compare['diff_sample'].median():.8f}"
    )

    print(
        "\nLargest single-attempt cases:"
    )

    print(
        single_compare.sort_values(
            "diff_population",
            ascending=False
        ).head(15).to_string(index=False)
    )

    single_path = os.path.join(
        OUT,
        "speed_cv_single_attempt_definition_check.csv"
    )

    single_compare.to_csv(
        single_path,
        index=False
    )

# ---------------------------------------------------------------------
# 10. Print relevant original feature-building code
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("ORIGINAL BUILD SCRIPT: SPEED-CV RELEVANT LINES")
print("=" * 90)

build_script = os.path.join(
    ROOT,
    "src",
    "features",
    "build_whole_route_features.py"
)

if os.path.exists(build_script):
    lines = open(
        build_script,
        "r",
        encoding="utf-8"
    ).readlines()

    keywords = [
        "speed_cv",
        "speed_std",
        "std(",
        "groupby",
        "median",
        "mean",
        "DAGGER_ROUTE_LEFT",
        "player_route",
    ]

    for i, line in enumerate(lines, start=1):
        if any(k in line for k in keywords):
            start = max(1, i - 3)
            end = min(len(lines), i + 3)

            print(
                f"\n--- lines {start}-{end} ---"
            )

            for j in range(start, end + 1):
                print(
                    f"{j:4d}: {lines[j-1].rstrip()}"
                )

print("\n" + "=" * 90)
print("AUDIT COMPLETE")
print("=" * 90)

print(f"\nFull comparison saved to:")
print(comparison_path)
