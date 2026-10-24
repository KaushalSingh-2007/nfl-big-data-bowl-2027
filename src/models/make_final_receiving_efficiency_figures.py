from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_outputs"
FIG = OUT / "final_figures"

FIG.mkdir(parents=True, exist_ok=True)

RESULTS = OUT / "receiving_efficiency_final_sanity" / "final_locked_model_results.csv"
FOLDS = OUT / "receiving_efficiency_final_sanity" / "final_locked_fold_predictions.csv"
DATASET = OUT / "wr_research_dataset.csv"
OUTCOME = OUT / "receiving_efficiency_final_sanity" / "receiving_efficiency_outcome_reconstructed.csv"

for p in [RESULTS, FOLDS, DATASET, OUTCOME]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")

results = pd.read_csv(RESULTS)
folds = pd.read_csv(FOLDS)
dataset = pd.read_csv(DATASET)
outcome = pd.read_csv(OUTCOME)

print("Loaded files successfully.")
print(f"Results shape: {results.shape}")
print(f"Fold predictions shape: {folds.shape}")
print(f"Research dataset shape: {dataset.shape}")
print(f"Outcome shape: {outcome.shape}")

# ------------------------------------------------------------
# Figure 1 — Model comparison
# ------------------------------------------------------------

required_cols = {
    "threshold",
    "model",
    "rmse",
    "mae",
}

missing = required_cols - set(results.columns)

if missing:
    raise ValueError(
        f"Missing columns in results file: {sorted(missing)}"
    )

threshold_order = sorted(
    results["threshold"].unique()
)

model_order = [
    "conventional",
    "movement",
    "combined",
]

model_labels = {
    "conventional": "40 yd + Vertical",
    "movement": "Route Speed CV",
    "combined": "40 yd + Vertical + Speed CV",
}

fig, ax = plt.subplots(
    figsize=(9, 5.5)
)

x = np.arange(
    len(threshold_order)
)

width = 0.24

for i, model in enumerate(model_order):

    sub = (
        results[
            results["model"] == model
        ]
        .set_index("threshold")
        .reindex(threshold_order)
    )

    ax.bar(
        x + (i - 1) * width,
        sub["rmse"],
        width=width,
        label=model_labels[model],
    )

labels = []

for threshold in threshold_order:

    if threshold == 0:
        labels.append("All players\n(n=64)")
    elif threshold == 10:
        labels.append("≥10 targets\n(n=55)")
    elif threshold == 20:
        labels.append("≥20 targets\n(n=47)")
    else:
        labels.append(
            f"≥{threshold} targets"
        )

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylabel(
    "Repeated-CV RMSE\n(receiving yards per target)"
)

ax.set_title(
    "Model comparison for future receiving efficiency"
)

ax.legend(
    frameon=False
)

ax.grid(
    axis="y",
    alpha=0.25
)

ax.set_axisbelow(True)

plt.tight_layout()

fig.savefig(
    FIG / "figure_1_model_comparison_rmse.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Figure 1 created successfully.")

# ------------------------------------------------------------
# Figure 2 — Paired prediction error
# ------------------------------------------------------------

fold20 = folds[
    folds["threshold"] == 20
].copy()

if fold20.empty:
    raise ValueError(
        "No threshold=20 predictions found."
    )

required = {
    "nfl_id",
    "model",
    "y_true",
    "prediction",
}

missing = required - set(fold20.columns)

if missing:
    raise ValueError(
        f"Missing columns in fold predictions: {sorted(missing)}"
    )

fold20["abs_error"] = (
    fold20["prediction"] -
    fold20["y_true"]
).abs()

player_model_error = (
    fold20
    .groupby(
        ["nfl_id", "model"]
    )["abs_error"]
    .mean()
    .reset_index()
)

error_pivot = (
    player_model_error
    .pivot(
        index="nfl_id",
        columns="model",
        values="abs_error",
    )
)

error_pivot = error_pivot.dropna(
    subset=[
        "conventional",
        "movement",
    ]
)

error_pivot[
    "movement_minus_conventional"
] = (
    error_pivot["movement"] -
    error_pivot["conventional"]
)

player_diff = (
    error_pivot[
        "movement_minus_conventional"
    ]
    .sort_values()
)

mean_diff = player_diff.mean()
median_diff = player_diff.median()

fig, ax = plt.subplots(
    figsize=(9, 5.5)
)

ax.hist(
    player_diff.values,
    bins=15,
    edgecolor="black",
    alpha=0.8,
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.5,
)

ax.axvline(
    mean_diff,
    linestyle="-",
    linewidth=1.5,
    label=f"Mean = {mean_diff:.2f}",
)

ax.set_xlabel(
    "Movement-model absolute error − conventional-model absolute error"
)

ax.set_ylabel(
    "Number of players"
)

ax.set_title(
    "Paired prediction-error difference for players with ≥20 targets"
)

ax.text(
    0.98,
    0.95,
    f"Median = {median_diff:.2f}\n"
    "Negative = movement model better",
    transform=ax.transAxes,
    ha="right",
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.35",
        alpha=0.08,
    ),
)

ax.legend(
    frameon=False
)

ax.grid(
    axis="y",
    alpha=0.2
)

ax.set_axisbelow(True)

plt.tight_layout()

fig.savefig(
    FIG / "figure_2_paired_error_difference_20_targets.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    f"Figure 2 created successfully "
    f"({len(player_diff)} players)."
)

# ------------------------------------------------------------
# Figure 3 — Speed CV vs receiving efficiency
# ------------------------------------------------------------

# The research dataset contains the movement feature,
# while the final reconstructed outcome is stored separately.
# Merge them by nfl_id.

if "nfl_id" not in dataset.columns:
    raise ValueError(
        "Research dataset does not contain nfl_id."
    )

if "nfl_id" not in outcome.columns:
    raise ValueError(
        "Outcome file does not contain nfl_id."
    )

# Identify the receiving-efficiency column.
possible_efficiency_columns = [
    "receiving_efficiency",
    "yards_per_target",
    "receiving_yards_per_target",
    "observed_yards_per_target",
]

eff_col = None

for col in possible_efficiency_columns:
    if col in outcome.columns:
        eff_col = col
        break

if eff_col is None:
    raise ValueError(
        "Could not identify receiving-efficiency column.\n"
        f"Available outcome columns: {list(outcome.columns)}"
    )

print(
    f"\nUsing outcome column for Figure 3: {eff_col}"
)

# Keep only the necessary columns from outcome.
outcome_small = outcome[
    ["nfl_id", eff_col]
].copy()

outcome_small = outcome_small.rename(
    columns={
        eff_col: "receiving_efficiency"
    }
)

# Avoid accidental many-to-many merges.
outcome_small = (
    outcome_small
    .drop_duplicates(subset=["nfl_id"])
)

scatter = dataset[
    [
        "nfl_id",
        "speed_cv_DAGGER_ROUTE_LEFT",
    ]
].merge(
    outcome_small,
    on="nfl_id",
    how="inner",
)

# Same ≥20-target reliability population used
# by the final locked analysis.
reliable_ids = set(
    folds[
        folds["threshold"] == 20
    ]["nfl_id"].unique()
)

scatter = scatter[
    scatter["nfl_id"].isin(
        reliable_ids
    )
].dropna()

if len(scatter) == 0:
    raise ValueError(
        "No reliable players available for Figure 3."
    )

xv = scatter[
    "speed_cv_DAGGER_ROUTE_LEFT"
].to_numpy()

yv = scatter[
    "receiving_efficiency"
].to_numpy()

coef = np.polyfit(
    xv,
    yv,
    1,
)

xx = np.linspace(
    xv.min(),
    xv.max(),
    100,
)

yy = (
    coef[0] * xx +
    coef[1]
)

spearman = pd.Series(
    xv
).corr(
    pd.Series(yv),
    method="spearman",
)

fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)

ax.scatter(
    xv,
    yv,
    alpha=0.75,
    s=42,
)

ax.plot(
    xx,
    yy,
    linestyle="--",
    linewidth=1.5,
    label="Linear trend",
)

ax.set_xlabel(
    "Dagger-route speed coefficient of variation"
)

ax.set_ylabel(
    "Receiving yards per target"
)

ax.set_title(
    "Route-level speed variability and future receiving efficiency\n"
    "(players with ≥20 observed targets)"
)

ax.text(
    0.03,
    0.95,
    f"Spearman ρ = {spearman:.2f}\n"
    f"n = {len(scatter)}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.35",
        alpha=0.08,
    ),
)

ax.legend(
    frameon=False
)

ax.grid(
    alpha=0.2
)

ax.set_axisbelow(True)

plt.tight_layout()

fig.savefig(
    FIG / "figure_3_speed_cv_vs_receiving_efficiency.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    f"Figure 3 created successfully "
    f"({len(scatter)} players)."
)

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL FIGURES CREATED")
print("=" * 65)

for p in sorted(
    FIG.glob("*.png")
):
    print(p.name)
    print(f"  {p}")

print("\nFigure 2:")
print(
    f"  Players: {len(player_diff)}"
)
print(
    f"  Mean paired error difference: "
    f"{mean_diff:.4f}"
)
print(
    f"  Median paired error difference: "
    f"{median_diff:.4f}"
)

print("\nFigure 3:")
print(
    f"  Players: {len(scatter)}"
)
print(
    f"  Spearman rho: "
    f"{spearman:.4f}"
)

print("\nDone.")
