from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research_outputs" / "wr_research_dataset.csv"
OUT = ROOT / "research_outputs" / "receiving_efficiency"

OUT.mkdir(parents=True, exist_ok=True)


def evaluate(df, features, target, model_name, n_repeats=10):
    cols = features + [target]
    d = df[cols].dropna().copy()

    X = d[features].to_numpy()
    y = d[target].to_numpy()

    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=n_repeats,
        random_state=42
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(rkf.split(X), start=1):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ])

        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        rows.append({
            "model": model_name,
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "rmse": np.sqrt(mean_squared_error(y[test_idx], pred)),
            "mae": mean_absolute_error(y[test_idx], pred),
            "r2": r2_score(y[test_idx], pred),
        })

    result = pd.DataFrame(rows)

    summary = {
        "model": model_name,
        "n_players": len(d),
        "rmse_mean": result["rmse"].mean(),
        "rmse_std": result["rmse"].std(ddof=1),
        "mae_mean": result["mae"].mean(),
        "mae_std": result["mae"].std(ddof=1),
        "r2_mean": result["r2"].mean(),
        "r2_std": result["r2"].std(ddof=1),
    }

    return result, summary


def run_population(df, population_name, target="observed_yards_per_observed_target"):
    models = [
        (
            "A_conventional",
            ["forty", "vertical"]
        ),
        (
            "B_movement",
            ["speed_cv_DAGGER_ROUTE_LEFT"]
        ),
        (
            "C_combined",
            ["forty", "vertical", "speed_cv_DAGGER_ROUTE_LEFT"]
        ),
    ]

    all_folds = []
    summaries = []

    for name, features in models:
        folds, summary = evaluate(
            df,
            features,
            target,
            name
        )

        folds["population"] = population_name
        folds["target"] = target
        all_folds.append(folds)

        summary["population"] = population_name
        summary["target"] = target
        summary["features"] = ", ".join(features)
        summaries.append(summary)

    return pd.concat(all_folds, ignore_index=True), pd.DataFrame(summaries)


print("=" * 80)
print("RECEIVING EFFICIENCY VALIDATION")
print("=" * 80)

df = pd.read_csv(DATA)

print(f"\nResearch population: {len(df)}")

target = "observed_yards_per_observed_target"

print("\nTarget:")
print(target)

print("\nTarget summary:")
print(df[target].describe().to_string())

# Main population: all players with complete model features + outcome.
folds_all, summary_all = run_population(
    df,
    "all_complete",
    target
)

# Reliability sensitivity: require at least 10 observed targets.
if "observed_separation_count" in df.columns:
    df_10 = df[df["observed_separation_count"] >= 10].copy()
    folds_10, summary_10 = run_population(
        df_10,
        "at_least_10_observed_targets",
        target
    )
else:
    folds_10 = pd.DataFrame()
    summary_10 = pd.DataFrame()

# Require at least 20 observed targets.
if "observed_separation_count" in df.columns:
    df_20 = df[df["observed_separation_count"] >= 20].copy()
    folds_20, summary_20 = run_population(
        df_20,
        "at_least_20_observed_targets",
        target
    )
else:
    folds_20 = pd.DataFrame()
    summary_20 = pd.DataFrame()

folds = pd.concat(
    [x for x in [folds_all, folds_10, folds_20] if not x.empty],
    ignore_index=True
)

summary = pd.concat(
    [x for x in [summary_all, summary_10, summary_20] if not x.empty],
    ignore_index=True
)

folds.to_csv(
    OUT / "receiving_efficiency_cv_folds.csv",
    index=False
)

summary.to_csv(
    OUT / "receiving_efficiency_cv_summary.csv",
    index=False
)

print("\n" + "=" * 80)
print("CV SUMMARY")
print("=" * 80)

print(
    summary[
        [
            "population",
            "model",
            "n_players",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "mae_std",
            "r2_mean",
            "r2_std",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 80)
print("MODEL DIFFERENCES")
print("=" * 80)

for population in summary["population"].unique():
    s = summary[summary["population"] == population].set_index("model")

    if "A_conventional" not in s.index:
        continue

    base_rmse = s.loc["A_conventional", "rmse_mean"]
    base_mae = s.loc["A_conventional", "mae_mean"]

    print(f"\nPopulation: {population}")

    for model in ["B_movement", "C_combined"]:
        if model not in s.index:
            continue

        rmse = s.loc[model, "rmse_mean"]
        mae = s.loc[model, "mae_mean"]

        rmse_pct = 100 * (base_rmse - rmse) / base_rmse
        mae_pct = 100 * (base_mae - mae) / base_mae

        print(
            f"{model}: "
            f"RMSE change = {rmse - base_rmse:+.6f} "
            f"({rmse_pct:+.2f}% improvement), "
            f"MAE change = {mae - base_mae:+.6f} "
            f"({mae_pct:+.2f}% improvement)"
        )

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(f"\nSaved:")
print(f" - {OUT / 'receiving_efficiency_cv_folds.csv'}")
print(f" - {OUT / 'receiving_efficiency_cv_summary.csv'}")
