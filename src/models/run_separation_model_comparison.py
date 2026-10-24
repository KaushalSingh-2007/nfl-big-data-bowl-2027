from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "research_outputs"
RESULTS_DIR = DATA_DIR / "model_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "wr_research_dataset.csv"


def build_model():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def main():
    print("=" * 90)
    print("NFL BIG DATA BOWL 2027")
    print("SEPARATION MODEL COMPARISON")
    print("=" * 90)

    df = pd.read_csv(DATA_FILE)

    target = "mean_separation"

    model_specs = {
        "baseline_40_vertical": [
            "forty",
            "vertical",
        ],
        "movement_only": [
            "peak_speed_DAGGER_ROUTE_LEFT",
            "speed_cv_DAGGER_ROUTE_LEFT",
        ],
        "combined": [
            "forty",
            "vertical",
            "peak_speed_DAGGER_ROUTE_LEFT",
            "speed_cv_DAGGER_ROUTE_LEFT",
        ],
    }

    # ------------------------------------------------------------------
    # FAIR COMPARISON POPULATION
    # ------------------------------------------------------------------
    # Every model is evaluated on the same 64 players:
    # all players with the baseline + movement + outcome available.
    # This prevents sample-population differences from driving the result.
    common_features = sorted(
        set(sum(model_specs.values(), []))
    )

    common_cols = common_features + [target]

    common_df = df[common_cols].dropna().copy()

    print(f"\nDataset: {DATA_FILE}")
    print(f"Common comparison population: {len(common_df)} players")

    if len(common_df) != 64:
        print(
            "\nWARNING: Expected 64 common players based on the previous "
            f"audit, but found {len(common_df)}."
        )

    X_all = common_df.drop(columns=[target])
    y = common_df[target]

    # ------------------------------------------------------------------
    # CROSS-VALIDATION
    # ------------------------------------------------------------------
    # 5-fold CV is used because the dataset is small.
    # shuffle=True with a fixed random_state makes the result reproducible.
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    rows = []

    for model_name, features in model_specs.items():

        X = common_df[features]

        model = build_model()

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            return_train_score=False,
        )

        rmse = -scores["test_rmse"]
        mae = -scores["test_mae"]
        r2 = scores["test_r2"]

        row = {
            "model": model_name,
            "n": len(X),
            "features": ", ".join(features),
            "cv_rmse_mean": rmse.mean(),
            "cv_rmse_std": rmse.std(ddof=1),
            "cv_mae_mean": mae.mean(),
            "cv_mae_std": mae.std(ddof=1),
            "cv_r2_mean": r2.mean(),
            "cv_r2_std": r2.std(ddof=1),
        }

        rows.append(row)

        print("\n" + "-" * 90)
        print(model_name)
        print("-" * 90)
        print("Features:", ", ".join(features))
        print("n:", len(X))
        print(f"CV RMSE: {rmse.mean():.4f} +/- {rmse.std(ddof=1):.4f}")
        print(f"CV MAE : {mae.mean():.4f} +/- {mae.std(ddof=1):.4f}")
        print(f"CV R2  : {r2.mean():.4f} +/- {r2.std(ddof=1):.4f}")

    results = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # INCREMENTAL VALUE OF MOVEMENT
    # ------------------------------------------------------------------
    baseline_rmse = results.loc[
        results["model"] == "baseline_40_vertical",
        "cv_rmse_mean",
    ].iloc[0]

    combined_rmse = results.loc[
        results["model"] == "combined",
        "cv_rmse_mean",
    ].iloc[0]

    baseline_mae = results.loc[
        results["model"] == "baseline_40_vertical",
        "cv_mae_mean",
    ].iloc[0]

    combined_mae = results.loc[
        results["model"] == "combined",
        "cv_mae_mean",
    ].iloc[0]

    baseline_r2 = results.loc[
        results["model"] == "baseline_40_vertical",
        "cv_r2_mean",
    ].iloc[0]

    combined_r2 = results.loc[
        results["model"] == "combined",
        "cv_r2_mean",
    ].iloc[0]

    rmse_improvement_pct = (
        (baseline_rmse - combined_rmse)
        / baseline_rmse
        * 100
    )

    mae_improvement_pct = (
        (baseline_mae - combined_mae)
        / baseline_mae
        * 100
    )

    r2_change = combined_r2 - baseline_r2

    print("\n" + "=" * 90)
    print("INCREMENTAL VALUE OF MOVEMENT FEATURES")
    print("=" * 90)

    print(f"Baseline CV RMSE : {baseline_rmse:.4f}")
    print(f"Combined CV RMSE : {combined_rmse:.4f}")
    print(f"RMSE improvement : {rmse_improvement_pct:.2f}%")

    print()

    print(f"Baseline CV MAE  : {baseline_mae:.4f}")
    print(f"Combined CV MAE  : {combined_mae:.4f}")
    print(f"MAE improvement  : {mae_improvement_pct:.2f}%")

    print()

    print(f"Baseline CV R2   : {baseline_r2:.4f}")
    print(f"Combined CV R2   : {combined_r2:.4f}")
    print(f"R2 change        : {r2_change:+.4f}")

    # ------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------
    results_file = RESULTS_DIR / "cv_model_comparison.csv"
    results.to_csv(results_file, index=False)

    summary = pd.DataFrame(
        [
            {
                "comparison_population_n": len(common_df),
                "baseline_model": "40 yd + vertical",
                "movement_model": "Dagger peak speed + Dagger speed CV",
                "combined_model": (
                    "40 yd + vertical + Dagger peak speed + Dagger speed CV"
                ),
                "baseline_rmse": baseline_rmse,
                "combined_rmse": combined_rmse,
                "rmse_improvement_pct": rmse_improvement_pct,
                "baseline_mae": baseline_mae,
                "combined_mae": combined_mae,
                "mae_improvement_pct": mae_improvement_pct,
                "baseline_r2": baseline_r2,
                "combined_r2": combined_r2,
                "r2_change": r2_change,
            }
        ]
    )

    summary_file = RESULTS_DIR / "incremental_value_summary.csv"
    summary.to_csv(summary_file, index=False)

    population_file = RESULTS_DIR / "model_comparison_population.csv"
    common_df.insert(0, "nfl_id", df.loc[common_df.index, "nfl_id"])
    common_df.to_csv(population_file, index=False)

    print("\n" + "=" * 90)
    print("FILES CREATED")
    print("=" * 90)
    print(results_file)
    print(summary_file)
    print(population_file)

    print("\nMODEL COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
