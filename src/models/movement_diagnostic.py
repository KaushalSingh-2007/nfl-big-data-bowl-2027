from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "research_outputs"
DATA_FILE = OUT / "wr_research_dataset.csv"
RESULTS_DIR = OUT / "model_comparison" / "movement_diagnostic"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_model(alpha):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def main():

    print("=" * 90)
    print("MOVEMENT FEATURE DIAGNOSTIC")
    print("=" * 90)

    df = pd.read_csv(DATA_FILE)

    target = "mean_separation"

    movement_features = [
        "peak_speed_DAGGER_ROUTE_LEFT",
        "speed_cv_DAGGER_ROUTE_LEFT",
        "straightness_ratio_DAGGER_ROUTE_LEFT",
        "late_early_speed_ratio_DAGGER_ROUTE_LEFT",
    ]

    combine_features = [
        "forty",
        "vertical",
    ]

    # All variables needed for the diagnostic are included here.
    all_features = combine_features + movement_features

    common = df[all_features + [target]].dropna().copy()

    print(f"\nCommon population: {len(common)} players")

    if len(common) != 64:
        print(
            f"WARNING: expected 64 players, found {len(common)}."
        )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    feature_sets = {
        "peak_speed_only": [
            "peak_speed_DAGGER_ROUTE_LEFT"
        ],

        "speed_cv_only": [
            "speed_cv_DAGGER_ROUTE_LEFT"
        ],

        "straightness_only": [
            "straightness_ratio_DAGGER_ROUTE_LEFT"
        ],

        "late_early_speed_only": [
            "late_early_speed_ratio_DAGGER_ROUTE_LEFT"
        ],

        "peak_speed_plus_speed_cv": [
            "peak_speed_DAGGER_ROUTE_LEFT",
            "speed_cv_DAGGER_ROUTE_LEFT",
        ],

        "peak_speed_plus_straightness": [
            "peak_speed_DAGGER_ROUTE_LEFT",
            "straightness_ratio_DAGGER_ROUTE_LEFT",
        ],

        "speed_cv_plus_straightness": [
            "speed_cv_DAGGER_ROUTE_LEFT",
            "straightness_ratio_DAGGER_ROUTE_LEFT",
        ],

        "all_four_movement": movement_features,
    }

    rows = []

    for feature_set_name, features in feature_sets.items():

        for alpha in [0.1, 1.0, 10.0, 100.0]:

            X = common[features]
            y = common[target]

            model = build_model(alpha)

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

            rows.append(
                {
                    "feature_set": feature_set_name,
                    "features": ", ".join(features),
                    "alpha": alpha,
                    "n": len(X),
                    "cv_rmse_mean": rmse.mean(),
                    "cv_rmse_std": rmse.std(ddof=1),
                    "cv_mae_mean": mae.mean(),
                    "cv_mae_std": mae.std(ddof=1),
                    "cv_r2_mean": r2.mean(),
                    "cv_r2_std": r2.std(ddof=1),
                }
            )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        ["cv_rmse_mean", "cv_mae_mean"]
    )

    print("\n" + "=" * 90)
    print("BEST MOVEMENT CONFIGURATIONS BY CV RMSE")
    print("=" * 90)

    display_cols = [
        "feature_set",
        "alpha",
        "cv_rmse_mean",
        "cv_rmse_std",
        "cv_mae_mean",
        "cv_r2_mean",
    ]

    print(
        results[display_cols]
        .head(15)
        .to_string(index=False)
    )

    best_per_feature_set = (
        results
        .sort_values("cv_rmse_mean")
        .groupby("feature_set", as_index=False)
        .first()
        .sort_values("cv_rmse_mean")
    )

    print("\n" + "=" * 90)
    print("BEST ALPHA WITHIN EACH MOVEMENT FEATURE SET")
    print("=" * 90)

    print(
        best_per_feature_set[display_cols]
        .to_string(index=False)
    )

    results_file = RESULTS_DIR / "movement_feature_diagnostic.csv"
    best_file = RESULTS_DIR / "best_movement_configurations.csv"

    results.to_csv(results_file, index=False)
    best_per_feature_set.to_csv(best_file, index=False)

    print("\n" + "=" * 90)
    print("FILES CREATED")
    print("=" * 90)

    print(results_file)
    print(best_file)

    print("\nMOVEMENT DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()
