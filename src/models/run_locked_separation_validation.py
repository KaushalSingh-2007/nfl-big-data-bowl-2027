from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

INPUT = (
    PROJECT_ROOT
    / "research_outputs"
    / "wr_research_dataset.csv"
)

OUT = (
    PROJECT_ROOT
    / "research_outputs"
    / "model_comparison"
    / "locked_validation"
)

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOCKED EXPERIMENT
# ============================================================

TARGET = "mean_separation"

FEATURES = {
    "A_conventional": [
        "forty",
        "vertical"
    ],
    "B_movement": [
        "speed_cv_DAGGER_ROUTE_LEFT"
    ],
    "C_combined": [
        "forty",
        "vertical",
        "speed_cv_DAGGER_ROUTE_LEFT"
    ]
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOCKED SEPARATION VALIDATION")
print("=" * 70)

print()
print("Loading research dataset...")

df = pd.read_csv(INPUT)

print(f"Dataset shape: {df.shape}")

required_columns = (
    [TARGET]
    + sorted(
        set(
            feature
            for features in FEATURES.values()
            for feature in features
        )
    )
    + ["draft_year", "nfl_id"]
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# SAME COMMON POPULATION
# ============================================================

all_features = sorted(
    set(
        feature
        for features in FEATURES.values()
        for feature in features
    )
)

common = df[
    df[TARGET].notna()
    & df[all_features].notna().all(axis=1)
].copy()

common = common.reset_index(drop=True)

print()
print("Common population:")
print(f"Players: {len(common)}")
print(
    "Draft years:",
    common["draft_year"].value_counts().sort_index().to_dict()
)

print()
print("Features used:")
for model_name, features in FEATURES.items():
    print(f"{model_name}: {features}")


if len(common) < 10:
    raise ValueError(
        "Too few complete observations for validation."
    )


# ============================================================
# MODEL
# ============================================================

def make_model():
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge(alpha=1.0)
            )
        ]
    )


# ============================================================
# REPEATED 5-FOLD CV
# ============================================================

print()
print("=" * 70)
print("10 x 5-FOLD REPEATED CROSS-VALIDATION")
print("=" * 70)

X_all = common[all_features]
y = common[TARGET]

cv = RepeatedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

fold_results = []

for split_id, (train_idx, test_idx) in enumerate(
    cv.split(X_all),
    start=1
):

    for model_name, feature_list in FEATURES.items():

        X_train = common.iloc[train_idx][feature_list]
        X_test = common.iloc[test_idx][feature_list]

        y_train = common.iloc[train_idx][TARGET]
        y_test = common.iloc[test_idx][TARGET]

        model = make_model()

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(X_test)

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                prediction
            )
        )

        mae = mean_absolute_error(
            y_test,
            prediction
        )

        r2 = r2_score(
            y_test,
            prediction
        )

        repeat_number = (
            (split_id - 1) // 5
        ) + 1

        fold_number = (
            (split_id - 1) % 5
        ) + 1

        fold_results.append(
            {
                "split_id": split_id,
                "repeat": repeat_number,
                "fold": fold_number,
                "model": model_name,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
        )


fold_results = pd.DataFrame(
    fold_results
)

fold_output = OUT / "repeated_cv_fold_results.csv"

fold_results.to_csv(
    fold_output,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    fold_results
    .groupby("model")
    .agg(
        n_folds=("split_id", "count"),

        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),

        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),

        r2_mean=("r2", "mean"),
        r2_std=("r2", "std")
    )
    .reset_index()
)

summary_output = OUT / "repeated_cv_summary.csv"

summary.to_csv(
    summary_output,
    index=False
)


print()
print("Repeated CV summary:")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# PAIRED FOLD COMPARISONS
# ============================================================

print()
print("=" * 70)
print("PAIRED FOLD COMPARISONS")
print("=" * 70)

wide = fold_results.pivot(
    index="split_id",
    columns="model",
    values=[
        "rmse",
        "mae",
        "r2"
    ]
)

wide.columns = [
    f"{metric}_{model}"
    for metric, model in wide.columns
]

wide = wide.reset_index()


wide["rmse_B_minus_A"] = (
    wide["rmse_B_movement"]
    - wide["rmse_A_conventional"]
)

wide["rmse_C_minus_A"] = (
    wide["rmse_C_combined"]
    - wide["rmse_A_conventional"]
)

wide["mae_B_minus_A"] = (
    wide["mae_B_movement"]
    - wide["mae_A_conventional"]
)

wide["mae_C_minus_A"] = (
    wide["mae_C_combined"]
    - wide["mae_A_conventional"]
)

wide["r2_B_minus_A"] = (
    wide["r2_B_movement"]
    - wide["r2_A_conventional"]
)

wide["r2_C_minus_A"] = (
    wide["r2_C_combined"]
    - wide["r2_A_conventional"]
)


paired_output = OUT / "paired_fold_comparisons.csv"

wide.to_csv(
    paired_output,
    index=False
)


paired_summary = pd.DataFrame(
    [
        {
            "comparison": "B_movement_vs_A_conventional",
            "rmse_difference_mean": wide[
                "rmse_B_minus_A"
            ].mean(),
            "rmse_difference_std": wide[
                "rmse_B_minus_A"
            ].std(),
            "mae_difference_mean": wide[
                "mae_B_minus_A"
            ].mean(),
            "r2_difference_mean": wide[
                "r2_B_minus_A"
            ].mean(),
            "folds_rmse_improved": (
                wide["rmse_B_minus_A"] < 0
            ).sum(),
            "folds_rmse_worse": (
                wide["rmse_B_minus_A"] > 0
            ).sum()
        },
        {
            "comparison": "C_combined_vs_A_conventional",
            "rmse_difference_mean": wide[
                "rmse_C_minus_A"
            ].mean(),
            "rmse_difference_std": wide[
                "rmse_C_minus_A"
            ].std(),
            "mae_difference_mean": wide[
                "mae_C_minus_A"
            ].mean(),
            "r2_difference_mean": wide[
                "r2_C_minus_A"
            ].mean(),
            "folds_rmse_improved": (
                wide["rmse_C_minus_A"] < 0
            ).sum(),
            "folds_rmse_worse": (
                wide["rmse_C_minus_A"] > 0
            ).sum()
        }
    ]
)

paired_summary_output = (
    OUT / "paired_comparison_summary.csv"
)

paired_summary.to_csv(
    paired_summary_output,
    index=False
)

print()
print(
    paired_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# LEAVE-ONE-DRAFT-YEAR-OUT
# ============================================================

print()
print("=" * 70)
print("LEAVE-ONE-DRAFT-YEAR-OUT VALIDATION")
print("=" * 70)

year_results = []

years = sorted(
    common["draft_year"].unique()
)

for test_year in years:

    train = common[
        common["draft_year"] != test_year
    ].copy()

    test = common[
        common["draft_year"] == test_year
    ].copy()

    print()
    print(
        f"Test year: {test_year} | "
        f"train={len(train)} | "
        f"test={len(test)}"
    )

    for model_name, feature_list in FEATURES.items():

        model = make_model()

        model.fit(
            train[feature_list],
            train[TARGET]
        )

        prediction = model.predict(
            test[feature_list]
        )

        rmse = np.sqrt(
            mean_squared_error(
                test[TARGET],
                prediction
            )
        )

        mae = mean_absolute_error(
            test[TARGET],
            prediction
        )

        r2 = r2_score(
            test[TARGET],
            prediction
        )

        year_results.append(
            {
                "test_year": test_year,
                "model": model_name,
                "n_train": len(train),
                "n_test": len(test),
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
        )


year_results = pd.DataFrame(
    year_results
)

year_output = (
    OUT / "leave_one_draft_year_out_results.csv"
)

year_results.to_csv(
    year_output,
    index=False
)

print()
print(
    year_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# YEAR-LEVEL COMPARISONS
# ============================================================

year_wide = year_results.pivot(
    index="test_year",
    columns="model",
    values=[
        "rmse",
        "mae",
        "r2"
    ]
)

year_wide.columns = [
    f"{metric}_{model}"
    for metric, model in year_wide.columns
]

year_wide = year_wide.reset_index()

year_wide["rmse_B_minus_A"] = (
    year_wide["rmse_B_movement"]
    - year_wide["rmse_A_conventional"]
)

year_wide["rmse_C_minus_A"] = (
    year_wide["rmse_C_combined"]
    - year_wide["rmse_A_conventional"]
)

year_wide["mae_B_minus_A"] = (
    year_wide["mae_B_movement"]
    - year_wide["mae_A_conventional"]
)

year_wide["mae_C_minus_A"] = (
    year_wide["mae_C_combined"]
    - year_wide["mae_A_conventional"]
)

year_wide["r2_B_minus_A"] = (
    year_wide["r2_B_movement"]
    - year_wide["r2_A_conventional"]
)

year_wide["r2_C_minus_A"] = (
    year_wide["r2_C_combined"]
    - year_wide["r2_A_conventional"]
)

year_comparison_output = (
    OUT / "leave_one_year_comparisons.csv"
)

year_wide.to_csv(
    year_comparison_output,
    index=False
)


# ============================================================
# OVERALL PERCENTAGE CHANGE
# ============================================================

lookup = summary.set_index("model")

baseline_rmse = lookup.loc[
    "A_conventional",
    "rmse_mean"
]

movement_rmse = lookup.loc[
    "B_movement",
    "rmse_mean"
]

combined_rmse = lookup.loc[
    "C_combined",
    "rmse_mean"
]

baseline_mae = lookup.loc[
    "A_conventional",
    "mae_mean"
]

movement_mae = lookup.loc[
    "B_movement",
    "mae_mean"
]

combined_mae = lookup.loc[
    "C_combined",
    "mae_mean"
]

overall = pd.DataFrame(
    [
        {
            "comparison": "Movement_vs_Conventional",
            "rmse_change_percent": (
                (baseline_rmse - movement_rmse)
                / baseline_rmse
                * 100
            ),
            "mae_change_percent": (
                (baseline_mae - movement_mae)
                / baseline_mae
                * 100
            )
        },
        {
            "comparison": "Combined_vs_Conventional",
            "rmse_change_percent": (
                (baseline_rmse - combined_rmse)
                / baseline_rmse
                * 100
            ),
            "mae_change_percent": (
                (baseline_mae - combined_mae)
                / baseline_mae
                * 100
            )
        }
    ]
)

overall_output = (
    OUT / "overall_percentage_changes.csv"
)

overall.to_csv(
    overall_output,
    index=False
)


# ============================================================
# FINAL CONSOLE SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL LOCKED VALIDATION SUMMARY")
print("=" * 70)

print()

print(
    f"A Conventional RMSE: "
    f"{baseline_rmse:.4f}"
)

print(
    f"B Movement RMSE:     "
    f"{movement_rmse:.4f}"
)

print(
    f"C Combined RMSE:     "
    f"{combined_rmse:.4f}"
)

print()

print(
    "Movement vs Conventional RMSE improvement: "
    f"{overall.loc[0, 'rmse_change_percent']:.2f}%"
)

print(
    "Combined vs Conventional RMSE improvement: "
    f"{overall.loc[1, 'rmse_change_percent']:.2f}%"
)

print()
print("Files saved to:")
print(OUT)
print()
print("Validation complete.")
