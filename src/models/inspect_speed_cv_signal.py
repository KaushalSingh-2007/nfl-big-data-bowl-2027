from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "research_outputs"
DATA = OUT / "wr_research_dataset.csv"
SIGNAL_OUT = OUT / "model_comparison" / "speed_cv_signal"
SIGNAL_OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

features = [
    "forty",
    "vertical",
    "speed_cv_DAGGER_ROUTE_LEFT",
    "mean_separation",
]

common = df[features].dropna().copy()

print("=" * 70)
print("SPEED CV SIGNAL INSPECTION")
print("=" * 70)

print(f"\nCommon complete population: {len(common)} players")

speed = common["speed_cv_DAGGER_ROUTE_LEFT"]

print("\n--- SPEED CV DESCRIPTIVE STATISTICS ---")
print(speed.describe().to_string())

print("\n--- SPEED CV QUANTILES ---")
print(
    speed.quantile(
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).to_string()
)

pearson = common.corr(method="pearson")
spearman = common.corr(method="spearman")

print("\n--- CORRELATION OF SPEED CV WITH OTHER VARIABLES ---")

for col in ["forty", "vertical", "mean_separation"]:
    print(
        f"{col:20s} "
        f"Pearson = {pearson.loc['speed_cv_DAGGER_ROUTE_LEFT', col]: .4f}   "
        f"Spearman = {spearman.loc['speed_cv_DAGGER_ROUTE_LEFT', col]: .4f}"
    )

print("\n--- CORRELATION WITH MEAN SEPARATION ---")

for col in ["forty", "vertical", "speed_cv_DAGGER_ROUTE_LEFT"]:
    print(
        f"{col:35s} "
        f"Pearson = {pearson.loc['mean_separation', col]: .4f}   "
        f"Spearman = {spearman.loc['mean_separation', col]: .4f}"
    )

print("\n--- STANDARDIZED RIDGE: SPEED CV ONLY ---")

X_speed = common[["speed_cv_DAGGER_ROUTE_LEFT"]].values
y = common["mean_separation"].values

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_speed_std = scaler_x.fit_transform(X_speed)
y_std = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

ridge_speed = Ridge(alpha=1.0)
ridge_speed.fit(X_speed_std, y_std)

print(
    f"Standardized coefficient: "
    f"{ridge_speed.coef_[0]: .4f}"
)

print(
    f"Descriptive training R2: "
    f"{ridge_speed.score(X_speed_std, y_std): .4f}"
)

print("\n--- STANDARDIZED RIDGE: 40 YD + VERTICAL + SPEED CV ---")

model_features = [
    "forty",
    "vertical",
    "speed_cv_DAGGER_ROUTE_LEFT",
]

X = common[model_features].values
X_std = StandardScaler().fit_transform(X)

ridge_combined = Ridge(alpha=1.0)
ridge_combined.fit(X_std, y_std)

for feature, coef in zip(model_features, ridge_combined.coef_):
    print(f"{feature:35s}: {coef: .4f}")

print(
    f"Descriptive training R2: "
    f"{ridge_combined.score(X_std, y_std): .4f}"
)

print("\n--- LOWEST 5 SPEED CV VALUES ---")

lowest = common[
    [
        "speed_cv_DAGGER_ROUTE_LEFT",
        "forty",
        "vertical",
        "mean_separation",
    ]
].sort_values("speed_cv_DAGGER_ROUTE_LEFT")

print(lowest.head(5).to_string(index=False))

print("\n--- HIGHEST 5 SPEED CV VALUES ---")

print(lowest.tail(5).to_string(index=False))

print("\n--- OUTLIER CHECK ---")

z = (speed - speed.mean()) / speed.std(ddof=1)

outliers = common.loc[
    z.abs() > 3,
    [
        "speed_cv_DAGGER_ROUTE_LEFT",
        "forty",
        "vertical",
        "mean_separation",
    ],
]

print(f"Players with |z| > 3: {len(outliers)}")

if len(outliers) > 0:
    print(outliers.to_string(index=False))
else:
    print("No |z| > 3 outliers found.")

pearson.to_csv(
    SIGNAL_OUT / "correlations_pearson.csv"
)

spearman.to_csv(
    SIGNAL_OUT / "correlations_spearman.csv"
)

common.to_csv(
    SIGNAL_OUT / "speed_cv_common_population.csv",
    index=False
)

coef_df = pd.DataFrame(
    {
        "feature": model_features,
        "standardized_ridge_coefficient": ridge_combined.coef_,
    }
)

coef_df.to_csv(
    SIGNAL_OUT / "standardized_ridge_coefficients.csv",
    index=False
)

summary = pd.DataFrame(
    {
        "metric": [
            "n",
            "speed_cv_mean",
            "speed_cv_median",
            "speed_cv_std",
            "speed_cv_min",
            "speed_cv_max",
            "speed_cv_pearson_with_40",
            "speed_cv_spearman_with_40",
            "speed_cv_pearson_with_vertical",
            "speed_cv_spearman_with_vertical",
            "speed_cv_pearson_with_mean_separation",
            "speed_cv_spearman_with_mean_separation",
            "speed_only_standardized_ridge_coefficient",
            "speed_only_training_r2",
        ],
        "value": [
            len(common),
            speed.mean(),
            speed.median(),
            speed.std(),
            speed.min(),
            speed.max(),
            pearson.loc[
                "speed_cv_DAGGER_ROUTE_LEFT",
                "forty"
            ],
            spearman.loc[
                "speed_cv_DAGGER_ROUTE_LEFT",
                "forty"
            ],
            pearson.loc[
                "speed_cv_DAGGER_ROUTE_LEFT",
                "vertical"
            ],
            spearman.loc[
                "speed_cv_DAGGER_ROUTE_LEFT",
                "vertical"
            ],
            pearson.loc[
                "speed_cv_DAGGER_ROUTE_LEFT",
                "mean_separation"
            ],
            spearman.loc[
                "speed_cv_DAGGER_ROUTE_LEFT",
                "mean_separation"
            ],
            ridge_speed.coef_[0],
            ridge_speed.score(X_speed_std, y_std),
        ],
    }
)

summary.to_csv(
    SIGNAL_OUT / "speed_cv_signal_summary.csv",
    index=False
)

print("\n" + "=" * 70)
print("STAGE 1 COMPLETE")
print("=" * 70)

print("\nOutput folder:")
print(SIGNAL_OUT)

print("\nFiles created:")

for file in sorted(SIGNAL_OUT.glob("*")):
    print(" - " + file.name)

print("\nSend me the PowerShell output above.")
