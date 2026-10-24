from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]

RESEARCH = ROOT / "research_outputs" / "wr_research_dataset.csv"
PLAYER_PLAY = ROOT / "data" / "player_play.csv"
PLAYERS = ROOT / "data" / "players.csv"
GAMES = ROOT / "data" / "games.csv"

OUT = ROOT / "research_outputs" / "receiving_efficiency_robustness"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "observed_yards_per_observed_target"

CONVENTIONAL = [
    "forty",
    "vertical"
]

MOVEMENT = [
    "speed_cv_DAGGER_ROUTE_LEFT"
]

COMBINED = [
    "forty",
    "vertical",
    "speed_cv_DAGGER_ROUTE_LEFT"
]


print("=" * 80)
print("RECEIVING-EFFICIENCY ROBUSTNESS ANALYSIS")
print("=" * 80)


# =====================================================================
# 1. RECONSTRUCT OUTCOME
# =====================================================================

research = pd.read_csv(RESEARCH)

players = pd.read_csv(
    PLAYERS,
    usecols=[
        "nfl_id",
        "draft_year",
        "nfl_position"
    ]
)

games = pd.read_csv(
    GAMES,
    usecols=[
        "game_id",
        "season",
        "season_type"
    ]
)

player_play = pd.read_csv(
    PLAYER_PLAY,
    usecols=[
        "game_id",
        "nfl_id",
        "target",
        "rec_yards"
    ]
)

research_ids = set(
    research["nfl_id"]
    .dropna()
    .astype(int)
)

players = players[
    players["nfl_id"].isin(research_ids)
    & players["nfl_position"].eq("WR")
].copy()

players["nfl_id"] = players["nfl_id"].astype(int)

player_play["nfl_id"] = pd.to_numeric(
    player_play["nfl_id"],
    errors="coerce"
)

player_play["game_id"] = pd.to_numeric(
    player_play["game_id"],
    errors="coerce"
)

player_play = player_play.dropna(
    subset=["nfl_id", "game_id"]
).copy()

player_play["nfl_id"] = player_play["nfl_id"].astype(int)
player_play["game_id"] = player_play["game_id"].astype(int)

df = player_play.merge(
    players[
        ["nfl_id", "draft_year"]
    ],
    on="nfl_id",
    how="inner",
    validate="many_to_one"
)

df = df.merge(
    games,
    on="game_id",
    how="left",
    validate="many_to_one"
)

df = df[
    (df["season_type"] == "REG")
    & (df["season"] >= df["draft_year"])
].copy()

df["target"] = pd.to_numeric(
    df["target"],
    errors="coerce"
)

df["rec_yards"] = pd.to_numeric(
    df["rec_yards"],
    errors="coerce"
)

target_rows = df[
    df["target"] == 1
].copy()

target_rows["rec_yards"] = target_rows[
    "rec_yards"
].fillna(0)

outcomes = (
    target_rows
    .groupby("nfl_id")
    .agg(
        observed_targets=("target", "count"),
        observed_receiving_yards=("rec_yards", "sum")
    )
    .reset_index()
)

outcomes[TARGET] = (
    outcomes["observed_receiving_yards"]
    / outcomes["observed_targets"]
)

data = research.merge(
    outcomes,
    on="nfl_id",
    how="left",
    validate="one_to_one"
)


# =====================================================================
# 2. LOCKED POPULATIONS
# =====================================================================

base = data.dropna(
    subset=[
        "forty",
        "vertical",
        "speed_cv_DAGGER_ROUTE_LEFT",
        TARGET
    ]
).copy()

populations = {
    "all_common_complete": base,
    "at_least_10_observed_targets": base[
        base["observed_targets"] >= 10
    ].copy(),
    "at_least_20_observed_targets": base[
        base["observed_targets"] >= 20
    ].copy()
}

print("\nLocked populations:")
for name, pop in populations.items():
    print(f"  {name}: {len(pop)} players")


# =====================================================================
# 3. COMMON CV FUNCTION
# =====================================================================

def make_predictions(
    population,
    features
):

    d = population[
        features + [TARGET]
    ].dropna().copy()

    X = d[features].to_numpy()
    y = d[TARGET].to_numpy()

    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(
        rkf.split(X),
        start=1
    ):

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ])

        model.fit(
            X[train_idx],
            y[train_idx]
        )

        pred = model.predict(
            X[test_idx]
        )

        rows.append({
            "fold": fold,
            "test_index": test_idx,
            "y_true": y[test_idx],
            "y_pred": pred
        })

    return d.reset_index(drop=True), rows


def metric_from_arrays(y, pred):

    return {
        "rmse": np.sqrt(
            mean_squared_error(y, pred)
        ),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred)
    }


# =====================================================================
# 4. PAIRED OOF PREDICTIONS
# =====================================================================

all_oof = {}

for population_name, population in populations.items():

    print(
        f"\nGenerating paired OOF predictions: "
        f"{population_name}"
    )

    # All models use exactly the same population.
    d = population[
        [
            "nfl_id",
            "draft_year",
            TARGET
        ]
        + CONVENTIONAL
        + MOVEMENT
    ].dropna().reset_index(drop=True)

    X_conv = d[CONVENTIONAL].to_numpy()
    X_move = d[MOVEMENT].to_numpy()
    X_comb = d[COMBINED].to_numpy()
    y = d[TARGET].to_numpy()

    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(
        rkf.split(X_conv),
        start=1
    ):

        models = {
            "A_conventional": (
                X_conv,
                CONVENTIONAL
            ),
            "B_movement": (
                X_move,
                MOVEMENT
            ),
            "C_combined": (
                X_comb,
                COMBINED
            )
        }

        preds = {}

        for model_name, (X, features) in models.items():

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0))
            ])

            model.fit(
                X[train_idx],
                y[train_idx]
            )

            preds[model_name] = model.predict(
                X[test_idx]
            )

        for i, idx in enumerate(test_idx):

            rows.append({
                "fold": fold,
                "nfl_id": d.iloc[idx]["nfl_id"],
                "draft_year": d.iloc[idx]["draft_year"],
                "y_true": y[idx],
                "A_pred": preds["A_conventional"][i],
                "B_pred": preds["B_movement"][i],
                "C_pred": preds["C_combined"][i]
            })

    oof = pd.DataFrame(rows)

    oof.to_csv(
        OUT /
        f"{population_name}_oof_predictions.csv",
        index=False
    )

    all_oof[population_name] = oof


# =====================================================================
# 5. PAIRED BOOTSTRAP
# =====================================================================

print("\n" + "=" * 80)
print("PAIRED BOOTSTRAP")
print("=" * 80)

rng = np.random.default_rng(42)

N_BOOT = 10000

bootstrap_summary = []
bootstrap_samples = []

for population_name, oof in all_oof.items():

    y = oof["y_true"].to_numpy()
    pred_a = oof["A_pred"].to_numpy()
    pred_b = oof["B_pred"].to_numpy()
    pred_c = oof["C_pred"].to_numpy()

    rmse_a = np.sqrt(
        mean_squared_error(y, pred_a)
    )
    rmse_b = np.sqrt(
        mean_squared_error(y, pred_b)
    )
    rmse_c = np.sqrt(
        mean_squared_error(y, pred_c)
    )

    mae_a = mean_absolute_error(y, pred_a)
    mae_b = mean_absolute_error(y, pred_b)
    mae_c = mean_absolute_error(y, pred_c)

    observed = {
        "B_vs_A_RMSE": rmse_b - rmse_a,
        "C_vs_A_RMSE": rmse_c - rmse_a,
        "B_vs_A_MAE": mae_b - mae_a,
        "C_vs_A_MAE": mae_c - mae_a
    }

    boot = {
        key: np.empty(N_BOOT)
        for key in observed
    }

    n = len(y)

    for b in range(N_BOOT):

        idx = rng.integers(
            0,
            n,
            size=n
        )

        yy = y[idx]
        aa = pred_a[idx]
        bb = pred_b[idx]
        cc = pred_c[idx]

        rmse_aa = np.sqrt(
            mean_squared_error(yy, aa)
        )
        rmse_bb = np.sqrt(
            mean_squared_error(yy, bb)
        )
        rmse_cc = np.sqrt(
            mean_squared_error(yy, cc)
        )

        mae_aa = mean_absolute_error(yy, aa)
        mae_bb = mean_absolute_error(yy, bb)
        mae_cc = mean_absolute_error(yy, cc)

        boot["B_vs_A_RMSE"][b] = (
            rmse_bb - rmse_aa
        )

        boot["C_vs_A_RMSE"][b] = (
            rmse_cc - rmse_aa
        )

        boot["B_vs_A_MAE"][b] = (
            mae_bb - mae_aa
        )

        boot["C_vs_A_MAE"][b] = (
            mae_cc - mae_aa
        )

    for metric_name, observed_value in observed.items():

        values = boot[metric_name]

        ci_low, ci_high = np.percentile(
            values,
            [2.5, 97.5]
        )

        # Negative difference means movement/combined
        # has lower error than conventional.
        p_improves = np.mean(
            values < 0
        )

        bootstrap_summary.append({
            "population": population_name,
            "comparison_metric": metric_name,
            "observed_difference": observed_value,
            "bootstrap_mean": values.mean(),
            "ci_2.5": ci_low,
            "ci_97.5": ci_high,
            "p_improves": p_improves
        })

        for value in values:
            bootstrap_samples.append({
                "population": population_name,
                "comparison_metric": metric_name,
                "difference": value
            })


bootstrap_summary = pd.DataFrame(
    bootstrap_summary
)

bootstrap_samples = pd.DataFrame(
    bootstrap_samples
)

bootstrap_summary.to_csv(
    OUT / "receiving_efficiency_bootstrap_summary.csv",
    index=False
)

bootstrap_samples.to_csv(
    OUT / "receiving_efficiency_bootstrap_samples.csv",
    index=False
)

print(
    bootstrap_summary.to_string(index=False)
)


# =====================================================================
# 6. LEAVE-ONE-DRAFT-YEAR-OUT
# =====================================================================

print("\n" + "=" * 80)
print("LEAVE-ONE-DRAFT-YEAR-OUT VALIDATION")
print("=" * 80)

year_rows = []

for population_name, population in populations.items():

    d = population[
        [
            "nfl_id",
            "draft_year",
            TARGET
        ]
        + CONVENTIONAL
        + MOVEMENT
    ].dropna().copy()

    for test_year in sorted(
        d["draft_year"].unique()
    ):

        train = d[
            d["draft_year"] != test_year
        ].copy()

        test = d[
            d["draft_year"] == test_year
        ].copy()

        if len(test) < 3:
            continue

        print(
            f"{population_name} | "
            f"test year {test_year} | "
            f"train {len(train)} | "
            f"test {len(test)}"
        )

        for model_name, features in [
            ("A_conventional", CONVENTIONAL),
            ("B_movement", MOVEMENT),
            ("C_combined", COMBINED)
        ]:

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0))
            ])

            model.fit(
                train[features],
                train[TARGET]
            )

            pred = model.predict(
                test[features]
            )

            metrics = metric_from_arrays(
                test[TARGET],
                pred
            )

            year_rows.append({
                "population": population_name,
                "test_year": int(test_year),
                "train_n": len(train),
                "test_n": len(test),
                "model": model_name,
                **metrics
            })


year_results = pd.DataFrame(year_rows)

year_results.to_csv(
    OUT /
    "receiving_efficiency_leave_one_year_out.csv",
    index=False
)

print("\n")
print(
    year_results.to_string(index=False)
)


# =====================================================================
# 7. FEATURE UNIQUENESS
# =====================================================================

print("\n" + "=" * 80)
print("FEATURE UNIQUENESS")
print("=" * 80)

feature_data = base[
    [
        "forty",
        "vertical",
        "speed_cv_DAGGER_ROUTE_LEFT",
        TARGET
    ]
].dropna().copy()

feature_pairs = [
    (
        "speed_cv_vs_forty",
        "speed_cv_DAGGER_ROUTE_LEFT",
        "forty"
    ),
    (
        "speed_cv_vs_vertical",
        "speed_cv_DAGGER_ROUTE_LEFT",
        "vertical"
    ),
    (
        "speed_cv_vs_receiving_efficiency",
        "speed_cv_DAGGER_ROUTE_LEFT",
        TARGET
    ),
    (
        "forty_vs_receiving_efficiency",
        "forty",
        TARGET
    ),
    (
        "vertical_vs_receiving_efficiency",
        "vertical",
        TARGET
    )
]

uniqueness_rows = []

for name, xcol, ycol in feature_pairs:

    x = feature_data[xcol]
    y = feature_data[ycol]

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    uniqueness_rows.append({
        "comparison": name,
        "n": len(feature_data),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p
    })


uniqueness = pd.DataFrame(
    uniqueness_rows
)

uniqueness.to_csv(
    OUT /
    "receiving_efficiency_feature_correlations.csv",
    index=False
)

print(
    uniqueness.to_string(index=False)
)


# =====================================================================
# 8. DESCRIPTIVE STANDARDIZED RIDGE COEFFICIENTS
# =====================================================================

print("\n" + "=" * 80)
print("STANDARDIZED FULL-SAMPLE COEFFICIENTS")
print("=" * 80)

X = feature_data[
    [
        "forty",
        "vertical",
        "speed_cv_DAGGER_ROUTE_LEFT"
    ]
]

y = feature_data[TARGET]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

model.fit(X, y)

coef = model.named_steps[
    "ridge"
].coef_

coef_table = pd.DataFrame({
    "feature": X.columns,
    "standardized_ridge_coefficient": coef
})

print(
    coef_table.to_string(index=False)
)

coef_table.to_csv(
    OUT /
    "receiving_efficiency_standardized_coefficients.csv",
    index=False
)


# =====================================================================
# 9. FINAL FILE LIST
# =====================================================================

print("\n" + "=" * 80)
print("SAVED FILES")
print("=" * 80)

for p in sorted(OUT.glob("*.csv")):
    print(p)

print("\nDONE")
