from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_outputs"

PRED_FILE = (
    OUT
    / "nonlinear_final"
    / "final_nonlinear_predictions.csv"
)

SAVE_DIR = OUT / "nonlinear_final"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

if not PRED_FILE.exists():
    raise FileNotFoundError(PRED_FILE)

pred = pd.read_csv(PRED_FILE)

print("=" * 70)
print("FINAL NONLINEAR PAIRED BOOTSTRAP")
print("=" * 70)

print(f"Prediction rows: {len(pred)}")
print(f"Columns: {list(pred.columns)}")


# ============================================================
# Important:
#
# Each player appears repeatedly because we used 5-fold x
# 10-repeat CV.
#
# We first average each player's absolute error across all
# repeated held-out predictions for each model.
#
# Bootstrap is then performed across players, preserving the
# paired comparison between models.
# ============================================================

pred["abs_error"] = (
    pred["prediction"] -
    pred["y_true"]
).abs()


def prepare_player_errors(
    df,
    model_a,
    model_b,
):

    sub = df[
        df["model"].isin(
            [model_a, model_b]
        )
    ].copy()

    player_errors = (
        sub
        .groupby(
            ["nfl_id", "model"]
        )["abs_error"]
        .mean()
        .reset_index()
    )

    pivot = (
        player_errors
        .pivot(
            index="nfl_id",
            columns="model",
            values="abs_error",
        )
        .dropna(
            subset=[
                model_a,
                model_b,
            ]
        )
    )

    return pivot


def bootstrap_difference(
    pivot,
    model_a,
    model_b,
    n_boot=10000,
    seed=42,
):

    rng = np.random.default_rng(seed)

    a = pivot[model_a].to_numpy()
    b = pivot[model_b].to_numpy()

    n = len(a)

    observed = np.mean(b - a)

    boot = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot[i] = np.mean(
            b[idx] - a[idx]
        )

    ci_low, ci_high = np.percentile(
        boot,
        [2.5, 97.5],
    )

    # Negative difference means model_b has
    # lower absolute error than model_a.
    probability_improves = np.mean(
        boot < 0
    )

    return {
        "n_players": n,
        "observed_difference": observed,
        "bootstrap_mean": boot.mean(),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_improves": probability_improves,
    }


# ============================================================
# Run comparisons
# ============================================================

comparisons = [
    (
        "poly_conventional",
        "poly_movement",
        "Polynomial movement vs polynomial conventional",
    ),
    (
        "linear_movement",
        "poly_movement",
        "Polynomial movement vs linear movement",
    ),
    (
        "linear_combined",
        "poly_combined",
        "Polynomial combined vs linear combined",
    ),
]


all_results = []


for threshold in [10, 20]:

    sub = pred[
        pred["threshold"] == threshold
    ].copy()

    print("\n" + "-" * 70)
    print(
        f"THRESHOLD >= {threshold}"
    )
    print("-" * 70)

    for model_a, model_b, label in comparisons:

        pivot = prepare_player_errors(
            sub,
            model_a,
            model_b,
        )

        result = bootstrap_difference(
            pivot,
            model_a,
            model_b,
            n_boot=10000,
            seed=42,
        )

        result_row = {
            "threshold": threshold,
            "model_a": model_a,
            "model_b": model_b,
            "comparison": label,
            **result,
        }

        all_results.append(
            result_row
        )

        print(
            f"\n{label}"
        )

        print(
            f"Players: "
            f"{result['n_players']}"
        )

        print(
            f"Observed MAE difference "
            f"(B - A): "
            f"{result['observed_difference']:+.5f}"
        )

        print(
            f"Bootstrap mean: "
            f"{result['bootstrap_mean']:+.5f}"
        )

        print(
            f"95% CI: "
            f"[{result['ci_low']:+.5f}, "
            f"{result['ci_high']:+.5f}]"
        )

        print(
            f"P(B improves): "
            f"{result['p_improves']:.4f}"
        )


results_df = pd.DataFrame(
    all_results
)

results_path = (
    SAVE_DIR
    / "nonlinear_paired_bootstrap_results.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)


# ============================================================
# RMSE bootstrap
#
# MAE above uses player-level average absolute error.
#
# For RMSE, we use the repeated prediction errors directly,
# but resample players so all repeated observations belonging
# to a selected player stay together.
# ============================================================

def prepare_player_squared_errors(
    df,
    model_a,
    model_b,
):

    sub = df[
        df["model"].isin(
            [model_a, model_b]
        )
    ].copy()

    sub["squared_error"] = (
        sub["prediction"] -
        sub["y_true"]
    ) ** 2

    player_model = (
        sub
        .groupby(
            ["nfl_id", "model"]
        )["squared_error"]
        .mean()
        .reset_index()
    )

    pivot = (
        player_model
        .pivot(
            index="nfl_id",
            columns="model",
            values="squared_error",
        )
        .dropna(
            subset=[
                model_a,
                model_b,
            ]
        )
    )

    return pivot


def bootstrap_rmse_difference(
    pivot,
    model_a,
    model_b,
    n_boot=10000,
    seed=42,
):

    rng = np.random.default_rng(seed)

    a = pivot[model_a].to_numpy()
    b = pivot[model_b].to_numpy()

    n = len(a)

    observed = (
        np.sqrt(
            np.mean(b)
        )
        -
        np.sqrt(
            np.mean(a)
        )
    )

    boot = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        rmse_a = np.sqrt(
            np.mean(
                a[idx]
            )
        )

        rmse_b = np.sqrt(
            np.mean(
                b[idx]
            )
        )

        boot[i] = (
            rmse_b -
            rmse_a
        )

    ci_low, ci_high = np.percentile(
        boot,
        [2.5, 97.5],
    )

    probability_improves = np.mean(
        boot < 0
    )

    return {
        "n_players": n,
        "observed_difference": observed,
        "bootstrap_mean": boot.mean(),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_improves": probability_improves,
    }


rmse_results = []


for threshold in [10, 20]:

    sub = pred[
        pred["threshold"] == threshold
    ].copy()

    print("\n" + "=" * 70)
    print(
        f"RMSE BOOTSTRAP — "
        f"THRESHOLD >= {threshold}"
    )
    print("=" * 70)

    for model_a, model_b, label in comparisons:

        pivot = prepare_player_squared_errors(
            sub,
            model_a,
            model_b,
        )

        result = bootstrap_rmse_difference(
            pivot,
            model_a,
            model_b,
            n_boot=10000,
            seed=42,
        )

        rmse_row = {
            "threshold": threshold,
            "model_a": model_a,
            "model_b": model_b,
            "comparison": label,
            **result,
        }

        rmse_results.append(
            rmse_row
        )

        print(
            f"\n{label}"
        )

        print(
            f"Players: "
            f"{result['n_players']}"
        )

        print(
            f"Observed RMSE difference "
            f"(B - A): "
            f"{result['observed_difference']:+.5f}"
        )

        print(
            f"Bootstrap mean: "
            f"{result['bootstrap_mean']:+.5f}"
        )

        print(
            f"95% CI: "
            f"[{result['ci_low']:+.5f}, "
            f"{result['ci_high']:+.5f}]"
        )

        print(
            f"P(B improves): "
            f"{result['p_improves']:.4f}"
        )


rmse_df = pd.DataFrame(
    rmse_results
)

rmse_path = (
    SAVE_DIR
    / "nonlinear_paired_bootstrap_rmse_results.csv"
)

rmse_df.to_csv(
    rmse_path,
    index=False,
)


# ============================================================
# Final decision summary
# ============================================================

print("\n" + "=" * 70)
print("FINAL DECISION CHECK")
print("=" * 70)

for threshold in [10, 20]:

    rows = rmse_df[
        rmse_df["threshold"] == threshold
    ]

    movement_vs_conventional = rows[
        (
            rows["model_a"] ==
            "poly_conventional"
        )
        &
        (
            rows["model_b"] ==
            "poly_movement"
        )
    ].iloc[0]

    nonlinear_vs_linear = rows[
        (
            rows["model_a"] ==
            "linear_movement"
        )
        &
        (
            rows["model_b"] ==
            "poly_movement"
        )
    ].iloc[0]

    print(
        f"\nThreshold >= {threshold}"
    )

    print(
        "Polynomial movement vs "
        "polynomial conventional:"
    )

    print(
        f"  RMSE diff = "
        f"{movement_vs_conventional['observed_difference']:+.5f}"
    )

    print(
        f"  95% CI = "
        f"[{movement_vs_conventional['ci_low']:+.5f}, "
        f"{movement_vs_conventional['ci_high']:+.5f}]"
    )

    print(
        f"  P(improves) = "
        f"{movement_vs_conventional['p_improves']:.4f}"
    )

    print(
        "\nPolynomial movement vs "
        "linear movement:"
    )

    print(
        f"  RMSE diff = "
        f"{nonlinear_vs_linear['observed_difference']:+.5f}"
    )

    print(
        f"  95% CI = "
        f"[{nonlinear_vs_linear['ci_low']:+.5f}, "
        f"{nonlinear_vs_linear['ci_high']:+.5f}]"
    )

    print(
        f"  P(improves) = "
        f"{nonlinear_vs_linear['p_improves']:.4f}"
    )


print("\nSaved:")
print(results_path)
print(rmse_path)

print("\nDone.")
