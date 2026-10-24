from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA = PROJECT_ROOT / "data" / "combine_tracking.csv"
OUT = (
    PROJECT_ROOT
    / "research_outputs"
    / "model_comparison"
    / "speed_cv_signal"
    / "outlier_diagnostic"
)

OUT.mkdir(parents=True, exist_ok=True)

NFL_ID = 56248

print("=" * 80)
print("RAW OUTLIER DATA INSPECTION")
print("=" * 80)

cols = [
    "draft_year",
    "event_id",
    "nfl_id",
    "entity_type",
    "time",
    "drill_type",
    "drill_name",
    "attempt",
    "x",
    "y",
    "s",
    "a",
    "dis",
    "dir",
]

print("\nLoading combine tracking...")

df = pd.read_csv(
    DATA,
    usecols=cols,
    low_memory=False
)

player = df[
    (df["nfl_id"] == NFL_ID)
    &
    (
        df["drill_name"]
        .astype(str)
        .str.upper()
        .str.contains(
            "DAGGER_ROUTE_LEFT",
            na=False
        )
    )
].copy()

print(f"\nRows found: {len(player)}")

print("\n--- RAW DATA TYPES ---")
print(player.dtypes.to_string())

print("\n--- MISSING VALUES ---")

for col in [
    "time",
    "x",
    "y",
    "s",
    "a",
    "dis",
    "dir",
]:
    print(
        f"{col:10s}: "
        f"{player[col].isna().sum()} missing / "
        f"{len(player)} total"
    )

print("\n--- UNIQUE VALUES ---")

for col in [
    "event_id",
    "attempt",
    "drill_type",
    "drill_name",
]:
    print(f"\n{col}:")
    print(
        player[col]
        .value_counts(dropna=False)
        .to_string()
    )

print("\n--- FIRST 20 RAW ROWS ---")

print(
    player[
        [
            "time",
            "x",
            "y",
            "s",
            "a",
            "dis",
            "dir",
            "attempt",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\n--- LAST 20 RAW ROWS ---")

print(
    player[
        [
            "time",
            "x",
            "y",
            "s",
            "a",
            "dis",
            "dir",
            "attempt",
        ]
    ]
    .tail(20)
    .to_string(index=False)
)

# ------------------------------------------------------------------
# Try explicit numeric conversion and report exactly what happens.
# ------------------------------------------------------------------

print("\n--- NUMERIC CONVERSION TEST ---")

test = player.copy()

for col in [
    "time",
    "x",
    "y",
    "s",
    "a",
    "dis",
    "dir",
]:
    test[col + "_num"] = pd.to_numeric(
        test[col],
        errors="coerce"
    )

for col in [
    "time",
    "x",
    "y",
    "s",
    "a",
    "dis",
    "dir",
]:
    numeric_col = col + "_num"

    print(
        f"{col:10s}: "
        f"valid numeric = "
        f"{test[numeric_col].notna().sum():3d} / "
        f"{len(test)}"
    )

# ------------------------------------------------------------------
# Test the exact filtering used by the previous script.
# ------------------------------------------------------------------

numeric_test = test.dropna(
    subset=[
        "time_num",
        "x_num",
        "y_num",
        "s_num",
    ]
).copy()

print("\n--- EXACT FEATURE CALCULATION CHECK ---")

print(
    f"Rows after requiring time/x/y/s: "
    f"{len(numeric_test)}"
)

if len(numeric_test) > 0:

    numeric_test = numeric_test.sort_values(
        "time_num"
    )

    speed = numeric_test["s_num"]

    mean_speed = speed.mean()
    std_speed = speed.std(ddof=0)

    print(
        f"Time min:       {numeric_test['time_num'].min()}"
    )

    print(
        f"Time max:       {numeric_test['time_num'].max()}"
    )

    print(
        f"Mean speed:     {mean_speed}"
    )

    print(
        f"Std speed:      {std_speed}"
    )

    print(
        f"Calculated CV:  "
        f"{std_speed / mean_speed if mean_speed != 0 else np.nan}"
    )

    dx = numeric_test["x_num"].diff()
    dy = numeric_test["y_num"].diff()

    distance = np.sqrt(
        dx ** 2 + dy ** 2
    )

    print(
        f"Path length:    {distance.sum()}"
    )

    direct = np.sqrt(
        (
            numeric_test["x_num"].iloc[-1]
            -
            numeric_test["x_num"].iloc[0]
        ) ** 2
        +
        (
            numeric_test["y_num"].iloc[-1]
            -
            numeric_test["y_num"].iloc[0]
        ) ** 2
    )

    print(
        f"Direct distance: {direct}"
    )

else:
    print(
        "\nNO VALID ROWS AFTER NUMERIC CONVERSION."
    )

# ------------------------------------------------------------------
# Save exact raw data for inspection.
# ------------------------------------------------------------------

raw_path = OUT / "outlier_56248_raw_rows.csv"

player.to_csv(
    raw_path,
    index=False
)

print("\nRaw rows saved to:")
print(raw_path)

print("\n" + "=" * 80)
print("RAW INSPECTION COMPLETE")
print("=" * 80)
