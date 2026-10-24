import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\singh\Downloads\nfl-big-data-bowl-2027\nfl-big-data-bowl-2027")
DATA = ROOT / "data"
SRC = ROOT / "src"

print("=" * 80)
print("DIAGNOSTIC: WHY DOES DAGGER HAVE 9917 ROWS INSTEAD OF 7551?")
print("=" * 80)

tracking = pd.read_csv(DATA / "combine_tracking.csv", low_memory=False)

dagger = tracking[
    tracking["drill_name"].astype(str).str.upper().eq("DAGGER_ROUTE_LEFT")
].copy()

print(f"\nAll Dagger rows: {len(dagger):,}")

print("\nEntity types:")
print(dagger["entity_type"].value_counts(dropna=False).to_string())

print("\nDrill types:")
print(dagger["drill_type"].value_counts(dropna=False).to_string())

print("\nAttempts:")
print(dagger["attempt"].value_counts(dropna=False).sort_index().to_string())

print("\nDraft years:")
print(dagger["draft_year"].value_counts(dropna=False).sort_index().to_string())

print("\nRows by entity_type + drill_type:")
print(
    dagger.groupby(
        ["entity_type", "drill_type"],
        dropna=False
    ).size().sort_values(ascending=False).to_string()
)

print("\nRows by entity_type + draft_year:")
print(
    dagger.groupby(
        ["entity_type", "draft_year"],
        dropna=False
    ).size().to_string()
)

print("\n" + "=" * 80)
print("ORIGINAL SCRIPT REFERENCES")
print("=" * 80)

script = SRC / "features" / "build_whole_route_features.py"

if script.exists():
    text = script.read_text(encoding="utf-8")

    lines = text.splitlines()

    for i, line in enumerate(lines, start=1):
        upper = line.upper()
        if (
            "ENTITY_TYPE" in upper
            or "DRILL_NAME" in upper
            or "DAGGER_ROUTE_LEFT" in upper
            or "COMEBACK_ROUTE_RIGHT" in upper
            or "KEYS" in upper
        ):
            start = max(1, i - 3)
            end = min(len(lines), i + 3)

            print(f"\n--- lines {start}-{end} ---")
            for j in range(start, end + 1):
                print(f"{j:4}: {lines[j-1]}")
else:
    print(f"Could not find: {script}")

print("\n" + "=" * 80)
print("DONE — NOTHING WAS MODIFIED")
print("=" * 80)
