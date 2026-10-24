
from pathlib import Path
import pandas as pd

DATA_DIR = Path("nfl-big-data-bowl-2027")

if not DATA_DIR.exists():
    DATA_DIR = Path(".")

print("=" * 80)
print("NFL BIG DATA BOWL 2027 — DATA AUDIT")
print("=" * 80)

for path in sorted(DATA_DIR.glob("*.csv")):
    print(f"\n{'=' * 80}")
    print(f"FILE: {path.name}")
    print(f"SIZE: {path.stat().st_size / (1024**2):.2f} MB")

    try:
        # Read only a few rows; don't load the entire dataset.
        df = pd.read_csv(path, nrows=5, low_memory=False)

        print(f"\nCOLUMNS ({len(df.columns)}):")
        print(df.columns.tolist())

        print("\nSAMPLE DATA:")
        print(df.head(3).to_string(index=False))

        print("\nSAMPLE DATA TYPES:")
        print(df.dtypes.to_string())

        print("\nMISSING VALUES IN SAMPLE:")
        print(df.isna().sum().to_string())

    except Exception as exc:
        print(f"ERROR READING FILE: {exc}")

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
