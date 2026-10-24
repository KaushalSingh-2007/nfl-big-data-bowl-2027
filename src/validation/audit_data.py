from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

print("=" * 72)
print("NFL BIG DATA BOWL 2027 - DATA AUDIT")
print("Project:", PROJECT_ROOT)
print("=" * 72)

for path in sorted(DATA_DIR.glob("*.csv")):
    print(f"\nFILE: {path.name}")
    print(f"SIZE: {path.stat().st_size / (1024**2):.2f} MB")
    try:
        sample = pd.read_csv(path, nrows=5, low_memory=False)
        print(f"Sample columns: {len(sample.columns)}")
        print("Columns:", ", ".join(map(str, sample.columns[:15])))
    except Exception as exc:
        print("Could not read sample:", exc)

print("\nAudit finished. Only the first five rows of each CSV were read.")
