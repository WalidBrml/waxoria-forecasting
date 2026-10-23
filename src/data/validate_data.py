
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "train.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_sales_data(path: Path = RAW_DATA) -> pd.DataFrame:
    """Load and validate the raw retail sales dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place train.csv inside data/raw/."
        )

    df = pd.read_csv(path)

    required_columns = {"date", "store", "item", "sales"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError("Invalid or missing dates were found.")

    if df[["store", "item", "sales"]].isna().any().any():
        raise ValueError("Missing values were found in required columns.")

    if (df["sales"] < 0).any():
        raise ValueError("Negative sales were found; investigate the records.")

    if df.duplicated(["date", "store", "item"]).any():
        raise ValueError(
            "Duplicate date/store/item records were found."
        )

    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_sales_data()

    output_path = PROCESSED_DIR / "sales_clean.csv"
    df.to_csv(output_path, index=False)

    print("\n=== WAXORIA DATA VALIDATION ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min().date()} to "
          f"{df['date'].max().date()}")
    print(f"Stores: {df['store'].nunique()}")
    print(f"Products: {df['item'].nunique()}")
    print(f"Total units sold: {df['sales'].sum():,}")
    print(f"Missing values:\n{df.isna().sum()}")
    print(f"\nValidated dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
