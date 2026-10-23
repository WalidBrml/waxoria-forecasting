
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sales_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def load_sales() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Clean data not found. Run the validation pipeline first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date")

    return df


def daily_series(df: pd.DataFrame, store: int, item: int) -> pd.Series:
    """Daily sales of one series; raises if any day is missing (no zero-filling)."""
    series = df.loc[(df["store"] == store) & (df["item"] == item)].set_index("date")["sales"]
    series = series.sort_index().asfreq("D")
    if series.isna().any():
        raise ValueError(
            f"Store {store}, item {item} has {int(series.isna().sum())} missing days; "
            "fix the data instead of filling them with 0."
        )
    return series.astype(float)


def evaluate_forecasts(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def main() -> None:
    df = load_sales()

    # First experiment: store 1, item 1.
    series = daily_series(df, store=1, item=1)

    # Reserve the last 90 days for evaluation.
    test_days = 90
    train = series.iloc[:-test_days]
    test = series.iloc[-test_days:]

    # Baseline: use the previous day's observed sales.
    baseline_predictions = series.shift(1).loc[test.index]

    # Model: predict using the average of the previous 7 days.
    rolling_predictions = series.shift(1).rolling(7).mean().loc[test.index]

    baseline_predictions = baseline_predictions.fillna(train.iloc[-1])
    rolling_predictions = rolling_predictions.fillna(train.mean())

    baseline_metrics = evaluate_forecasts(test, baseline_predictions)
    rolling_metrics = evaluate_forecasts(test, rolling_predictions)

    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(
        [
            {"model": "Previous-day baseline", **baseline_metrics},
            {"model": "Seven-day moving average", **rolling_metrics},
        ]
    )

    results.to_csv(REPORTS_DIR / "forecast_metrics.csv", index=False)

    predictions = pd.DataFrame(
        {
            "date": test.index,
            "actual_sales": test.values,
            "baseline_prediction": baseline_predictions.values,
            "moving_average_prediction": rolling_predictions.values,
        }
    )
    predictions.to_csv(REPORTS_DIR / "forecast_predictions.csv", index=False)

    print("\n=== WAXORIA FORECAST EVALUATION ===")
    print("Store: 1 | Item: 1")
    print(f"Training observations: {len(train)}")
    print(f"Test observations: {len(test)}")
    print("\nMetrics (lower is better):")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nMetrics saved to: {REPORTS_DIR / 'forecast_metrics.csv'}")
    print(f"Predictions saved to: {REPORTS_DIR / 'forecast_predictions.csv'}")


if __name__ == "__main__":
    main()
