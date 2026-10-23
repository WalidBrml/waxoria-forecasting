"""Compare all models on ONE store-product series.

Usage:
    python -m src.forecasting.compare_models
    python -m src.forecasting.compare_models --store 3 --item 17

Uses the same shared features and rolling-origin evaluation as the full
500-series benchmark (no duplicated feature code). Writes
reports/model_comparison.csv with columns
model, MAE, RMSE, WAPE, MASE, Bias, n_obs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.forecasting.metrics import per_series_metrics
from src.forecasting.rolling_origin import (
    EvaluationConfig,
    load_sales_matrix,
    run_evaluation,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "sales_clean.csv"
REPORTS = ROOT / "reports"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=int, default=1)
    ap.add_argument("--item", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--n-windows", type=int, default=12)
    ap.add_argument("--refit-every", type=int, default=4)
    args = ap.parse_args(argv)

    matrix = load_sales_matrix(DATA_PATH).subset([args.store], [args.item])
    if matrix.n_series != 1:
        raise ValueError("Select exactly one store-item series.")

    config = EvaluationConfig(
        horizon=args.horizon, n_windows=args.n_windows, refit_every=args.refit_every
    )
    result = run_evaluation(matrix, config, progress_every=0)
    table = per_series_metrics(result.predictions, result.scales)
    table = (
        table[["model", "MAE", "RMSE", "WAPE", "MASE", "Bias", "n_obs"]]
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / "model_comparison.csv", index=False)

    last = result.predictions["date"].max().date()
    first = result.predictions["date"].min().date()
    print("\n=== WAXORIA MODEL COMPARISON ===")
    print(f"Store {args.store} | Item {args.item} | {first} to {last} | "
          f"{config.n_windows} windows x {config.horizon}-day horizon")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved results to: {REPORTS / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
