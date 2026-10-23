"""Evaluate all store-product series with chronological rolling-origin windows.

Usage (from the project root):

    python -m src.forecasting.evaluate_multiple_series
    python -m src.forecasting.evaluate_multiple_series --n-jobs 4

Models compared on identical series, dates and horizons:
    Previous-day baseline, 7-day moving average, Seasonal naive, Random Forest.

Outputs (in reports/ unless a store/item filter is used):
    multi_series_metrics.csv     per series x model metrics (dashboard input)
    model_summary.csv            pooled metrics per model
    predictions_all_models.csv   every prediction, long format
    evaluation_report.json       coverage, configuration, checks, runtime
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.forecasting.metrics import per_series_metrics, pooled_metrics
from src.forecasting.rolling_origin import (
    EvaluationConfig,
    EvaluationResult,
    MODEL_NAMES,
    load_sales_matrix,
    run_evaluation,
    validate_predictions,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "sales_clean.csv"
REPORTS_DIR = ROOT / "reports"


def build_report(result: EvaluationResult, matrix_total_series: int) -> dict:
    p = result.predictions
    cfg = result.config
    series = p[["store", "item"]].drop_duplicates()
    per_model = p.groupby("model", observed=True).size()
    return {
        "coverage": {
            "stores": int(series["store"].nunique()),
            "products": int(series["item"].nunique()),
            "series": int(len(series)),
            "series_in_dataset": int(matrix_total_series),
            "full_coverage": bool(len(series) == matrix_total_series),
            "forecast_windows": int(len(result.windows)),
            "observations_per_model": int(per_model.iloc[0]),
            "total_prediction_rows": int(len(p)),
            "test_start": str(p["date"].min().date()),
            "test_end": str(p["date"].max().date()),
        },
        "forecast_type": (
            f"Direct multi-horizon: every {cfg.horizon} days a forecast is "
            f"issued for the next {cfg.horizon} days (h=1..{cfg.horizon}). "
            "Models are retrained every "
            f"{cfg.refit_every} windows ({cfg.refit_every * cfg.horizon} days) "
            "on an expanding history."
        ),
        "config": {
            "horizon": cfg.horizon,
            "n_windows": cfg.n_windows,
            "refit_every_windows": cfg.refit_every,
            "random_forest": {
                "n_estimators": cfg.n_estimators,
                "max_depth": cfg.max_depth,
                "min_samples_leaf": cfg.min_samples_leaf,
                "random_state": cfg.random_state,
            },
        },
        "models": MODEL_NAMES,
        "checks_passed": [
            "chronological split assertions (training labels <= cutoff < test targets)",
            "features unchanged when all data after each origin is replaced (no leakage)",
            "every model scored on identical (store, item, date) rows",
            "no NaN or negative predictions",
        ],
        "runtime_seconds": round(result.seconds, 1),
    }


def write_outputs(result: EvaluationResult, out_dir: Path, total_series: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = validate_predictions(result.predictions)

    per_series = per_series_metrics(result.predictions, result.scales)
    summary = pooled_metrics(per_series)

    per_series.to_csv(out_dir / "multi_series_metrics.csv", index=False)
    summary.to_csv(out_dir / "model_summary.csv", index=False)
    result.predictions.to_csv(
        out_dir / "predictions_all_models.csv", index=False, float_format="%.4f"
    )
    report = build_report(result, total_series)
    report["validation"] = checks
    (out_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    return {"per_series": per_series, "summary": summary, "report": report}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horizon", type=int, default=7, help="forecast horizon in days")
    ap.add_argument("--n-windows", type=int, default=12, help="number of consecutive forecast windows")
    ap.add_argument("--refit-every", type=int, default=4, help="refit the models every N windows")
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=1, help="parallel worker processes (series are independent)")
    ap.add_argument("--stores", type=int, nargs="*", help="optional subset; output goes to reports/partial")
    ap.add_argument("--items", type=int, nargs="*", help="optional subset; output goes to reports/partial")
    ap.add_argument("--output-dir", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    matrix = load_sales_matrix(DATA_PATH)
    total_series = matrix.n_series

    partial = bool(args.stores or args.items)
    if partial:
        matrix = matrix.subset(args.stores, args.items)
    out_dir = args.output_dir or (REPORTS_DIR / "partial" if partial else REPORTS_DIR)

    config = EvaluationConfig(
        horizon=args.horizon,
        n_windows=args.n_windows,
        refit_every=args.refit_every,
        n_estimators=args.n_estimators,
    )

    print("\n=== WAXORIA ROLLING-ORIGIN EVALUATION ===")
    print(f"Series: {matrix.n_series} of {total_series}"
          f"{'  (PARTIAL: filtered by --stores/--items)' if partial else ''}")
    print(f"Horizon: {config.horizon} days | windows: {config.n_windows} | "
          f"refit every {config.refit_every} windows | jobs: {args.n_jobs}")

    result = run_evaluation(matrix, config, n_jobs=args.n_jobs)
    out = write_outputs(result, out_dir, total_series)
    rep = out["report"]["coverage"]

    pd.set_option("display.width", 140)
    print("\n--- Coverage ---")
    for k, v in rep.items():
        print(f"{k:>26}: {v}")
    print("\n--- Pooled metrics (lower is better; WAPE/Bias are ratios of sums) ---")
    show = out["summary"][["model", "MAE", "RMSE", "WAPE", "Bias", "MASE_mean", "MASE_median", "series_best"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nFiles written to: {out_dir}")
    print(f"Runtime: {result.seconds:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
