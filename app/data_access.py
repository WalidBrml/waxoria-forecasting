"""Data access for the Waxoria dashboard (no Streamlit imports).

Everything the dashboard shows comes from the saved project files:

    data/processed/sales_clean.csv        history
    reports/predictions_all_models.csv    backtest predictions, every model
    reports/multi_series_metrics.csv      per series x model metrics
    reports/model_summary.csv             pooled metrics per model
    reports/evaluation_report.json        coverage, configuration, checks

Each loader validates its schema and reports a ``FileStatus`` instead of
raising, so a missing or malformed file degrades one page, not the app.
Filtered metrics are recomputed from the predictions with the same
functions used by the evaluation pipeline (``src.forecasting.metrics``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.forecasting.metrics import mase_scale, per_series_metrics, pooled_metrics

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "sales": ROOT / "data" / "processed" / "sales_clean.csv",
    "predictions": ROOT / "reports" / "predictions_all_models.csv",
    "per_series": ROOT / "reports" / "multi_series_metrics.csv",
    "summary": ROOT / "reports" / "model_summary.csv",
    "report": ROOT / "reports" / "evaluation_report.json",
}
LABELS = {
    "sales": "Clean sales history",
    "predictions": "Predictions (all models)",
    "per_series": "Per-series metrics",
    "summary": "Model summary",
    "report": "Evaluation report",
}
HOW_TO_CREATE = {
    "sales": "python -m src.data.validate_data",
    "predictions": "python -m src.forecasting.evaluate_multiple_series --n-jobs 4",
    "per_series": "python -m src.forecasting.evaluate_multiple_series --n-jobs 4",
    "summary": "python -m src.forecasting.evaluate_multiple_series --n-jobs 4",
    "report": "python -m src.forecasting.evaluate_multiple_series --n-jobs 4",
}

REQUIRED = {
    "sales": {"date", "store", "item", "sales"},
    "predictions": {
        "store", "item", "forecast_window", "forecast_origin", "horizon",
        "date", "actual_sales", "model", "predicted_sales",
    },
    "per_series": {"store", "item", "model", "MAE", "RMSE", "WAPE", "MASE", "Bias", "n_obs"},
    "summary": {"model", "n_series", "n_obs", "MAE", "RMSE", "WAPE", "Bias", "MASE_mean"},
}

MODEL_ORDER = [
    "Random Forest",
    "Seasonal naive",
    "7-day moving average",
    "Previous-day baseline",
]
RF_MODEL = "Random Forest"


# ------------------------------------------------------------- status/types

@dataclass
class FileStatus:
    key: str
    label: str
    path: Path
    exists: bool = False
    ok: bool = False
    message: str = ""
    rows: int | None = None
    size_bytes: int | None = None
    columns: list[str] = field(default_factory=list)

    @property
    def state(self) -> str:
        return "Ready" if self.ok else ("Invalid" if self.exists else "Missing")


@dataclass
class AppData:
    sales: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    per_series: pd.DataFrame | None = None
    summary: pd.DataFrame | None = None
    report: dict | None = None
    scales: pd.DataFrame | None = None
    status: dict[str, FileStatus] = field(default_factory=dict)

    @property
    def has_sales(self) -> bool:
        return self.sales is not None and not self.sales.empty

    @property
    def has_predictions(self) -> bool:
        return self.predictions is not None and not self.predictions.empty

    def models(self) -> list[str]:
        if not self.has_predictions:
            return []
        present = set(self.predictions["model"].astype(str).unique())
        return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


@dataclass(frozen=True)
class Filters:
    start: pd.Timestamp
    end: pd.Timestamp
    stores: tuple[int, ...]


def file_signature(paths: dict[str, Path] = PATHS) -> tuple:
    """Changes whenever any input file changes (used as a cache key)."""
    sig = []
    for key, p in paths.items():
        try:
            st = Path(p).stat()
            sig.append((key, st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((key, None, None))
    return tuple(sig)


# ----------------------------------------------------------------- loaders

def _read_sales(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["date", "store", "item", "sales"],
        dtype={"store": "int16", "item": "int16", "sales": "int32"},
        parse_dates=["date"],
    )
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)


def _read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"store": "int16", "item": "int16", "forecast_window": "int16",
               "horizon": "int8", "model": "category"},
        parse_dates=["date", "forecast_origin"],
    )
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_report(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "coverage" not in data:
        raise ValueError("JSON has no 'coverage' section")
    return data


READERS: dict[str, Callable[[Path], object]] = {
    "sales": _read_sales,
    "predictions": _read_predictions,
    "per_series": _read_csv,
    "summary": _read_csv,
    "report": _read_report,
}


def load_one(key: str, path: Path):
    """Return (object_or_None, FileStatus). Never raises for file problems."""
    status = FileStatus(key=key, label=LABELS[key], path=Path(path))
    path = Path(path)
    if not path.exists():
        status.message = f"File not found. Create it with: {HOW_TO_CREATE[key]}"
        return None, status
    status.exists = True
    status.size_bytes = path.stat().st_size
    try:
        obj = READERS[key](path)
    except Exception as exc:  # malformed file: report, do not crash
        status.message = f"Could not read file: {exc}"
        return None, status

    if isinstance(obj, pd.DataFrame):
        status.columns = list(obj.columns)
        status.rows = len(obj)
        missing = REQUIRED.get(key, set()) - set(obj.columns)
        if missing:
            status.message = f"Missing columns: {sorted(missing)}"
            return None, status
        if obj.empty:
            status.message = "File contains no rows."
            return None, status
    status.ok = True
    status.message = "OK"
    return obj, status


def compute_mase_scales(sales: pd.DataFrame, first_origin: pd.Timestamp) -> pd.DataFrame:
    """Per-series MASE scale from history up to the first forecast origin.

    Identical to what the evaluation pipeline used (in-sample lag-7 MAE).
    """
    hist = sales[sales["date"] <= first_origin]
    rows = []
    for (store, item), g in hist.groupby(["store", "item"], sort=True):
        rows.append(
            {"store": int(store), "item": int(item),
             "mase_scale": mase_scale(g.sort_values("date")["sales"].to_numpy())}
        )
    return pd.DataFrame(rows, columns=["store", "item", "mase_scale"])


def load_all(paths: dict[str, Path] = PATHS) -> AppData:
    data = AppData()
    for key in ("sales", "predictions", "per_series", "summary", "report"):
        obj, status = load_one(key, paths[key])
        setattr(data, key, obj)
        data.status[key] = status
    if data.has_sales and data.has_predictions:
        first_origin = data.predictions["forecast_origin"].min()
        data.scales = compute_mase_scales(data.sales, first_origin)
    return data


# ----------------------------------------------------------------- filters

def default_filters(data: AppData) -> Filters:
    if not data.has_sales:
        today = pd.Timestamp.today().normalize()
        return Filters(today, today, ())
    s = data.sales
    return Filters(s["date"].min(), s["date"].max(), tuple(sorted(int(x) for x in s["store"].unique())))


def filter_sales(sales: pd.DataFrame, f: Filters, items: list[int] | None = None) -> pd.DataFrame:
    mask = (sales["date"] >= f.start) & (sales["date"] <= f.end) & sales["store"].isin(f.stores)
    if items:
        mask &= sales["item"].isin(items)
    return sales[mask]


def filter_predictions(
    pred: pd.DataFrame,
    f: Filters | None = None,
    *,
    models: list[str] | None = None,
    store: int | None = None,
    item: int | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    mask = np.ones(len(pred), dtype=bool)
    if f is not None:
        mask &= ((pred["date"] >= f.start) & (pred["date"] <= f.end)
                 & pred["store"].isin(f.stores)).to_numpy()
    if models is not None:
        mask &= pred["model"].isin(models).to_numpy()
    if store is not None:
        mask &= (pred["store"] == store).to_numpy()
    if item is not None:
        mask &= (pred["item"] == item).to_numpy()
    if horizons is not None:
        mask &= pred["horizon"].isin(horizons).to_numpy()
    return pred[mask]


# ----------------------------------------------------------------- metrics

def metric_tables(pred: pd.DataFrame, scales: pd.DataFrame | None):
    """(per_series, pooled) recomputed from ``pred`` rows; empty if no rows."""
    if pred is None or pred.empty:
        return pd.DataFrame(), pd.DataFrame()
    if scales is None:
        scales = pred[["store", "item"]].drop_duplicates().assign(mase_scale=np.nan)
    per_series = per_series_metrics(pred, scales)
    return per_series, pooled_metrics(per_series)


def order_models(df: pd.DataFrame, col: str = "model") -> pd.DataFrame:
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return df.sort_values(col, key=lambda s: s.map(lambda m: rank.get(m, 99))).reset_index(drop=True)


def metrics_by_horizon(pred: pd.DataFrame) -> pd.DataFrame:
    """Pooled MAE / WAPE / Bias per (model, horizon)."""
    if pred.empty:
        return pd.DataFrame()
    err = pred["predicted_sales"] - pred["actual_sales"]
    work = pred.assign(abs_err=err.abs(), signed=err)
    g = work.groupby(["model", "horizon"], observed=True).agg(
        n_obs=("abs_err", "size"),
        MAE=("abs_err", "mean"),
        sum_abs=("abs_err", "sum"),
        sum_signed=("signed", "sum"),
        sum_actual=("actual_sales", "sum"),
    ).reset_index()
    nz = g["sum_actual"] > 0
    g["WAPE"] = (g["sum_abs"] / g["sum_actual"]).where(nz)
    g["Bias"] = (g["sum_signed"] / g["sum_actual"]).where(nz)
    return g[["model", "horizon", "n_obs", "MAE", "WAPE", "Bias"]]


def pooled_by_store(per_series: pd.DataFrame) -> pd.DataFrame:
    """Pooled WAPE per (store, model): ratio of sums, never a mean of ratios."""
    if per_series.empty:
        return pd.DataFrame()
    g = per_series.groupby(["store", "model"], observed=True)[
        ["sum_abs_err", "sum_actual"]
    ].sum().reset_index()
    g["WAPE"] = (g["sum_abs_err"] / g["sum_actual"]).where(g["sum_actual"] > 0)
    return g


def window_errors(pred: pd.DataFrame) -> pd.DataFrame:
    """MAE per forecast window and model for the rows in ``pred``."""
    if pred.empty:
        return pd.DataFrame()
    err = (pred["predicted_sales"] - pred["actual_sales"]).abs()
    out = (
        pred.assign(abs_err=err)
        .groupby(["forecast_window", "forecast_origin", "model"], observed=True)["abs_err"]
        .mean()
        .unstack("model")
        .reset_index()
    )
    out["forecast_origin"] = out["forecast_origin"].dt.date
    return out


def origin_weekdays(pred: pd.DataFrame) -> list[str]:
    """Distinct weekday names of the forecast origins (e.g. ['Sunday'])."""
    if pred is None or pred.empty:
        return []
    return sorted(pred["forecast_origin"].dt.day_name().unique())


def horizon_labels(pred: pd.DataFrame) -> dict[int, str]:
    """Axis label per horizon; adds the target weekday when it is constant.

    If every forecast origin falls on the same weekday, horizon and target
    weekday are the same thing, and the label says so ("1 · Mon").
    """
    labels: dict[int, str] = {}
    if pred is None or pred.empty:
        return labels
    for h, g in pred.groupby("horizon"):
        days = g["date"].dt.day_name().str[:3]
        top = days.value_counts(normalize=True)
        labels[int(h)] = f"{int(h)} · {top.index[0]}" if top.iloc[0] >= 0.99 else str(int(h))
    return labels


# ---------------------------------------------------------------- coverage

def coverage_from_predictions(pred: pd.DataFrame) -> dict:
    """Coverage as actually present in ``pred`` (same keys as the JSON report)."""
    if pred is None or pred.empty:
        return {"stores": 0, "products": 0, "series": 0, "forecast_windows": 0,
                "observations_per_model": 0, "total_prediction_rows": 0,
                "test_start": None, "test_end": None}
    series = pred[["store", "item"]].drop_duplicates()
    per_model = pred.groupby("model", observed=True).size()
    return {
        "stores": int(series["store"].nunique()),
        "products": int(series["item"].nunique()),
        "series": int(len(series)),
        "forecast_windows": int(pred["forecast_window"].nunique()),
        "observations_per_model": int(per_model.min()),
        "total_prediction_rows": int(len(pred)),
        "test_start": str(pred["date"].min().date()),
        "test_end": str(pred["date"].max().date()),
    }


def report_mismatches(report: dict | None, actual: dict) -> list[str]:
    """Differences between evaluation_report.json and the predictions file."""
    if not report:
        return []
    rep = report.get("coverage", {})
    out = []
    for key in ("stores", "products", "series", "forecast_windows",
                "observations_per_model", "total_prediction_rows",
                "test_start", "test_end"):
        if key in rep and rep[key] != actual.get(key):
            out.append(f"{key}: report says {rep[key]}, predictions file has {actual.get(key)}")
    return out


# ------------------------------------------------------------- sales stats

def yoy_growth(sales_f: pd.DataFrame, end: pd.Timestamp) -> float | None:
    """Units in the 365 days to ``end`` vs the 365 days before; None if unsupported."""
    if sales_f.empty:
        return None
    cur = sales_f[(sales_f["date"] > end - pd.Timedelta(days=365)) & (sales_f["date"] <= end)]
    prev = sales_f[(sales_f["date"] > end - pd.Timedelta(days=730))
                   & (sales_f["date"] <= end - pd.Timedelta(days=365))]
    if prev.empty or cur["date"].nunique() < 365 or prev["date"].nunique() < 365:
        return None
    p = float(prev["sales"].sum())
    return None if p == 0 else float(cur["sales"].sum()) / p - 1.0


def mae_reduction(model_mae: float, baseline_mae: float) -> float | None:
    """Share of the baseline's MAE removed by the model (positive = model is better).

    Shown on the Overview as "MAE reduction", not "MAE vs baseline", so that a
    positive number cannot be read as higher error.
    """
    if baseline_mae is None or not np.isfinite(baseline_mae) or baseline_mae <= 0:
        return None
    return float(1.0 - model_mae / baseline_mae)


def zero_demand_summary(sales: pd.DataFrame) -> tuple[int, int]:
    """(number of zero-sales days, number of series with no sales at all)."""
    zero_days = int((sales["sales"] == 0).sum())
    totals = sales.groupby(["store", "item"])["sales"].sum()
    return zero_days, int((totals == 0).sum())


def zero_demand_caption(sales: pd.DataFrame) -> str:
    """Caption that stays true whatever the data contains."""
    zero_days, all_zero = zero_demand_summary(sales)
    tail = ("No series is entirely zero." if all_zero == 0
            else f"{all_zero} series have no sales at all.")
    return f"Zero-sales days: {zero_days:,}. {tail}"


def series_history(sales: pd.DataFrame, store: int, item: int) -> pd.Series:
    """Daily sales of one series as a date-indexed Series (daily frequency)."""
    g = sales[(sales["store"] == store) & (sales["item"] == item)]
    s = g.set_index("date")["sales"].sort_index().astype(float)
    return s.asfreq("D") if len(s) else s


def rf_forecast_error_std(per_series: pd.DataFrame | None, store: int, item: int) -> float | None:
    """Random Forest RMSE for the series from the saved evaluation, if present."""
    if per_series is None or per_series.empty:
        return None
    r = per_series[(per_series["store"] == store) & (per_series["item"] == item)
                   & (per_series["model"].astype(str) == RF_MODEL)]
    return None if r.empty else float(r["RMSE"].iloc[0])
