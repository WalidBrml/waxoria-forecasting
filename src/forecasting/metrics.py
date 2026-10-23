"""Forecast accuracy metrics for Waxoria.

Definitions (a = actual, p = prediction, e = p - a)
---------------------------------------------------
MAE   mean(|e|)
RMSE  sqrt(mean(e^2))
WAPE  sum(|e|) / sum(a)                      (weighted absolute % error)
Bias  sum(e) / sum(a)                        (positive = over-forecasting)
MASE  MAE / scale, scale = mean(|y_t - y_{t-7}|) on the training history
      (mean absolute error of an in-sample seasonal-naive forecast).

Pooled vs per-series
--------------------
WAPE and Bias are ratios. Pooled WAPE is ``sum(|e|) / sum(a)`` over ALL
observations, never the mean of per-series ratios (that would let a tiny
series count as much as a huge one). MASE is already scale-free per series,
so the mean/median across series is a valid summary.

Edge cases
----------
* sum(a) == 0 (a zero-demand evaluation window): WAPE and Bias are
  undefined and returned as NaN, never inf and never 0.
* MASE scale == 0 (history is perfectly weekly-periodic, e.g. all zeros) or
  history shorter than one season: MASE is NaN.
* NaN values are excluded from summaries and counted explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASON = 7


def mae(actual, predicted) -> float:
    a, p = _pair(actual, predicted)
    return float(np.mean(np.abs(p - a)))


def rmse(actual, predicted) -> float:
    a, p = _pair(actual, predicted)
    return float(np.sqrt(np.mean((p - a) ** 2)))


def wape(actual, predicted) -> float:
    a, p = _pair(actual, predicted)
    denom = float(np.sum(a))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(p - a)) / denom)


def bias(actual, predicted) -> float:
    a, p = _pair(actual, predicted)
    denom = float(np.sum(a))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(p - a) / denom)


def mase_scale(history, season: int = SEASON) -> float:
    """In-sample seasonal-naive MAE; NaN if history is too short."""
    h = np.asarray(history, dtype=float)
    if len(h) <= season:
        return float("nan")
    return float(np.mean(np.abs(h[season:] - h[:-season])))


def mase(actual, predicted, scale: float) -> float:
    if not np.isfinite(scale) or scale == 0.0:
        return float("nan")
    return mae(actual, predicted) / scale


def _pair(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError("actual and predicted must have the same shape")
    if a.size == 0:
        raise ValueError("cannot score an empty forecast")
    return a, p


def per_series_metrics(
    predictions: pd.DataFrame, scales: pd.DataFrame
) -> pd.DataFrame:
    """One row per (store, item, model).

    ``predictions`` is the long prediction table; ``scales`` has columns
    ``store, item, mase_scale``.
    """
    df = predictions[["store", "item", "model", "actual_sales", "predicted_sales"]]
    err = df["predicted_sales"] - df["actual_sales"]
    work = df.assign(
        abs_err=err.abs(), sq_err=err**2, signed_err=err
    )
    grouped = work.groupby(["store", "item", "model"], observed=True)
    out = grouped.agg(
        n_obs=("abs_err", "size"),
        sum_abs_err=("abs_err", "sum"),
        sum_sq_err=("sq_err", "sum"),
        sum_signed_err=("signed_err", "sum"),
        sum_actual=("actual_sales", "sum"),
    ).reset_index()

    out["MAE"] = out["sum_abs_err"] / out["n_obs"]
    out["RMSE"] = np.sqrt(out["sum_sq_err"] / out["n_obs"])
    nonzero = out["sum_actual"] > 0
    out["WAPE"] = np.where(nonzero, out["sum_abs_err"] / out["sum_actual"].where(nonzero), np.nan)
    out["Bias"] = np.where(nonzero, out["sum_signed_err"] / out["sum_actual"].where(nonzero), np.nan)

    out = out.merge(scales[["store", "item", "mase_scale"]], on=["store", "item"], how="left")
    valid = out["mase_scale"].notna() & (out["mase_scale"] > 0)
    out["MASE"] = np.where(valid, out["MAE"] / out["mase_scale"].where(valid), np.nan)

    # Column order keeps the old ones (store, item, model, MAE, RMSE) first
    # so existing consumers (the dashboard) keep working.
    return out[
        ["store", "item", "model", "MAE", "RMSE", "WAPE", "MASE", "Bias",
         "n_obs", "sum_abs_err", "sum_sq_err", "sum_signed_err", "sum_actual"]
    ]


def pooled_metrics(per_series: pd.DataFrame) -> pd.DataFrame:
    """Model-level summary computed from per-series sums (no ratio averaging).

    MAE/RMSE/WAPE/Bias are pooled over every observation. ``mean_series_MAE``
    and the MASE columns average per-series values and are labelled as such.
    ``series_best`` counts series where the model has the lowest MAE.
    """
    g = per_series.groupby("model", observed=True)
    pooled = g.agg(
        n_series=("MAE", "size"),
        n_obs=("n_obs", "sum"),
        sum_abs_err=("sum_abs_err", "sum"),
        sum_sq_err=("sum_sq_err", "sum"),
        sum_signed_err=("sum_signed_err", "sum"),
        sum_actual=("sum_actual", "sum"),
        mean_series_MAE=("MAE", "mean"),
        mean_series_RMSE=("RMSE", "mean"),
        MASE_mean=("MASE", "mean"),
        MASE_median=("MASE", "median"),
        series_mase_defined=("MASE", "count"),
        series_wape_defined=("WAPE", "count"),
    )
    pooled["MAE"] = pooled["sum_abs_err"] / pooled["n_obs"]
    pooled["RMSE"] = np.sqrt(pooled["sum_sq_err"] / pooled["n_obs"])
    has_demand = pooled["sum_actual"] > 0
    pooled["WAPE"] = (pooled["sum_abs_err"] / pooled["sum_actual"]).where(has_demand)
    pooled["Bias"] = (pooled["sum_signed_err"] / pooled["sum_actual"]).where(has_demand)

    best = per_series.loc[
        per_series.groupby(["store", "item"])["MAE"].idxmin(), "model"
    ].value_counts()
    pooled["series_best"] = best.reindex(pooled.index).fillna(0).astype(int)

    cols = [
        "n_series", "n_obs", "MAE", "RMSE", "WAPE", "Bias",
        "MASE_mean", "MASE_median", "mean_series_MAE", "mean_series_RMSE",
        "series_best", "series_mase_defined", "series_wape_defined",
    ]
    return pooled[cols].sort_values("MAE").reset_index()
