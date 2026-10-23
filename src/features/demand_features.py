"""Shared, leakage-safe demand features for Waxoria.

Every forecasting script (Random Forest and baselines) builds its inputs
here, so a model and the baseline it is compared with can never disagree
about what information was available.

Forecast framing
----------------
A forecast is issued at an *origin* day ``T`` (the last day whose sales are
known) for a *target* day ``t = T + h`` with horizon ``h`` in ``1..H``.
Predicting several horizons from one origin is a "direct multi-horizon"
setup: the horizon ``h`` is itself a feature.

Leakage rule
------------
Every feature is a function of ``values[0..T]`` plus the calendar of the
target day (day of week and month are known in advance). Nothing after the
origin is ever read. Positions are integer offsets into the daily series,
and the helper that picks seasonal lags guarantees ``source <= T``.

Feature semantics (for origin ``T`` and target ``t = T + h``)
------------------------------------------------------------
horizon          h, number of days between origin and target.
dow              Day of week of the *target* (Monday=0 ... Sunday=6).
month            Month of the *target* (1..12).
last_value       sales on the origin day T (the "previous-day" signal).
mean_7           mean of sales on days T-6..T (trailing 7 days incl. T).
mean_28          mean of sales on days T-27..T.
seasonal_lag_1   sales on the most recent day <= T with the same weekday as
                 t, i.e. ``t - 7*ceil(h/7)``. For h in 1..7 this is t-7.
seasonal_lag_2   the same weekday one week earlier than seasonal_lag_1.
seasonal_mean_4  mean of the four most recent same-weekday observations
                 (seasonal_lag_1 .. seasonal_lag_4).

Baselines are *columns of the same feature table*:
    Previous-day baseline  -> last_value
    7-day moving average   -> mean_7
    Seasonal naive         -> seasonal_lag_1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASON = 7
SHORT_WINDOW = 7
LONG_WINDOW = 28
N_SEASONAL_LAGS = 4

FEATURE_COLUMNS: list[str] = [
    "horizon",
    "dow",
    "month",
    "last_value",
    "mean_7",
    "mean_28",
    "seasonal_lag_1",
    "seasonal_lag_2",
    "seasonal_mean_4",
]

# Baseline name -> feature column holding that baseline's forecast.
# Names for the first two match the labels already used by the dashboard.
BASELINE_COLUMNS: dict[str, str] = {
    "Previous-day baseline": "last_value",
    "7-day moving average": "mean_7",
    "Seasonal naive": "seasonal_lag_1",
}


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of ``values[i-window+1 .. i]`` for each i (NaN while i < window-1).

    Element ``i`` depends only on ``values[:i+1]``.
    """
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    if window < 1:
        raise ValueError("window must be >= 1")
    if len(v) >= window:
        csum = np.concatenate(([0.0], np.cumsum(v)))
        out[window - 1:] = (csum[window:] - csum[:-window]) / window
    return out


def seasonal_step(horizon_days: np.ndarray | int) -> np.ndarray | int:
    """Smallest multiple of 7 that is >= h: ``7 * ceil(h / 7)``.

    ``target - seasonal_step(h)`` is the latest same-weekday day that is
    guaranteed to be at or before the origin.
    """
    h = np.asarray(horizon_days)
    step = SEASON * (-(-h // SEASON))
    return step if isinstance(horizon_days, np.ndarray) else int(step)


def min_origin(horizon: int) -> int:
    """Earliest origin position with all lags/windows fully available."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    h = np.arange(1, horizon + 1)
    deepest = seasonal_step(h) + SEASON * (N_SEASONAL_LAGS - 1) - h
    return int(max(LONG_WINDOW - 1, deepest.max()))


def _expand(origins: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Row layout: for each origin, horizons 1..H consecutively."""
    origin_rep = np.repeat(origins, horizon)
    h_rep = np.tile(np.arange(1, horizon + 1), len(origins))
    return origin_rep, h_rep


def _check_origins(
    values: np.ndarray, origins: np.ndarray, horizon: int, n_dates: int
) -> None:
    if origins.ndim != 1 or len(origins) == 0:
        raise ValueError("origins must be a non-empty 1-D array")
    if origins.min() < min_origin(horizon):
        raise ValueError(
            f"origin {origins.min()} has too little history; "
            f"minimum is {min_origin(horizon)} for horizon {horizon}"
        )
    if origins.max() > len(values) - 1:
        raise ValueError("origin lies beyond the observed values")
    if origins.max() + horizon > n_dates - 1:
        raise ValueError("calendar `dates` must cover origin + horizon")


def seasonal_naive_forecast(
    values: np.ndarray, origins: np.ndarray, horizon: int
) -> np.ndarray:
    """Seasonal-naive forecast: same weekday from the most recent week(s).

    For horizons 1..7 this is ``y[t-7]``. For longer horizons it steps back
    in whole weeks until the source day is at or before the origin. Rows are
    ordered by origin, then horizon (same layout as ``build_origin_features``).
    """
    values = np.asarray(values, dtype=float)
    origins = np.asarray(origins, dtype=int)
    if origins.min() < min_origin(horizon) or origins.max() > len(values) - 1:
        raise ValueError("origin outside the valid range")
    origin_rep, h_rep = _expand(origins, horizon)
    target = origin_rep + h_rep
    return values[target - seasonal_step(h_rep)]


def build_origin_features(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    origins: np.ndarray,
    horizon: int,
) -> pd.DataFrame:
    """Feature table with one row per (origin, horizon).

    Parameters
    ----------
    values:  daily sales of ONE series, ``values[i]`` belongs to ``dates[i]``.
             Only positions <= each origin are read.
    dates:   daily calendar; may extend past ``values`` (future targets).
    origins: integer positions of the last known day of each forecast.
    horizon: forecast horizon H; rows for h = 1..H are produced per origin.

    Returns a DataFrame with ``origin_pos``, ``target_pos`` and every column
    in ``FEATURE_COLUMNS``, ordered by origin then horizon.
    """
    values = np.asarray(values, dtype=float)
    origins = np.asarray(origins, dtype=int)
    dates = pd.DatetimeIndex(dates)
    _check_origins(values, origins, horizon, len(dates))

    origin_rep, h_rep = _expand(origins, horizon)
    target = origin_rep + h_rep
    step = seasonal_step(h_rep)

    lags = [values[target - step - SEASON * k] for k in range(N_SEASONAL_LAGS)]
    # Guard against alignment errors: every source day must be known.
    assert (target - step <= origin_rep).all()

    mean_7 = trailing_mean(values, SHORT_WINDOW)[origin_rep]
    mean_28 = trailing_mean(values, LONG_WINDOW)[origin_rep]

    frame = pd.DataFrame(
        {
            "origin_pos": origin_rep,
            "target_pos": target,
            "horizon": h_rep.astype(float),
            "dow": dates.day_of_week.to_numpy()[target].astype(float),
            "month": dates.month.to_numpy()[target].astype(float),
            "last_value": values[origin_rep],
            "mean_7": mean_7,
            "mean_28": mean_28,
            "seasonal_lag_1": lags[0],
            "seasonal_lag_2": lags[1],
            "seasonal_mean_4": np.mean(lags, axis=0),
        }
    )
    return frame
