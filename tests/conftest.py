"""Shared synthetic data for the Milestone 1 tests (no real dataset needed)."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.rolling_origin import SalesMatrix

START = "2020-01-06"  # a Monday, so day position p has weekday p % 7


def make_dates(n_days: int) -> pd.DatetimeIndex:
    return pd.date_range(START, periods=n_days, freq="D")


def weekly_series(n_days: int, seed: int = 0, noise: bool = True) -> np.ndarray:
    """Strong weekday pattern, optional small noise, never negative."""
    rng = np.random.default_rng(seed)
    base = np.array([10, 12, 14, 16, 20, 30, 25], dtype=float)
    y = np.tile(base, n_days // 7 + 1)[:n_days].copy()
    if noise:
        y += rng.integers(0, 3, n_days)
    return y


@pytest.fixture
def small_matrix() -> SalesMatrix:
    n_days = 400
    keys = pd.DataFrame({"store": [1, 1, 2], "item": [1, 2, 1]})
    values = np.vstack([weekly_series(n_days, seed=s) for s in range(3)])
    return SalesMatrix(keys=keys, dates=make_dates(n_days), values=values)
