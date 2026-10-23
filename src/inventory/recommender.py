"""Inventory recommendations for Waxoria (periodic-review, order-up-to policy).

Policy
------
* Demand over ``n`` days is modelled as Normal with mean ``E(n)`` and
  standard deviation ``sigma_d * sqrt(n)`` (daily demands assumed independent).
* ``reorder_point``  = E(L)   + z * sigma_d * sqrt(L)
* ``recommended_stock`` (order-up-to target) = E(L+R) + z * sigma_d * sqrt(L+R)
  where L = lead time, R = review period, z = Normal quantile of the service
  level (``service_level = P(no stockout during the lead time)``).
* ``suggested_order_quantity`` = max(0, target - current_stock).

Risk tiers (consistent with the quantities above, by construction)
------------------------------------------------------------------
Low     no order is needed (current_stock >= target)
Medium  an order is needed, but stock is still above the reorder point
High    an order is needed and stock is at or below the reorder point
Because E and the safety stock are non-decreasing in n, reorder_point <=
target always holds, so "High" always implies an order quantity > 0.

Demand estimation (what the data supports)
------------------------------------------
* Only the most recent ``lookback_days`` (default 90) are used, so old
  years do not dilute the current level.
* With a DatetimeIndex and >= 56 days of recent data, a weekday profile is
  used: E(n) sums the mean demand of each coming weekday, starting the day
  after the last observation, and sigma_d is the std of the residuals around
  that profile (weekly seasonality is not counted as randomness).
  Otherwise a flat mean and the plain std are used.
* ``demand_std_override`` lets the caller supply a daily forecast-error
  standard deviation (e.g. the Random Forest RMSE from the saved evaluation)
  in place of the historical residual std.
* No future forecast is used: the saved evaluation contains backtest
  predictions only, not forecasts beyond the last observed day.

Limitations
-----------
Normal approximation (poor for slow-moving or intermittent items); daily
demands treated as independent; trends, promotions, holidays and price
changes are not modelled; sales are treated as demand (stockouts censor
demand); lead time is fixed and certain; ``current_stock`` is on-hand stock
supplied by the user, and open purchase orders are not considered; minimum
order quantities and pack sizes are ignored. Output is decision support, not
an automated purchasing rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

DEFAULT_SERVICE_FACTOR = 1.65
DEFAULT_LOOKBACK_DAYS = 90
MIN_DAYS_FOR_WEEKDAY_PROFILE = 56

_NORMAL = NormalDist()


def service_level_to_factor(service_level: float) -> float:
    """Normal quantile z for a cycle service level in (0, 1)."""
    if not 0.0 < service_level < 1.0:
        raise ValueError("Service level must be strictly between 0 and 1.")
    return float(_NORMAL.inv_cdf(service_level))


def factor_to_service_level(service_factor: float) -> float:
    """Service level implied by a Normal quantile z."""
    return float(_NORMAL.cdf(service_factor))


@dataclass(frozen=True)
class InventoryRecommendation:
    average_daily_demand: float
    demand_std: float
    lead_time_days: int
    safety_stock: float
    reorder_point: int
    recommended_stock: int
    current_stock: int
    suggested_order_quantity: int
    stockout_risk: str
    # --- added in Milestone 2 (defaults keep old positional use working) ---
    review_period_days: int = 0
    service_level: float = float("nan")
    service_factor: float = float("nan")
    demand_method: str = "flat mean"
    uncertainty_source: str = "historical residual std"
    lookback_days_used: int = 0
    lead_time_demand: float = 0.0
    target_demand: float = 0.0
    lead_time_stockout_probability: float = float("nan")
    days_of_cover: float = float("nan")


def explain_risk(risk: str, service_level: float) -> str:
    """Plain-language meaning of a risk tier, consistent with the maths.

    At the reorder point the lead-time stockout probability is about
    ``1 - service_level`` (5% for a 95% target), so "High" means the chance of
    a stockout is at or above the target tolerance, not that it is likely.
    """
    tolerance = f"{(1.0 - service_level) * 100:.1f}%"
    return {
        "High": (
            "Stock is at or below the reorder point, so the chance of running out during the lead time "
            f"is at or above the {tolerance} tolerance implied by the service level, and it rises as stock falls."
        ),
        "Medium": "An order is due to reach the target, but stock is still above the reorder point.",
        "Low": "Stock already covers the order-up-to target for the lead time plus review period.",
    }[risk]


def recommendation_as_dict(
    recommendation: InventoryRecommendation, **extra
) -> dict:
    """Flat dict for CSV export; ``extra`` (store, item, ...) comes first."""
    return {**extra, **asdict(recommendation)}


def _weekday_means(recent: pd.Series) -> np.ndarray | None:
    """Mean demand per weekday (Mon=0), or None if the data cannot support it."""
    if not isinstance(recent.index, pd.DatetimeIndex):
        return None
    if len(recent) < MIN_DAYS_FOR_WEEKDAY_PROFILE:
        return None
    by_day = recent.groupby(recent.index.day_of_week).agg(["mean", "size"])
    if len(by_day) < 7 or (by_day["size"] < 4).any():
        return None
    return by_day["mean"].to_numpy(dtype=float)


def expected_demand_path(
    historical_sales: pd.Series,
    n_days: int,
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
) -> tuple[np.ndarray, str]:
    """Expected demand for each of the next ``n_days`` and the method used.

    Uses exactly the same demand model as ``recommend_inventory`` (so
    ``path[:L].sum()`` equals its ``lead_time_demand``); intended for charts.
    """
    if historical_sales.empty:
        raise ValueError("Historical sales cannot be empty.")
    recent = (
        historical_sales if lookback_days is None else historical_sales.iloc[-lookback_days:]
    ).astype(float)
    weekday_mean = _weekday_means(recent)
    if weekday_mean is None:
        return np.full(n_days, float(recent.mean())), "flat mean"
    last_dow = int(recent.index[-1].day_of_week)
    return weekday_mean[(last_dow + np.arange(1, n_days + 1)) % 7], "weekday profile"


def recommend_inventory(
    historical_sales: pd.Series,
    current_stock: int,
    lead_time_days: int = 7,
    review_period_days: int = 7,
    service_factor: float | None = None,
    *,
    service_level: float | None = None,
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
    demand_std_override: float | None = None,
) -> InventoryRecommendation:
    """Calculate an inventory recommendation from daily sales.

    ``service_level`` (e.g. 0.95) or the legacy ``service_factor`` (z, default
    1.65 ~ 95%) sets the safety stock; passing both is an error.
    See the module docstring for the method and its limitations.
    """
    if historical_sales.empty:
        raise ValueError("Historical sales cannot be empty.")
    if historical_sales.isna().any():
        raise ValueError("Historical sales contain missing values.")
    if (historical_sales < 0).any():
        raise ValueError("Historical sales cannot be negative.")
    if current_stock < 0:
        raise ValueError("Current stock cannot be negative.")
    if lead_time_days < 1:
        raise ValueError("Lead time must be at least 1 day.")
    if review_period_days < 1:
        raise ValueError("Review period must be at least 1 day.")
    if lookback_days is not None and lookback_days < 2:
        raise ValueError("Lookback must be at least 2 days.")
    if demand_std_override is not None and demand_std_override < 0:
        raise ValueError("Demand standard deviation cannot be negative.")

    if service_level is not None and service_factor is not None:
        raise ValueError("Provide either service_level or service_factor, not both.")
    if service_level is not None:
        z = service_level_to_factor(service_level)
    else:
        z = DEFAULT_SERVICE_FACTOR if service_factor is None else float(service_factor)
        if z < 0:
            raise ValueError("Service factor cannot be negative.")
        service_level = factor_to_service_level(z)

    recent = (
        historical_sales if lookback_days is None else historical_sales.iloc[-lookback_days:]
    ).astype(float)

    mean_daily = float(recent.mean())
    weekday_mean = _weekday_means(recent)

    if weekday_mean is not None:
        last_dow = int(recent.index[-1].day_of_week)

        def expected(n: int) -> float:
            dows = (last_dow + np.arange(1, n + 1)) % 7
            return float(weekday_mean[dows].sum())

        residuals = recent - weekday_mean[recent.index.day_of_week]
        hist_std = float(residuals.std(ddof=1))
        method = "weekday profile"
    else:

        def expected(n: int) -> float:
            return mean_daily * n

        hist_std = float(recent.std(ddof=1))
        method = "flat mean"

    if demand_std_override is not None:
        sigma_d, source = float(demand_std_override), "supplied forecast error"
    else:
        sigma_d, source = (0.0 if np.isnan(hist_std) else hist_std), "historical residual std"

    horizon_target = lead_time_days + review_period_days
    lead_demand = expected(lead_time_days)
    target_demand = expected(horizon_target)

    safety_stock = z * sigma_d * np.sqrt(lead_time_days)
    reorder_point = int(np.ceil(lead_demand + safety_stock))
    target_stock = int(
        np.ceil(target_demand + z * sigma_d * np.sqrt(horizon_target))
    )

    order_quantity = max(0, target_stock - int(current_stock))
    if order_quantity == 0:
        risk = "Low"
    elif current_stock <= reorder_point:
        risk = "High"
    else:
        risk = "Medium"

    sigma_lead = sigma_d * np.sqrt(lead_time_days)
    if sigma_lead > 0:
        p_stockout = 1.0 - _NORMAL.cdf((current_stock - lead_demand) / sigma_lead)
    else:
        p_stockout = 1.0 if current_stock < lead_demand else 0.0

    cover = current_stock / mean_daily if mean_daily > 0 else float("nan")

    return InventoryRecommendation(
        average_daily_demand=mean_daily,
        demand_std=sigma_d,
        lead_time_days=lead_time_days,
        safety_stock=float(safety_stock),
        reorder_point=reorder_point,
        recommended_stock=target_stock,
        current_stock=int(current_stock),
        suggested_order_quantity=order_quantity,
        stockout_risk=risk,
        review_period_days=review_period_days,
        service_level=float(service_level),
        service_factor=float(z),
        demand_method=method,
        uncertainty_source=source,
        lookback_days_used=len(recent),
        lead_time_demand=float(lead_demand),
        target_demand=float(target_demand),
        lead_time_stockout_probability=float(p_stockout),
        days_of_cover=float(cover),
    )
