import numpy as np
import pandas as pd
import pytest

from src.inventory.recommender import (
    InventoryRecommendation,
    expected_demand_path,
    factor_to_service_level,
    recommend_inventory,
    recommendation_as_dict,
    service_level_to_factor,
)

WEEK = [10, 12, 14, 16, 20, 30, 25]    # Mon..Sun


def weekly_history(weeks=12, noise=None, end_dow_sunday=True):
    """Perfectly weekly demand ending on a Sunday (next day = Monday)."""
    n = weeks * 7
    idx = pd.date_range(end="2020-01-12", periods=n, freq="D")   # 2020-01-12 = Sunday
    y = np.tile(WEEK, weeks).astype(float)
    if noise is not None:
        y = y + noise
    return pd.Series(y, index=idx)


# ------------------------------------------------------ service level

def test_service_level_and_factor_round_trip():
    assert service_level_to_factor(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert factor_to_service_level(1.65) == pytest.approx(0.9505, abs=1e-3)
    assert factor_to_service_level(service_level_to_factor(0.9)) == pytest.approx(0.9)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            service_level_to_factor(bad)


def test_legacy_default_factor_is_exposed_as_a_service_level():
    r = recommend_inventory(pd.Series([10.0, 12.0, 8.0] * 10), current_stock=5)
    assert r.service_factor == pytest.approx(1.65)
    assert r.service_level == pytest.approx(0.9505, abs=1e-3)


def test_higher_service_level_means_more_safety_stock_and_higher_targets():
    rng = np.random.default_rng(0)
    h = pd.Series(rng.integers(5, 25, 60).astype(float))
    lo = recommend_inventory(h, 0, service_level=0.80)
    hi = recommend_inventory(h, 0, service_level=0.99)
    assert hi.safety_stock > lo.safety_stock
    assert hi.reorder_point > lo.reorder_point
    assert hi.recommended_stock > lo.recommended_stock


def test_service_level_and_service_factor_together_is_an_error():
    with pytest.raises(ValueError):
        recommend_inventory(pd.Series([1.0] * 10), 1, service_factor=1.65, service_level=0.95)
    with pytest.raises(ValueError):
        recommend_inventory(pd.Series([1.0] * 10), 1, service_level=1.0)


# ----------------------------------------- consistency of ROP/target/risk

@pytest.mark.parametrize("seed", range(25))
def test_reorder_point_target_and_risk_are_consistent(seed):
    rng = np.random.default_rng(seed)
    h = pd.Series(rng.integers(0, 40, int(rng.integers(10, 200))).astype(float))
    lead, review = int(rng.integers(1, 30)), int(rng.integers(1, 30))
    stock = int(rng.integers(0, 600))
    r = recommend_inventory(h, stock, lead, review, service_level=float(rng.uniform(0.5, 0.995)))

    assert r.reorder_point <= r.recommended_stock
    assert r.suggested_order_quantity == max(0, r.recommended_stock - stock)
    if r.stockout_risk == "Low":
        assert r.suggested_order_quantity == 0
    else:
        assert r.suggested_order_quantity > 0
    if r.stockout_risk == "High":
        assert stock <= r.reorder_point
    if r.stockout_risk == "Medium":
        assert r.reorder_point < stock < r.recommended_stock


def test_risk_boundaries_are_exact():
    h = pd.Series([10.0, 14.0] * 20)
    base = recommend_inventory(h, 0, 5, 5, service_level=0.95)
    rop, target = base.reorder_point, base.recommended_stock
    at = lambda s: recommend_inventory(h, s, 5, 5, service_level=0.95)
    assert at(rop).stockout_risk == "High"
    assert at(rop + 1).stockout_risk == "Medium"
    assert at(target - 1).stockout_risk == "Medium"
    assert at(target).stockout_risk == "Low"           # no order needed -> Low
    assert at(target).suggested_order_quantity == 0
    assert at(target - 1).suggested_order_quantity == 1


def test_zero_demand_history_is_low_risk_not_an_error():
    r = recommend_inventory(pd.Series([0.0] * 30), 0)
    assert r.reorder_point == 0 and r.recommended_stock == 0
    assert r.stockout_risk == "Low" and r.suggested_order_quantity == 0
    assert np.isnan(r.days_of_cover)


def test_stockout_probability_matches_the_service_level_at_the_reorder_point():
    rng = np.random.default_rng(1)
    h = pd.Series(rng.normal(50, 10, 90).clip(0))
    r0 = recommend_inventory(h, 0, 9, 7, service_level=0.95)
    r = recommend_inventory(h, r0.reorder_point, 9, 7, service_level=0.95)
    assert r.lead_time_stockout_probability == pytest.approx(0.05, abs=0.005)
    at_mean = recommend_inventory(h, int(round(r.lead_time_demand)), 9, 7)
    assert at_mean.lead_time_stockout_probability == pytest.approx(0.5, abs=0.03)
    assert recommend_inventory(h, 0, 9, 7).lead_time_stockout_probability > 0.99


# --------------------------------------------------------- demand patterns

def test_weekday_profile_is_used_and_respects_the_coming_weekdays():
    h = weekly_history()
    one_day = recommend_inventory(h, 0, lead_time_days=1, review_period_days=1, service_level=0.95)
    assert one_day.demand_method == "weekday profile"
    assert one_day.lead_time_demand == pytest.approx(WEEK[0])             # next day is Monday
    assert one_day.target_demand == pytest.approx(WEEK[0] + WEEK[1])
    assert one_day.demand_std == pytest.approx(0.0)                        # seasonality is not noise
    assert one_day.reorder_point == WEEK[0]

    full_week = recommend_inventory(h, 0, 7, 7)
    assert full_week.lead_time_demand == pytest.approx(sum(WEEK))
    assert full_week.average_daily_demand == pytest.approx(np.mean(WEEK))


def test_weekday_profile_lowers_noise_estimate_vs_flat_mean():
    h = weekly_history()
    flat = recommend_inventory(h.reset_index(drop=True), 0)    # no DatetimeIndex -> flat
    prof = recommend_inventory(h, 0)
    assert flat.demand_method == "flat mean" and prof.demand_method == "weekday profile"
    assert prof.demand_std < flat.demand_std
    assert prof.safety_stock < flat.safety_stock


def test_short_history_falls_back_to_flat_mean():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    r = recommend_inventory(pd.Series(10.0, index=idx), 0)
    assert r.demand_method == "flat mean" and r.lead_time_demand == pytest.approx(70.0)


def test_only_the_recent_lookback_window_is_used():
    old = [100.0] * 200
    new = [10.0] * 60
    h = pd.Series(old + new)
    r = recommend_inventory(h, 0, lead_time_days=5, review_period_days=5, lookback_days=60)
    assert r.average_daily_demand == pytest.approx(10.0) and r.lookback_days_used == 60
    assert r.reorder_point == 50
    all_time = recommend_inventory(h, 0, 5, 5, lookback_days=None)
    assert all_time.average_daily_demand > 50 and all_time.lookback_days_used == 260


def test_supplied_forecast_error_replaces_historical_std():
    h = weekly_history(noise=np.random.default_rng(2).integers(0, 6, 84))
    r = recommend_inventory(h, 0, 4, 7, service_level=0.95, demand_std_override=8.0)
    assert r.uncertainty_source == "supplied forecast error"
    assert r.demand_std == 8.0
    assert r.safety_stock == pytest.approx(r.service_factor * 8.0 * 2.0)   # sqrt(4) = 2
    with pytest.raises(ValueError):
        recommend_inventory(h, 0, demand_std_override=-1.0)


# ------------------------------------------------- compat and export

def test_old_positional_calls_and_fields_still_work():
    r = recommend_inventory(pd.Series([10] * 30), 50, 5, 5, 1.65)
    assert r.reorder_point == 50 and r.safety_stock == pytest.approx(0)
    assert isinstance(r, InventoryRecommendation)
    for field in ("average_daily_demand", "demand_std", "lead_time_days", "safety_stock",
                  "reorder_point", "recommended_stock", "current_stock",
                  "suggested_order_quantity", "stockout_risk"):
        assert hasattr(r, field)


def test_invalid_lookback_is_rejected():
    with pytest.raises(ValueError):
        recommend_inventory(pd.Series([1.0] * 10), 1, lookback_days=1)


def test_recommendation_as_dict_is_flat_and_keeps_extras_first():
    r = recommend_inventory(pd.Series([10.0] * 30), 5)
    d = recommendation_as_dict(r, store=3, item=7)
    assert list(d)[:2] == ["store", "item"]
    assert d["service_level"] == pytest.approx(r.service_level) and d["stockout_risk"] == "High"
    assert all(np.isscalar(v) or isinstance(v, str) for v in d.values())


def test_expected_demand_path_matches_the_recommendation():
    h = weekly_history()
    path, method = expected_demand_path(h, 14)
    r = recommend_inventory(h, 0, lead_time_days=5, review_period_days=9)
    assert method == "weekday profile" and len(path) == 14
    assert path[:5].sum() == pytest.approx(r.lead_time_demand)
    assert path.sum() == pytest.approx(r.target_demand)
    flat, m = expected_demand_path(pd.Series([4.0] * 20), 3)
    assert m == "flat mean" and list(flat) == [4.0, 4.0, 4.0]
