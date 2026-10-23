
import pandas as pd
import pytest

from src.inventory.recommender import recommend_inventory


def test_recommendation_with_constant_demand():
    sales = pd.Series([10] * 30)

    result = recommend_inventory(
        historical_sales=sales,
        current_stock=50,
        lead_time_days=5,
        review_period_days=5,
        service_factor=1.65,
    )

    assert result.average_daily_demand == 10
    assert result.safety_stock == pytest.approx(0)
    assert result.reorder_point == 50
    assert result.suggested_order_quantity >= 0


def test_low_stock_triggers_high_risk():
    sales = pd.Series([10] * 30)

    result = recommend_inventory(
        historical_sales=sales,
        current_stock=5,
        lead_time_days=7,
    )

    assert result.stockout_risk == "High"
    assert result.suggested_order_quantity > 0


def test_negative_stock_is_rejected():
    with pytest.raises(ValueError):
        recommend_inventory(
            historical_sales=pd.Series([10] * 30),
            current_stock=-1,
        )


def test_empty_history_is_rejected():
    with pytest.raises(ValueError):
        recommend_inventory(
            historical_sales=pd.Series(dtype=float),
            current_stock=20,
        )
