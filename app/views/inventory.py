"""Page 5: Inventory Intelligence.

The recommendation is stored in ``st.session_state["inventory_result"]`` so it
survives reruns (download clicks, filter changes, switching pages). It is only
replaced when the user presses Calculate again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data_access as da
from app import theme, ui
from src.inventory.recommender import (
    expected_demand_path,
    explain_risk,
    recommend_inventory,
    recommendation_as_dict,
)

RESULT_KEY = "inventory_result"
SRC_HIST = "Historical variability (residual std)"
SRC_RF = "Random Forest 7-day forecast error (saved evaluation)"


def _inputs() -> dict:
    return {
        "store": st.session_state["keep_inv_store"],
        "item": st.session_state["keep_inv_item"],
        "stock": int(st.session_state["keep_inv_stock"]),
        "lead": int(st.session_state["keep_inv_lead"]),
        "review": int(st.session_state["keep_inv_review"]),
        "service_pct": float(st.session_state["keep_inv_service"]),
        "lookback": int(st.session_state["keep_inv_lookback"]),
        "uncertainty": st.session_state["keep_inv_uncertainty"],
    }


def _calculate(data: da.AppData, inp: dict) -> dict:
    history = da.series_history(data.sales, inp["store"], inp["item"])
    if history.empty:
        raise ValueError("No sales history for this store and product.")
    override = None
    if inp["uncertainty"] == SRC_RF:
        override = da.rf_forecast_error_std(data.per_series, inp["store"], inp["item"])
    rec = recommend_inventory(
        history,
        current_stock=inp["stock"],
        lead_time_days=inp["lead"],
        review_period_days=inp["review"],
        service_level=inp["service_pct"] / 100.0,
        lookback_days=inp["lookback"],
        demand_std_override=override,
    )
    path, _ = expected_demand_path(history, inp["lead"] + inp["review"], inp["lookback"])
    return {
        "inputs": inp,
        "rec": rec,
        "path": path,
        "history": history.iloc[-max(120, inp["lookback"]):],
        "as_of": history.index[-1],
    }


def render(data: da.AppData, f: da.Filters) -> None:
    ui.page_header(
        "Inventory Intelligence",
        "Reorder point, target stock and order quantity for a chosen store and product",
    )
    if not data.has_sales:
        ui.empty_state("Sales data not found", data.status["sales"].message, da.HOW_TO_CREATE["sales"])
        return

    stores = sorted(int(s) for s in data.sales["store"].unique())
    items = sorted(int(i) for i in data.sales["item"].unique())
    ui.init_state("keep_inv_stock", 100)
    ui.init_state("keep_inv_lead", 7)
    ui.init_state("keep_inv_review", 7)
    ui.init_state("keep_inv_service", 95.0)
    ui.init_state("keep_inv_lookback", 90)
    ui.sanitize_choice("keep_inv_store", stores)
    ui.sanitize_choice("keep_inv_item", items)

    ui.note(
        "Uses each series' full history up to its last observed day, so the sidebar date filter does not "
        "apply here. Current stock, lead time and review period are your inputs; they are not in the "
        "Kaggle dataset."
    )

    c1, c2, c3 = st.columns(3)
    c1.selectbox("Store", stores, format_func=lambda s: f"Store {s}", key="keep_inv_store")
    c2.selectbox("Product", items, format_func=lambda i: f"Product {i}", key="keep_inv_item")
    c3.number_input("Current on-hand stock (units)", min_value=0, step=10, key="keep_inv_stock")

    c4, c5, c6 = st.columns(3)
    c4.number_input("Supplier lead time (days)", min_value=1, max_value=365, step=1, key="keep_inv_lead")
    c5.number_input("Review period (days)", min_value=1, max_value=90, step=1, key="keep_inv_review")
    c6.slider("Target service level (%)", 80.0, 99.9, step=0.5, format="%.1f%%", key="keep_inv_service",
              help="Probability of not stocking out during the lead time (cycle service level).")

    c7, c8 = st.columns(2)
    c7.slider("Demand look-back (days)", 28, 365, step=7, key="keep_inv_lookback",
              help="Only the most recent days are used to estimate demand.")
    rf_std = da.rf_forecast_error_std(data.per_series, st.session_state["keep_inv_store"],
                                      st.session_state["keep_inv_item"])
    options = [SRC_HIST] + ([SRC_RF] if rf_std is not None else [])
    ui.sanitize_choice("keep_inv_uncertainty", options)
    c8.radio("Demand uncertainty from", options, key="keep_inv_uncertainty",
             help="Safety stock scales with the daily standard deviation chosen here.")
    if rf_std is None:
        c8.caption("Forecast-error option unavailable: no saved Random Forest metrics for this series.")
    else:
        c8.caption(f"Saved Random Forest RMSE for this series: {rf_std:.2f} units/day.")

    inp = _inputs()
    if st.button("Calculate inventory recommendation", type="primary", key="inv_calc"):
        try:
            st.session_state[RESULT_KEY] = _calculate(data, inp)
        except ValueError as exc:
            st.error(f"Could not calculate the recommendation: {exc}")

    result = st.session_state.get(RESULT_KEY)
    if not result:
        ui.empty_state("No recommendation yet",
                       "Set the inputs above and press Calculate. The result stays on screen until you recalculate.")
        _limitations()
        return

    if result["inputs"] != inp:
        st.warning("Inputs have changed since this recommendation was calculated. "
                   "Press Calculate to update it.")
    _show_result(result)
    _limitations()


def _show_result(result: dict) -> None:
    rec, inp = result["rec"], result["inputs"]
    tone = {"High": "bad", "Medium": "warn", "Low": "good"}[rec.stockout_risk]
    action = (f"Order {rec.suggested_order_quantity:,} units now"
              if rec.suggested_order_quantity > 0 else "No order needed at this review")

    ui.section(f"Store {inp['store']} · Product {inp['item']}",
               f"As of {result['as_of'].date()} (last observed day) · on-hand stock {inp['stock']:,} units")
    st.markdown(
        f"{ui.badge(rec.stockout_risk + ' risk', theme.RISK_COLORS[rec.stockout_risk])} &nbsp; **{action}**",
        unsafe_allow_html=True,
    )
    explain = explain_risk(rec.stockout_risk, rec.service_level)
    st.caption(explain)

    ui.kpi_row(
        [
            {"label": "Avg daily demand", "value": f"{rec.average_daily_demand:.1f}",
             "sub": f"last {rec.lookback_days_used} days"},
            {"label": "Safety stock", "value": ui.fmt_int(rec.safety_stock),
             "sub": f"z = {rec.service_factor:.2f}"},
            {"label": "Reorder point", "value": ui.fmt_int(rec.reorder_point),
             "sub": f"covers {rec.lead_time_days}-day lead time"},
            {"label": "Target stock", "value": ui.fmt_int(rec.recommended_stock),
             "sub": f"lead + {rec.review_period_days}-day review"},
            {"label": "Order quantity", "value": ui.fmt_int(rec.suggested_order_quantity),
             "tone": tone},
        ]
    )
    st.write("")
    ui.kpi_row(
        [
            {"label": "Service level", "value": ui.fmt_pct(rec.service_level)},
            {"label": "Days of cover", "value": ui.fmt_num(rec.days_of_cover, 1),
             "sub": "on-hand ÷ avg daily demand"},
            {"label": "Stockout chance in lead time",
             "value": ui.fmt_pct(rec.lead_time_stockout_probability),
             "sub": "if nothing arrives sooner"},
            {"label": "Daily std used", "value": f"{rec.demand_std:.2f}", "sub": rec.uncertainty_source},
            {"label": "Demand model",
             "value": "Weekday" if rec.demand_method == "weekday profile" else "Flat mean",
             "sub": "seasonal profile" if rec.demand_method == "weekday profile" else "no weekday profile"},
        ]
    )

    a, b = st.columns(2)
    with a:
        names = ["Current stock", "Reorder point", "Target stock"]
        vals = [rec.current_stock, rec.reorder_point, rec.recommended_stock]
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                               marker_color=[theme.RISK_COLORS[rec.stockout_risk], theme.BLUE, theme.NAVY],
                               text=[f"{v:,}" for v in vals], textposition="outside"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Units", range=[0, max(vals) * 1.18 if max(vals) > 0 else 1])
        st.plotly_chart(theme.style_fig(fig, "Stock position", 330), width="stretch", key="inv_position")
    with b:
        days = np.arange(0, len(result["path"]) + 1)
        cum = np.concatenate(([0.0], np.cumsum(result["path"])))
        expected_stock = inp["stock"] - cum
        cautious = inp["stock"] - (cum + rec.service_factor * rec.demand_std * np.sqrt(days))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=expected_stock, mode="lines", name="Expected stock",
                                 line={"color": theme.NAVY, "width": 2.6}))
        fig.add_trace(go.Scatter(x=days, y=cautious, mode="lines",
                                 name=f"Stock at {rec.service_level:.0%} service level",
                                 line={"color": theme.BLUE, "dash": "dash", "width": 2}))
        fig.add_hline(y=0, line_color=theme.BAD, line_width=1)
        fig.add_vline(x=rec.lead_time_days, line_dash="dot", line_color=theme.SILVER)
        fig.update_xaxes(title="Days from now (no replenishment)", dtick=max(1, len(days) // 14))
        fig.update_yaxes(title="Units on hand")
        st.plotly_chart(theme.style_fig(fig, "Projected stock, no replenishment (dotted = lead time)", 330),
                        width="stretch", key="inv_projection")

    hist = result["history"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Daily sales",
                             line={"color": theme.BLUE, "width": 1.6}))
    window_start = hist.index[-1] - pd.Timedelta(days=rec.lookback_days_used - 1)
    fig.add_vrect(x0=window_start, x1=hist.index[-1], fillcolor=theme.SKY, opacity=0.6, line_width=0,
                  annotation_text="look-back window", annotation_position="top left")
    fig.add_hline(y=rec.average_daily_demand, line_dash="dash", line_color=theme.NAVY,
                  annotation_text="average", annotation_position="bottom right")
    fig.update_yaxes(title="Units per day", rangemode="tozero")
    st.plotly_chart(theme.style_fig(fig, "Recent demand used for the estimate", 320),
                    width="stretch", key="inv_history")

    row = recommendation_as_dict(rec, store=inp["store"], item=inp["item"],
                                 as_of_date=str(result["as_of"].date()))
    ui.download_df("Download this recommendation (CSV)", pd.DataFrame([row]),
                   f"waxoria_inventory_store{inp['store']}_item{inp['item']}.csv", "dl_inventory")


def _limitations() -> None:
    with st.expander("Method, assumptions and limitations"):
        st.markdown(
            """
**Policy.** Periodic review with an order-up-to level. With lead time *L*, review period *R* and service
factor *z* (Normal quantile of the service level):

- Reorder point = expected demand over *L* + *z* · σ · √*L*
- Target stock = expected demand over *L + R* + *z* · σ · √(*L + R*)
- Order quantity = max(0, target − current stock)

**Risk is consistent with the numbers.** *Low*: no order is needed. *Medium*: an order is needed but stock is
above the reorder point. *High*: an order is needed and stock is at or below the reorder point.

**Demand.** Only the look-back window is used. With at least 8 weeks of data, a weekday profile gives the
expected demand for the coming days, and σ is measured around that profile (weekly seasonality is not treated
as randomness). Otherwise a flat mean is used. No future forecast is used: the saved evaluation holds
backtest predictions, not forecasts beyond the last observed day.

**Limitations.** Normal approximation (weak for slow or intermittent items); daily demands assumed
independent (forecast errors of one origin are in fact positively correlated, so the √n scaling understates
risk for longer lead times); the Random Forest error option uses an RMSE measured on one Q4 backtest period and
includes any bias; no trend, promotions, holidays or price effects; sales are treated as demand, so stockouts hide
demand; lead time is fixed and certain; open purchase orders, minimum order quantities and pack sizes are
ignored. Treat the output as decision support, not an automatic purchase order.
"""
        )
