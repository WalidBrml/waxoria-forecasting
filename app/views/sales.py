"""Page 2: Sales Intelligence."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import data_access as da
from app import theme, ui

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def render(data: da.AppData, f: da.Filters) -> None:
    ui.page_header(
        "Sales Intelligence",
        "Where, when and what customers buy: trends, seasonality and rankings",
    )
    if not data.has_sales:
        ui.empty_state("Sales data not found", data.status["sales"].message,
                       da.HOW_TO_CREATE["sales"])
        return

    ui.init_state("keep_sales_smooth", "7-day average")
    c1, c2, c3 = st.columns([2, 1, 1])
    all_items = sorted(int(i) for i in data.sales["item"].unique())
    items = c1.multiselect("Products (optional, default all)", all_items,
                           format_func=lambda i: f"Product {i}", key="keep_sales_items")
    grain = c2.radio("Time grain", ["Daily", "Weekly", "Monthly"], horizontal=True,
                     key="keep_sales_grain")
    smooth = c3.selectbox("Smoothing (daily only)", ["None", "7-day average", "28-day average"],
                          key="keep_sales_smooth", disabled=grain != "Daily")

    sales_f = da.filter_sales(data.sales, f, items or None)
    if sales_f.empty:
        ui.empty_state("No sales match the current filters",
                       "Widen the date range, select at least one store, or clear the product filter.")
        return

    daily = sales_f.groupby("date")["sales"].sum()
    store_tot = sales_f.groupby("store")["sales"].sum().sort_values(ascending=False)
    item_tot = sales_f.groupby("item")["sales"].sum().sort_values(ascending=False)
    dow = daily.groupby(daily.index.day_of_week).mean().reindex(range(7))

    ui.kpi_row(
        [
            {"label": "Total units", "value": ui.fmt_int(daily.sum()), "tone": "navy"},
            {"label": "Average daily units", "value": ui.fmt_int(daily.mean())},
            {"label": "Top store", "value": f"Store {store_tot.index[0]}",
             "sub": f"{ui.fmt_int(store_tot.iloc[0])} units"},
            {"label": "Top product", "value": f"Product {item_tot.index[0]}",
             "sub": f"{ui.fmt_int(item_tot.iloc[0])} units"},
            {"label": "Busiest weekday", "value": WEEKDAYS[int(dow.idxmax())],
             "sub": f"{dow.max() / dow.mean() - 1:+.0%} vs average"},
        ]
    )

    # -------------------------------------------------------------- trend
    ui.section("Trend")
    fig = go.Figure()
    if grain == "Daily":
        fig.add_trace(go.Scatter(x=daily.index, y=daily.values, mode="lines", name="Units sold",
                                 line={"color": theme.BLUE_LIGHT if smooth != "None" else theme.BLUE,
                                       "width": 1.2 if smooth != "None" else 2}))
        if smooth != "None":
            window = 7 if smooth.startswith("7") else 28
            roll = daily.rolling(window, min_periods=window).mean()
            fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", name=smooth,
                                     line={"color": theme.NAVY, "width": 2.4}))
        ytitle, title = "Units per day", "Units sold per day"
    else:
        rule = "W" if grain == "Weekly" else "MS"
        agg = daily.resample(rule).mean()
        fig.add_trace(go.Scatter(x=agg.index, y=agg.values, mode="lines", name="Average daily units",
                                 line={"color": theme.BLUE, "width": 2.5}))
        ytitle = "Average units per day"
        title = f"{grain} average of daily units (robust to partial periods)"
    fig.update_yaxes(title=ytitle)
    st.plotly_chart(theme.style_fig(fig, title, 380), width="stretch", key="sales_trend")

    # ------------------------------------------------------- seasonality
    ui.section("Seasonality", "Average units per day, aggregated over the selection")
    a, b = st.columns(2)
    with a:
        colors = [theme.NAVY if i == dow.idxmax() else theme.BLUE_LIGHT for i in range(7)]
        fig = go.Figure(go.Bar(x=WEEKDAYS, y=dow.values, marker_color=colors,
                               hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
        fig.update_yaxes(title="Units per day")
        st.plotly_chart(theme.style_fig(fig, "By day of week", 320), width="stretch", key="sales_dow")
    with b:
        month = daily.groupby(daily.index.month).mean().reindex(range(1, 13))
        colors = [theme.NAVY if m == month.idxmax() else theme.BLUE_LIGHT for m in range(1, 13)]
        fig = go.Figure(go.Bar(x=MONTHS, y=month.values, marker_color=colors,
                               hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
        fig.update_yaxes(title="Units per day")
        st.plotly_chart(theme.style_fig(fig, "By calendar month", 320), width="stretch", key="sales_month")

    # ---------------------------------------------------------- rankings
    ui.section("Rankings")
    a, b = st.columns(2)
    with a:
        order = store_tot.sort_values()
        fig = go.Figure(go.Bar(x=order.values, y=[f"Store {s}" for s in order.index],
                               orientation="h", marker_color=theme.BLUE,
                               hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
        fig.update_xaxes(title="Units sold")
        st.plotly_chart(theme.style_fig(fig, "Units by store", 380), width="stretch", key="sales_stores")
    with b:
        ui.init_state("keep_sales_topn", 10)
        top_n = st.slider("Products shown", 5, 25, key="keep_sales_topn")
        top = item_tot.head(top_n).sort_values()
        fig = go.Figure(go.Bar(x=top.values, y=[f"Product {i}" for i in top.index],
                               orientation="h", marker_color=theme.NAVY_2,
                               hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
        fig.update_xaxes(title="Units sold")
        st.plotly_chart(theme.style_fig(fig, f"Top {top_n} products", 380), width="stretch",
                        key="sales_products")

    # ----------------------------------------------------------- heatmap
    ui.section("Store × month", "Average units per day for each store in each calendar month")
    per_store_day = sales_f.groupby(["store", "date"])["sales"].sum().reset_index()
    per_store_day["month"] = per_store_day["date"].dt.month
    heat = per_store_day.groupby(["store", "month"])["sales"].mean().unstack("month").reindex(columns=range(1, 13))
    fig = go.Figure(go.Heatmap(z=heat.values, x=MONTHS, y=[f"Store {s}" for s in heat.index],
                               colorscale="Blues", colorbar={"title": "Units/day"},
                               hovertemplate="%{y}, %{x}: %{z:,.0f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(theme.style_fig(fig, None, 380), width="stretch", key="sales_heat")

    # --------------------------------------------------------- downloads
    ui.section("Downloads", "CSV exports of the filtered view")
    d1, d2, d3 = st.columns(3)
    with d1:
        out = daily.rename("units_sold").reset_index()
        ui.download_df("Daily totals (CSV)", out, "waxoria_daily_sales.csv", "dl_sales_daily")
    with d2:
        out = store_tot.rename("units_sold").reset_index()
        ui.download_df("Store totals (CSV)", out, "waxoria_store_totals.csv", "dl_sales_store")
    with d3:
        out = item_tot.rename("units_sold").reset_index()
        ui.download_df("Product totals (CSV)", out, "waxoria_product_totals.csv", "dl_sales_item")
