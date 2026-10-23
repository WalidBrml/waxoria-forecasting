"""Page 1: Executive Overview."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data_access as da
from app import theme, ui

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def render(data: da.AppData, f: da.Filters) -> None:
    ui.page_header(
        "Executive Overview",
        "Sales performance, forecast quality and data health at a glance",
    )

    if not data.has_sales:
        ui.empty_state(
            "Sales data not found",
            data.status["sales"].message,
            da.HOW_TO_CREATE["sales"],
        )
        return

    sales_f = da.filter_sales(data.sales, f)
    if sales_f.empty:
        ui.empty_state(
            "No sales match the current filters",
            "Widen the date range or select at least one store in the sidebar.",
        )
        return

    daily = sales_f.groupby("date")["sales"].sum()
    total = int(sales_f["sales"].sum())
    yoy = da.yoy_growth(sales_f, f.end)

    ui.section("Sales", f"{f.start.date()} to {f.end.date()} · {len(f.stores)} store(s) selected")
    ui.kpi_row(
        [
            {"label": "Total units sold", "value": ui.fmt_int(total), "tone": "navy"},
            {"label": "Average daily units", "value": ui.fmt_int(daily.mean()),
             "sub": f"across {len(daily):,} days"},
            {"label": "Stores · Products",
             "value": f"{sales_f['store'].nunique()} · {sales_f['item'].nunique()}"},
            {"label": "Growth, last 365 days vs prior",
             "value": ui.fmt_pct(yoy, signed=True),
             "sub": "" if yoy is not None else "needs two full years in range",
             "tone": "" if yoy is None else ("good" if yoy >= 0 else "bad")},
        ]
    )

    # ---------------------------------------------------------- forecasting
    ui.section("Forecast quality", "Backtest on the evaluation period; filters apply")
    pooled = pd.DataFrame()
    if data.has_predictions:
        pred_f = da.filter_predictions(data.predictions, f)
        _, pooled = da.metric_tables(pred_f, data.scales)

    if pooled.empty:
        if not data.has_predictions:
            ui.empty_state(
                "Predictions not found",
                data.status["predictions"].message,
                da.HOW_TO_CREATE["predictions"],
            )
        else:
            cov = da.coverage_from_predictions(data.predictions)
            ui.note(
                "No evaluated predictions fall inside the selected filters. "
                f"The evaluation period is {cov['test_start']} to {cov['test_end']}."
            )
    else:
        pooled = da.order_models(pooled)
        rf = pooled[pooled["model"] == da.RF_MODEL]
        baselines = pooled[pooled["model"] != da.RF_MODEL]
        focus = rf.iloc[0] if not rf.empty else pooled.sort_values("MAE").iloc[0]
        cards = [
            {"label": f"{focus['model']} · WAPE", "value": ui.fmt_pct(focus["WAPE"]),
             "sub": f"MAE {focus['MAE']:.2f} units/day", "tone": "navy"},
        ]
        if not rf.empty and not baselines.empty:
            best_base = baselines.sort_values("MAE").iloc[0]
            gain = da.mae_reduction(focus["MAE"], best_base["MAE"])
            cards.append(
                {"label": "MAE reduction vs best baseline", "value": ui.fmt_pct(gain, signed=True),
                 "sub": f"best baseline: {best_base['model']}",
                 "tone": "good" if gain is not None and gain > 0 else "warn"}
            )
        cards.append(
            {"label": "Bias", "value": ui.fmt_pct(focus["Bias"], signed=True),
             "sub": "negative = under-forecast"}
        )
        cov = da.coverage_from_predictions(pred_f)
        cards.append(
            {"label": "Evaluation coverage", "value": f"{cov['series']:,} series",
             "sub": f"{cov['forecast_windows']} windows · {cov['observations_per_model']:,} obs/model"}
        )
        ui.kpi_row(cards)

    # ---------------------------------------------------------------- charts
    ui.section("Trends")
    left, right = st.columns([3, 2])
    with left:
        monthly = daily.resample("MS").mean()
        fig = go.Figure(go.Scatter(x=monthly.index, y=monthly.values, mode="lines",
                                   line={"color": theme.BLUE, "width": 2.5},
                                   hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra></extra>"))
        fig.update_yaxes(title="Average daily units")
        st.plotly_chart(theme.style_fig(fig, "Monthly average daily units", 340),
                        width="stretch", key="ov_trend")
    with right:
        if pooled.empty:
            ui.note("Model comparison appears here once evaluated predictions are in range.")
        else:
            fig = go.Figure(go.Bar(
                x=pooled["MAE"], y=pooled["model"], orientation="h",
                marker_color=[theme.model_color(m) for m in pooled["model"]],
                text=[f"{v:.2f}" for v in pooled["MAE"]], textposition="outside",
                hovertemplate="%{y}: MAE %{x:.2f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title="MAE (units/day, lower is better)", range=[0, float(pooled["MAE"].max()) * 1.2])
            st.plotly_chart(theme.style_fig(fig, "Model accuracy (MAE)", 340),
                            width="stretch", key="ov_models")

    # -------------------------------------------------------------- insights
    ui.section("What the data says")
    store_tot = sales_f.groupby("store")["sales"].sum().sort_values(ascending=False)
    dow = daily.groupby(daily.index.day_of_week).mean()
    month = daily.groupby(daily.index.month).mean()
    lines = [
        f"Store {store_tot.index[0]} sells the most: {ui.fmt_int(store_tot.iloc[0])} units "
        f"({store_tot.iloc[0] / total:.1%} of the selection).",
        f"{WEEKDAYS[int(dow.idxmax())]} is the busiest weekday "
        f"({dow.max() / dow.mean() - 1:+.0%} vs the weekly average); "
        f"{WEEKDAYS[int(dow.idxmin())]} the quietest ({dow.min() / dow.mean() - 1:+.0%}).",
        f"Demand peaks in {pd.Timestamp(2000, int(month.idxmax()), 1):%B} "
        f"and is lowest in {pd.Timestamp(2000, int(month.idxmin()), 1):%B}.",
    ]
    if not pooled.empty and da.RF_MODEL in set(pooled["model"]) and len(pooled) > 1:
        lines.append(
            f"On the evaluation period, {pooled.sort_values('MAE').iloc[0]['model']} has the "
            "lowest pooled MAE (see Model Performance for all metrics)."
        )
    for line in lines:
        st.markdown(f"- {line}")

    # ----------------------------------------------------------- data health
    ui.section("Data health")
    health = pd.DataFrame(
        [{"File": s.label, "Status": s.state,
          "Rows": ui.fmt_int(s.rows) if s.rows is not None else "-"}
         for s in data.status.values()]
    )
    ui.show_table(health)
