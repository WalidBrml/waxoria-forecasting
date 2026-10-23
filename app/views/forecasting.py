"""Page 3: Demand Forecasting (actual vs predicted, backtest)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data_access as da
from app import theme, ui


def render(data: da.AppData, f: da.Filters) -> None:
    ui.page_header(
        "Demand Forecasting",
        "Actual versus predicted demand for any store, product and model",
    )
    if not data.has_predictions:
        ui.empty_state("Predictions not found", data.status["predictions"].message,
                       da.HOW_TO_CREATE["predictions"])
        return

    pred_f = da.filter_predictions(data.predictions, f)
    if pred_f.empty:
        cov = da.coverage_from_predictions(data.predictions)
        ui.empty_state(
            "No predictions in the selected date range or stores",
            f"Evaluated predictions cover {cov['test_start']} to {cov['test_end']}. "
            "Adjust the sidebar filters to include that period and at least one store.",
        )
        return

    report = data.report or {}
    ui.note(
        "These are backtest forecasts. "
        + report.get("forecast_type", "Each window was predicted using only data available at its origin date.")
        + " No forecast beyond the last observed day is generated yet."
    )

    stores = sorted(int(s) for s in pred_f["store"].unique())
    all_models = [m for m in data.models() if m in set(pred_f["model"].astype(str))]

    ui.sanitize_choice("keep_fc_store", stores)
    c1, c2, c3 = st.columns([1, 1, 2])
    store = c1.selectbox("Store", stores, format_func=lambda s: f"Store {s}", key="keep_fc_store")
    items = sorted(int(i) for i in pred_f.loc[pred_f["store"] == store, "item"].unique())
    ui.sanitize_choice("keep_fc_item", items)
    item = c2.selectbox("Product", items, format_func=lambda i: f"Product {i}", key="keep_fc_item")
    default_models = [m for m in (da.RF_MODEL, "Seasonal naive") if m in all_models] or all_models[:2]
    if "keep_fc_models" not in st.session_state:
        st.session_state["keep_fc_models"] = default_models
    ui.sanitize_multi("keep_fc_models", all_models)
    models = c3.multiselect("Models", all_models, key="keep_fc_models")

    o1, o2 = st.columns(2)
    ui.init_state("keep_fc_context", 28)
    ui.init_state("keep_fc_windows", True)
    context_days = o1.slider("Days of history shown before the evaluation period", 0, 120,
                             key="keep_fc_context")
    show_windows = o2.checkbox("Mark forecast windows", key="keep_fc_windows")

    if not models:
        ui.note("Select at least one model to see predictions.")
        return

    series_all = da.filter_predictions(pred_f, store=store, item=item)
    if series_all.empty:
        ui.empty_state("No predictions for this series in the selected range", "Try another product or date range.")
        return
    sel = series_all[series_all["model"].isin(models)]

    # -------------------------------------------------------------- metrics
    per_series, _ = da.metric_tables(series_all, data.scales)
    per_series = da.order_models(per_series[per_series["model"].isin(models)])
    best = per_series.sort_values("MAE").iloc[0]
    ui.section(f"Store {store} · Product {item}",
               f"{series_all['date'].nunique()} evaluated days · {series_all['forecast_window'].nunique()} windows · "
               f"best selected model by MAE: {best['model']}")
    cards = []
    for _, r in per_series.iterrows():
        cards.append({"label": r["model"], "value": f"MAE {r['MAE']:.2f}",
                      "sub": f"WAPE {ui.fmt_pct(r['WAPE'])} · MASE {ui.fmt_num(r['MASE'])} · Bias {ui.fmt_pct(r['Bias'], signed=True)}",
                      "tone": "navy" if r["model"] == best["model"] else ""})
    ui.kpi_row(cards)

    # ---------------------------------------------------------------- chart
    tab_chart, tab_error, tab_windows, tab_table = st.tabs(
        ["Actual vs predicted", "Forecast error", "Window errors", "Data"]
    )
    actual = series_all.drop_duplicates("date").sort_values("date")[["date", "actual_sales"]]

    with tab_chart:
        fig = go.Figure()
        if context_days > 0:
            start = actual["date"].min() - pd.Timedelta(days=context_days)
            hist = da.series_history(data.sales, store, item) if data.has_sales else pd.Series(dtype=float)
            hist = hist[(hist.index >= start) & (hist.index < actual["date"].min())]
            if not hist.empty:
                fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines",
                                         name="Actual (history)",
                                         line={"color": theme.SILVER, "width": 1.6}))
        fig.add_trace(go.Scatter(x=actual["date"], y=actual["actual_sales"], mode="lines+markers",
                                 name="Actual", line={"color": theme.ACTUAL_COLOR, "width": 2.6},
                                 marker={"size": 4}))
        for m in models:
            g = sel[sel["model"] == m].sort_values("date")
            fig.add_trace(go.Scatter(
                x=g["date"], y=g["predicted_sales"], mode="lines", name=m,
                line={"color": theme.model_color(m), "width": 2.2, "dash": theme.MODEL_DASH.get(m, "solid")},
                customdata=g[["horizon"]],
                hovertemplate="%{y:.1f} (h=%{customdata[0]})<extra>" + m + "</extra>"))
        if show_windows:
            for origin in sorted(series_all["forecast_origin"].unique()):
                x = pd.Timestamp(origin) + pd.Timedelta(hours=12)
                fig.add_shape(type="line", x0=x, x1=x, yref="paper", y0=0, y1=1,
                              line={"color": theme.BORDER, "width": 1, "dash": "dot"})
        fig.update_yaxes(title="Units per day", rangemode="tozero")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(theme.style_fig(fig, "Actual vs predicted units per day", 440),
                        width="stretch", key="fc_main")
        st.caption("Dotted vertical lines mark forecast origins: each 7-day window is predicted "
                   "from data up to its origin. Hover shows the forecast horizon (days ahead).")

    with tab_error:
        fig = go.Figure()
        merged = sel.assign(error=sel["predicted_sales"] - sel["actual_sales"])
        for m in models:
            g = merged[merged["model"] == m].sort_values("date")
            fig.add_trace(go.Bar(x=g["date"], y=g["error"], name=m, marker_color=theme.model_color(m)))
        fig.add_hline(y=0, line_color=theme.NAVY, line_width=1)
        fig.update_layout(barmode="group", hovermode="x unified")
        fig.update_yaxes(title="Predicted − actual (units)")
        st.plotly_chart(theme.style_fig(fig, "Forecast error by day (positive = over-forecast)", 380),
                        width="stretch", key="fc_err")

    with tab_windows:
        wins = da.window_errors(sel)
        ui.note("Mean absolute error per 7-day forecast window (units per day).")
        ui.show_table(wins, column_config={m: st.column_config.NumberColumn(format="%.2f") for m in models})

    with tab_table:
        view = sel.sort_values(["date", "model"]).copy()
        view["date"] = view["date"].dt.date
        view["forecast_origin"] = view["forecast_origin"].dt.date
        ui.show_table(view)
        ui.download_df("Download these predictions (CSV)", sel.sort_values(["date", "model"]),
                       f"waxoria_predictions_store{store}_item{item}.csv", "dl_fc_series")
