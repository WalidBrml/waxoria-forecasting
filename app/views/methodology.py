"""Page 6: Data & Methodology."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app import data_access as da
from app import ui

DATA_DICTIONARY = pd.DataFrame(
    [
        ("sales_clean.csv", "date", "Calendar day of sales"),
        ("sales_clean.csv", "store", "Store identifier (1-10)"),
        ("sales_clean.csv", "item", "Product identifier (1-50)"),
        ("sales_clean.csv", "sales", "Units sold that day"),
        ("predictions_all_models.csv", "forecast_window", "Index of the 7-day forecast window (1 = earliest)"),
        ("predictions_all_models.csv", "forecast_origin", "Last day of data used to make the forecast"),
        ("predictions_all_models.csv", "horizon", "Days ahead of the origin (1-7)"),
        ("predictions_all_models.csv", "date", "Day being predicted"),
        ("predictions_all_models.csv", "actual_sales", "Observed sales on that day"),
        ("predictions_all_models.csv", "model", "Forecasting model"),
        ("predictions_all_models.csv", "predicted_sales", "Predicted units"),
        ("multi_series_metrics.csv", "MAE, RMSE, WAPE, MASE, Bias", "Metrics per store-product series and model"),
        ("multi_series_metrics.csv", "sum_*", "Error sums that allow exact pooled recomputation"),
        ("model_summary.csv", "*", "Pooled metrics per model over the whole evaluation"),
    ],
    columns=["File", "Column", "Meaning"],
)


def render(data: da.AppData, f: da.Filters) -> None:
    ui.page_header(
        "Data & Methodology",
        "What the numbers are built from, how the models were evaluated, and what to be careful about",
    )

    # ----------------------------------------------------------------- files
    ui.section("Project files", "Everything shown in this dashboard is read from these files")
    rows = []
    for s in data.status.values():
        try:
            shown = str(s.path.relative_to(da.ROOT))
        except ValueError:
            shown = str(s.path)
        rows.append({
            "File": s.label, "Path": shown, "Status": s.state,
            "Rows": ui.fmt_int(s.rows) if s.rows is not None else "-",
            "Size (KB)": ui.fmt_int(s.size_bytes / 1024) if s.size_bytes else "-",
            "Detail": "" if s.ok else s.message,
        })
    ui.show_table(pd.DataFrame(rows))
    missing = [s for s in data.status.values() if not s.ok]
    if missing:
        st.warning("Some files are missing or invalid. Pages that need them show an empty state with the "
                   "command to create them.")
        for cmd in dict.fromkeys(da.HOW_TO_CREATE[s.key] for s in missing):
            st.code(cmd, language="powershell")

    # --------------------------------------------------------------- dataset
    ui.section("Dataset", "Kaggle Store Item Demand Forecasting, computed from sales_clean.csv")
    if data.has_sales:
        s = data.sales
        n_series = s[["store", "item"]].drop_duplicates().shape[0]
        n_days = s["date"].nunique()
        complete = len(s) == n_series * n_days
        ui.kpi_row(
            [
                {"label": "Observations", "value": ui.fmt_int(len(s))},
                {"label": "Series", "value": f"{n_series}",
                 "sub": f"{s['store'].nunique()} stores × {s['item'].nunique()} products"},
                {"label": "Period", "value": f"{s['date'].min().year}–{s['date'].max().year}",
                 "sub": f"{s['date'].min().date()} to {s['date'].max().date()}"},
                {"label": "Total units", "value": ui.fmt_int(s["sales"].sum())},
                {"label": "Complete daily grid", "value": "Yes" if complete else "No",
                 "sub": f"{n_days:,} days per series" if complete else "some series have missing days",
                 "tone": "good" if complete else "warn"},
            ]
        )
        st.caption(da.zero_demand_caption(s))
    else:
        ui.note("Dataset facts are unavailable until sales_clean.csv exists.")

    # ------------------------------------------------------------ evaluation
    ui.section("Evaluation design")
    rep = data.report
    if rep:
        cfg = rep.get("config", {})
        st.markdown(f"**Forecast type.** {rep.get('forecast_type', 'n/a')}")
        st.markdown(
            f"- Horizon: **{cfg.get('horizon', 'n/a')} days**, windows: **{cfg.get('n_windows', 'n/a')}**, "
            f"refit every **{cfg.get('refit_every_windows', 'n/a')}** windows\n"
            f"- Random Forest settings: `{json.dumps(cfg.get('random_forest', {}))}`\n"
            f"- Models: {', '.join(rep.get('models', []))}\n"
            f"- Evaluation runtime recorded in the report: {rep.get('runtime_seconds', 'n/a')} s"
        )
        st.markdown("**Checks recorded by the evaluation run** (written by the pipeline, not re-run here):")
        for c in rep.get("checks_passed", []):
            st.markdown(f"- {c}")
        ui.download_df("Download evaluation coverage (CSV)",
                       pd.DataFrame([rep.get("coverage", {})]), "waxoria_evaluation_coverage.csv", "dl_meth_cov")
        st.download_button("Download evaluation_report.json", data=json.dumps(rep, indent=2),
                           file_name="evaluation_report.json", mime="application/json", key="dl_meth_json")
    else:
        ui.note("evaluation_report.json not found.")
        st.code(da.HOW_TO_CREATE["report"], language="powershell")

    st.markdown(
        "**How leakage is prevented.** Every feature at a forecast origin uses only data up to that day plus the "
        "calendar of the target day. Models are retrained on an expanding history whose labels end at or before "
        "the retrain date, and all four models are scored on identical store-date-horizon rows."
    )

    # --------------------------------------------------------------- metrics
    ui.section("Metrics")
    st.latex(r"\mathrm{MAE}=\frac{1}{n}\sum|p-a|\qquad \mathrm{RMSE}=\sqrt{\frac{1}{n}\sum(p-a)^2}")
    st.latex(r"\mathrm{WAPE}=\frac{\sum|p-a|}{\sum a}\qquad \mathrm{Bias}=\frac{\sum(p-a)}{\sum a}")
    st.latex(r"\mathrm{MASE}=\frac{\mathrm{MAE}}{\frac{1}{T-7}\sum_{t=8}^{T}|y_t-y_{t-7}|}")
    st.markdown(
        "- WAPE and Bias are **pooled as ratios of sums**; they are never averages of per-series ratios.\n"
        "- If a series has no demand in the evaluated window, WAPE and Bias are undefined (n/a). MASE is undefined "
        "when the seasonal-naive scale is 0.\n"
        "- MASE uses the history up to the first forecast origin; below 1 means better than that in-sample "
        "seasonal-naive benchmark."
    )

    # ------------------------------------------------------------- inventory
    ui.section("Inventory method")
    st.latex(r"\mathrm{ROP}=E_L+z\,\sigma\sqrt{L}\qquad S=E_{L+R}+z\,\sigma\sqrt{L+R}\qquad Q=\max(0,\,S-\text{stock})")
    st.markdown(
        "Demand is estimated from a recent look-back window with a weekday profile where the data supports it. "
        "The service level sets *z*. See the assumptions list on the Inventory Intelligence page."
    )

    # ----------------------------------------------------------- limitations
    ui.section("Limitations")
    st.markdown(
        "- The dataset is public and anonymised; stock levels, lead times and costs are not in it, so inventory "
        "inputs are assumptions you supply.\n"
        "- Predictions are backtests on one 12-week period (a single seasonal window). Other periods may rank "
        "models differently.\n"
        "- Every forecast origin in the saved backtest is a Sunday, so horizon and target weekday are confounded, "
        "and the Previous-day baseline (last value held for 7 days) is a weak benchmark. Rolling origins on every "
        "weekday would separate the two effects.\n"
        "- The dashboard does not yet produce forecasts beyond the last observed day.\n"
        "- The Random Forest margin over the baselines is measured on one period and has no confidence interval; "
        "it is consistent across series and windows here, but may not hold in other seasons.\n"
        "- Not covered by any model here: promotions, holidays, prices, stockout-censored demand."
    )

    # ------------------------------------------------------------- reproduce
    ui.section("Reproduce")
    st.code(
        "python -m src.data.validate_data\n"
        "python -m src.forecasting.evaluate_multiple_series --n-jobs 4\n"
        "python -m pytest -q\n"
        "python -m streamlit run app/dashboard.py",
        language="powershell",
    )
    ui.section("Data dictionary")
    ui.show_table(DATA_DICTIONARY)
    ui.download_df("Download data dictionary (CSV)", DATA_DICTIONARY, "waxoria_data_dictionary.csv", "dl_meth_dict")
