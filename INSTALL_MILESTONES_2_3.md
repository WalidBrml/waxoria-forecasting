# Waxoria: Milestones 2 and 3 (inventory engine + six-page dashboard)

This ZIP contains only the files added or replaced since Milestone 1. No new
Python packages are required (streamlit, plotly, pandas, scikit-learn, pytest
are already in requirements.txt). The expensive forecasting evaluation is NOT
re-run: the dashboard reads your existing files in `reports\`.

## 1. Back up and install (PowerShell)

```powershell
cd C:\Users\DELL
Copy-Item -Recurse -Force .\Waxoria .\Waxoria_backup_before_m2_m3
Expand-Archive -Path "$HOME\Downloads\Waxoria-milestones-2-3.zip" -DestinationPath C:\Users\DELL\Waxoria -Force
cd C:\Users\DELL\Waxoria
.\.venv\Scripts\Activate.ps1
```

Adjust the ZIP path if you saved it elsewhere. `-Force` overwrites
`app\dashboard.py` and `src\inventory\recommender.py` (both are backed up above).

## 2. Run the tests

```powershell
python -m pytest -q
```

Expected: all tests pass (118 when run in the Linux environment where this was
built; the count should be identical on your machine). The dashboard app tests
take about 30 seconds in total because they run every page headlessly. They are
skipped automatically if the data or report files are missing.

## 3. Start the dashboard

```powershell
python -m streamlit run app/dashboard.py
```

Open http://localhost:8501. Pages: Executive Overview, Sales Intelligence,
Demand Forecasting, Model Performance, Inventory Intelligence, Data & Methodology.

## 4. Files

| Path | Status |
|---|---|
| src/inventory/recommender.py | replacement (old calls still work) |
| app/dashboard.py | replacement (entry point + navigation + filters) |
| app/data_access.py, app/theme.py, app/ui.py | new |
| app/views/__init__.py, overview.py, sales.py, forecasting.py, performance.py, inventory.py, methodology.py | new |
| .streamlit/config.toml | new (navy/blue theme) |
| tests/test_inventory_policy.py, tests/test_dashboard_data.py, tests/test_dashboard_app.py | new |

## 5. Required input files (created by Milestone 1; not in this ZIP)

data\processed\sales_clean.csv, reports\predictions_all_models.csv,
reports\multi_series_metrics.csv, reports\model_summary.csv,
reports\evaluation_report.json. If one is missing, the affected pages show an
empty state with the command that creates it; the app does not crash.
