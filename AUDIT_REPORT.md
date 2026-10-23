# Waxoria audit report (revision 2)

## Revision 2: weekday deprecation (from your Windows run)
Your run: 124 passed, 1 failed (test_project_code_raises_no_deprecation_warnings), 586 warnings.
I did not receive the traceback or warnings text, only the counts and your description, so I could not
inspect them directly. Instead I searched every .py file for the weekday attribute `dayofweek`.
Found and replaced with `day_of_week` (same values, Monday = 0):
- src/features/demand_features.py:188 (the `dow` model feature)
- src/inventory/recommender.py (4 places: weekday profile, residuals, expected demand path)
- app/views/overview.py, app/views/sales.py (day-of-week charts)
- tests/test_features.py (3 places; assertions unchanged)
Reproduction: pandas 3.0.2 here does not warn, so I patched DatetimeIndex.dayofweek to emit a Pandas4Warning.
Before the fix that produced the same failure shape (557 warnings in demand_features.py:188,
recommender.py:129/215, test_features.py:126; the audit test failed). After the fix: 0 warnings, test passes.
Your 586 vs my 557 simulated: the gap is most plausibly the Timestamp uses (recommender line 153, now fixed),
the dashboard tests I cannot run, and the added audit test's own run. I cannot confirm that split.
Whether the 586 contain any OTHER independent warning is NOT established: I only know about weekday access.
Re-run `python -m pytest -q -rw` and send the summary if anything remains.
New tests: a static AST guard against `.dayofweek` anywhere in the project (also catches Timestamp and
unexecuted code), and a weekday-correctness test (features, weekday profile, expected path). Mutation check:
putting one `.dayofweek` back makes the static guard fail.


## What was and was not executed
- Executed (revision 2): 116 tests passed, 0 failed, using a minimal pytest
  stand-in (real pytest is not installable here; no network). Result: 114 passed, 0 failed.
- NOT executed: the 11 tests in tests/test_dashboard_app.py (need Streamlit; not installed). Real pytest, real
  Streamlit, plotly, statsmodels and the pinned pandas 3.1.0 / numpy 2.5.4 / scikit-learn 1.9.1 were unavailable.
  I ran on pandas 3.0.2 / numpy 2.4.4 / scikit-learn 1.8.0 / Python 3.12.
- Also executed: every dashboard view's pandas code against the real data with fake Streamlit/plotly objects
  (no browser; this checks the data code only, not rendering).
- The 500-series evaluation was NOT re-run. Saved outputs were verified, not regenerated.

## Archive finding
Every entry in Waxoria-current.zip is stored without its folder path (all 47 files at the root, six
`__init__.py` collapsed in name). I rebuilt the tree from imports and path constants
(app/, app/views/, src/{data,features,forecasting,inventory}/, tests/, data/{raw,processed}/, reports/, .streamlit/).
Present: application code, 9 test files + conftest, dataset (train/test/sample_submission, sales_clean),
saved outputs, and the six-page dashboard. Empty: README.md and .gitignore (0 bytes).

## Verified OK
- sales_clean.csv equals train.csv (913,000 rows, 10x50 series, 1,826 days, no gaps, no NaN).
- All saved metrics recompute exactly from predictions_all_models.csv (model_summary, multi_series_metrics).
- actual_sales matches the sales history on every prediction row; 42,000 rows per model on identical keys.
- Random Forest beats the 7-day average in 11/12 windows; per-series MAE gain 1.72 units (SE 0.05), 97.6% of series.
- Identical Bias for Seasonal naive and 7-day average (0.0202) is correct by construction, not an error.

## Confirmed problems and what changed
1. Inventory "High" text said lead-time demand is "likely to exceed" stock. At the reorder point the stockout
   probability is about 1 - service level (about 5%). Reworded via `explain_risk` (recommender.py), tested.
2. Overview card "MAE vs best baseline +19.3%" read as higher error. Now "MAE reduction vs best baseline" via
   `da.mae_reduction`, tested.
3. Data & Methodology caption hard-coded "No series is entirely zero" next to a computed count. Now
   `da.zero_demand_caption`, true for any data, tested.
4. Legacy src/forecasting/baseline.py zero-filled missing days (`asfreq("D").fillna(0)`), contradicting the
   pipeline's no-imputation rule. Now `daily_series` raises on gaps, tested. (Current data has no gaps.)
5. Inventory and Methodology text now state that the Random Forest RMSE option scales by sqrt(n) although
   errors from one origin are correlated, that it comes from one Q4 period and includes bias, and that the
   backtest margin has no confidence interval.
6. .gitignore was empty; now ignores .venv, caches.

## Methodological weaknesses (documented, not changed)
- One 12-week test period (Oct 9 to Dec 31, 2017); every origin is a Sunday, so horizon and weekday are confounded.
- Previous-day baseline is a weak benchmark for h = 2..7.
- Inventory demand uses the last 90 days (no trend/seasonality); Jan demand is lower than Q4 in this dataset.
- Reorder point mixes continuous-review logic with an order-up-to (periodic review) target.
- reports/model_comparison.csv, forecast_metrics.csv, forecast_predictions.csv and figures/forecast_comparison.html
  are stale Milestone 1 single-series files with a different protocol and older schema; the dashboard does not read
  them. Not regenerated, to preserve your outputs.

## (Revision 1 note, superseded above) 580 pandas warnings: not reproduced on pandas 3.0.2
With pandas 3.0.2 the full suite and all view code produced 0 warnings (recorded with "always", nothing suppressed).
I found no deprecated pandas call by static search either, so I changed nothing for this item. The warnings
likely come from pandas 3.1.0 (pinned in requirements.txt), Streamlit/plotly internals, or the AppTest runs.
I added tests/test_audit_fixes.py::test_project_code_raises_no_deprecation_warnings, which fails if any
FutureWarning/DeprecationWarning/Pandas4Warning is attributed to a project file. To fix the real ones, run
`python -m pytest -q -rw` on Windows and send the "warnings summary" block.

## Files changed (cumulative against your original upload)
.gitignore, INSTALL_AUDIT_FIXES.md (new), AUDIT_REPORT.md (new), app/data_access.py, app/views/inventory.py,
app/views/methodology.py, app/views/overview.py, app/views/sales.py, src/features/demand_features.py,
src/forecasting/baseline.py, src/inventory/recommender.py, tests/test_features.py,
tests/test_audit_fixes.py (new, 9 tests).

## SHA-256 (first 16 hex) of preserved files, identical before and after
sales_clean.csv 4f0a6c57bde0df6e; predictions_all_models.csv 8e7786390259f18c; multi_series_metrics.csv 62bf5db949f00476;
model_summary.csv d9b0c60b24be0cdd; evaluation_report.json 9dbc0bc19f2fcc01.
