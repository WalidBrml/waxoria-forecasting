# Waxoria audit fixes: install notes

The overlay ZIP (`Waxoria-audit-fixes-overlay-v2.zip`) contains ONLY changed or new files, stored with
their real folder paths. It is cumulative: it includes the fixes from the first overlay, so apply it on top of either state. It does not touch `data\` or `reports\`.

```powershell
cd C:\Users\DELL
# keep your existing backup; add a new one for this step
Copy-Item -Recurse -Force .\Waxoria .\Waxoria_backup_before_audit_fixes
Expand-Archive -Path "$HOME\Downloads\Waxoria-audit-fixes-overlay-v2.zip" -DestinationPath C:\Users\DELL\Waxoria -Force
cd C:\Users\DELL\Waxoria
.\.venv\Scripts\Activate.ps1
python -m pytest -q -rw
```

`-rw` prints the warnings summary. If warnings remain, send me the "warnings summary" block.

Check that your data and reports were not changed (hashes are in AUDIT_REPORT.md):

```powershell
Get-FileHash .\reports\predictions_all_models.csv, .\reports\model_summary.csv, .\data\processed\sales_clean.csv -Algorithm SHA256
```
