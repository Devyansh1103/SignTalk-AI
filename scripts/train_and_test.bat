@echo off
REM SignTalk AI: audit -> train ST-GCN -> evaluate on test split.
REM Run from anywhere:  scripts\train_and_test.bat
REM Run only AFTER convert_include_mvp10.py has finished.

cd /d "%~dp0.."
call .venv\Scripts\activate.bat || (echo [ERROR] .venv not found & exit /b 1)
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs

echo === 1/3 Dataset audit ===
python scripts\audit_dataset.py --videos NONE || (echo [ERROR] audit failed & exit /b 1)

echo === 2/3 Training (CPU, ~5-7 h) - live progress: logs\train_stgcn.log ===
echo     Dekhne ke liye dusri window me:  powershell Get-Content logs\train_stgcn.log -Wait -Tail 20
REM -u = unbuffered, so each epoch line reaches the log immediately instead of at the end
python -u scripts\train_stgcn.py > logs\train_stgcn.log 2>&1
if errorlevel 1 (echo [ERROR] training failed, see logs\train_stgcn.log & exit /b 1)

echo === 3/3 Test evaluation ===
python scripts\evaluate_stgcn.py --split test || (echo [ERROR] evaluation failed & exit /b 1)

echo.
echo Done. Results: results\stgcn\metrics.json, results\stgcn\report.md
echo Model:   experiments\stgcn\checkpoints\best_checkpoint.pt
