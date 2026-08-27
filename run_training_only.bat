@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "VENV_PY=%PROJECT_DIR%.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [ERROR] .venv not found at "%PROJECT_DIR%.venv".
    echo         Run setup_and_run.bat first to create the environment
    echo         and install dependencies.
    exit /b 1
)

echo ============================================================
echo   Training only  (profile: laptop_mx450)
echo ============================================================
"%VENV_PY%" "%PROJECT_DIR%src\train.py" --profile laptop_mx450
set "TRAIN_EXIT=%errorlevel%"

echo.
if "%TRAIN_EXIT%"=="0" (
    echo [DONE] Training finished successfully.
) else (
    echo [ERROR] Training exited with code %TRAIN_EXIT%.
)

exit /b %TRAIN_EXIT%
