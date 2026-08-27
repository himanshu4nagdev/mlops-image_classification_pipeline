@echo off
setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "CONDA_ENV_NAME=mlops-image-clf"
set "VENV_DIR=%PROJECT_DIR%.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

echo ============================================================
echo   MLOps Image Classification - Setup and Run
echo ============================================================
echo.

REM ------------------------------------------------------------
REM Step 0: conda must be on PATH. A plain "python -m venv" on
REM top of this machine's base Miniconda interpreter is known to
REM crash torch's CUDA DLLs on import (older bundled msvcp140.dll
REM shadows the newer one torch needs). Building .venv from a real
REM conda environment instead avoids that, since each conda env
REM carries its own modern runtime DLLs.
REM ------------------------------------------------------------
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 'conda' was not found on PATH.
    echo         Run this script from an Anaconda Prompt / a shell where
    echo         Miniconda's Scripts and condabin directories are on PATH.
    exit /b 1
)

REM ------------------------------------------------------------
REM Step 1: ensure the seed conda environment exists
REM ------------------------------------------------------------
conda env list | findstr /r /c:"^%CONDA_ENV_NAME% " >nul 2>&1
if errorlevel 1 (
    echo [INSTALL] Seed conda environment '%CONDA_ENV_NAME%' not found - creating it...
    call conda create -n %CONDA_ENV_NAME% python=3.12 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment '%CONDA_ENV_NAME%'.
        exit /b 1
    )
) else (
    echo [SKIP]    Seed conda environment '%CONDA_ENV_NAME%' already exists.
)

REM ------------------------------------------------------------
REM Step 2: create .venv (seeded from the conda env) if missing
REM ------------------------------------------------------------
if exist "%VENV_PY%" (
    echo [SKIP]    .venv already exists at "%VENV_DIR%".
) else (
    echo [INSTALL] Creating .venv from the '%CONDA_ENV_NAME%' conda environment...
    call conda run -n %CONDA_ENV_NAME% python -m venv "%VENV_DIR%"
    if not exist "%VENV_PY%" (
        echo [ERROR] venv creation failed - "%VENV_PY%" was not created.
        exit /b 1
    )
    echo [INSTALL] .venv created.
)

REM ------------------------------------------------------------
REM Step 3: torch + CUDA check
REM ------------------------------------------------------------
set "TORCH_CHECK="
for /f "usebackq delims=" %%A in (`"%VENV_PY%" -c "import torch; print(torch.cuda.is_available())" 2^>nul`) do set "TORCH_CHECK=%%A"

if "%TORCH_CHECK%"=="True" (
    echo [SKIP]    torch is already installed with working CUDA support.
) else (
    echo [INSTALL] torch with CUDA support not detected - installing CUDA 12.8 build...
    "%VENV_PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 (
        echo [ERROR] torch/torchvision install failed.
        exit /b 1
    )
    set "TORCH_CHECK="
    for /f "usebackq delims=" %%A in (`"%VENV_PY%" -c "import torch; print(torch.cuda.is_available())" 2^>nul`) do set "TORCH_CHECK=%%A"
    if not "!TORCH_CHECK!"=="True" (
        echo [ERROR] torch installed but CUDA is not available afterwards. Check the GPU driver.
        exit /b 1
    )
    echo [INSTALL] torch installed and CUDA verified.
)

REM ------------------------------------------------------------
REM Step 4: remaining requirements
REM ------------------------------------------------------------
"%VENV_PIP%" show transformers datasets mlflow >nul 2>&1
if errorlevel 1 (
    echo [INSTALL] Installing remaining packages from requirements.txt...
    "%VENV_PY%" -m pip install -r "%PROJECT_DIR%requirements.txt"
    if errorlevel 1 (
        echo [ERROR] pip install -r requirements.txt failed.
        exit /b 1
    )
) else (
    echo [SKIP]    Core requirements ^(transformers, datasets, mlflow^) already installed.
)

REM ------------------------------------------------------------
REM Step 5: run the pipeline
REM ------------------------------------------------------------
echo.
echo ============================================================
echo   Running pipeline  (profile: laptop_mx450)
echo ============================================================
"%VENV_PY%" "%PROJECT_DIR%run_pipeline.py" --profile laptop_mx450
set "PIPELINE_EXIT=%errorlevel%"

echo.
if "%PIPELINE_EXIT%"=="0" (
    echo [DONE] Pipeline finished successfully.
) else (
    echo [ERROR] Pipeline exited with code %PIPELINE_EXIT%.
)

exit /b %PIPELINE_EXIT%
