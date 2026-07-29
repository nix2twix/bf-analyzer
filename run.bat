@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

set "CURRENT_DIR=%~dp0"
set "CURRENT_DIR=%CURRENT_DIR:~0,-1%"

set "USERNAME=%USERNAME%"

set "VENV_BASE=C:\Users\%USERNAME%\Projects"
set "VENV_DIR=%VENV_BASE%\bf_analyzer"

set "REQUIREMENTS=%CURRENT_DIR%\requirements.txt"

set "APP_DIR=%CURRENT_DIR%\app"

set NO_ALBUMENTATIONS_UPDATE=1

set PIP_DEFAULT_TIMEOUT=300
set PIP_RETRIES=10
set PIP_INDEX_URL=https://pypi.org/simple/
set PIP_TRUSTED_HOST=pypi.org
set PIP_TRUSTED_HOST=%PIP_TRUSTED_HOST% files.pythonhosted.org

python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10 from:
    echo https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
echo [INFO] Python version: %PYTHON_VER%

if not exist "%VENV_BASE%" (
    echo [INFO] Creating projects directory: %VENV_BASE%
    mkdir "%VENV_BASE%"
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment at %VENV_DIR%...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully
) else (
    echo [INFO] Virtual environment already exists at %VENV_DIR%
)

echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

if not defined VIRTUAL_ENV (
    echo [WARNING] Virtual environment activation failed.
    echo [INFO] Trying alternative activation method...
    call "%VENV_DIR%\Scripts\activate.bat"
    if not defined VIRTUAL_ENV (
        echo [ERROR] Cannot activate virtual environment
        pause
        exit /b 1
    )
)
echo [INFO] Virtual environment activated: %VIRTUAL_ENV%

echo [INFO] Checking dependencies...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo.
    echo [INFO] Installing dependencies from requirements.txt...
    echo.
    
    echo [INFO] Updating pip...
    python -m pip install --upgrade pip
    pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
    
    echo.
    echo [INFO] Dependencies installation completed!
) else (
    echo [INFO] Dependencies already installed
)

set PYTHONPATH=%CURRENT_DIR%;%PYTHONPATH%
cd /d "%CURRENT_DIR%"

echo [DEBUG] Current directory: %CD%
echo [DEBUG] Python path: %PYTHONPATH%
echo.

echo [INFO] Starting Streamlit server on http://localhost:8501
echo.

python -m streamlit run app.py --server.address=localhost --server.port=8501

if errorlevel 1 (
    echo.
    echo [WARNING] First launch attempt failed, trying alternative method...
    streamlit run app.py --server.address=localhost --server.port=8501
)

if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed!
    echo.
    echo ========================================
    echo Manual launch instructions:
    echo ========================================
    echo 1. Open Command Prompt
    echo 2. Run: call "%VENV_DIR%\Scripts\activate.bat"
    echo 3. Run: cd /d "%CURRENT_DIR%"
    echo 4. Run: set PYTHONPATH=%CURRENT_DIR%
    echo 5. Run: streamlit run app.py
    echo ========================================
)

echo.
echo [INFO] Application stopped
pause