@echo off
REM Starts the local MNIST app at http://127.0.0.1:8501 and opens the browser.
REM The virtual environment lives in a short path (%USERPROFILE%\.venvs\mnist-a1)
REM because TensorFlow files exceed the Windows 260-character path limit
REM when installed inside this deeply nested folder.
setlocal
cd /d "%~dp0"
set "VENV=%USERPROFILE%\.venvs\mnist-a1"

if not exist "%VENV%\Scripts\python.exe" (
  echo [1/2] Creating virtual environment in "%VENV%" ...
  py -3.12 -m venv "%VENV%" || (echo Python 3.12 is required: https://www.python.org/downloads/ & pause & exit /b 1)
)

"%VENV%\Scripts\python.exe" -c "import tensorflow, streamlit, streamlit_drawable_canvas" 2>nul
if errorlevel 1 (
  echo [2/2] Installing dependencies, this takes a few minutes the first time ...
  "%VENV%\Scripts\python.exe" -m pip install --upgrade pip
  "%VENV%\Scripts\python.exe" -m pip install -r requirements-app.txt || (pause & exit /b 1)
)

echo Opening http://127.0.0.1:8501  (close this window or press Ctrl+C to stop)
"%VENV%\Scripts\python.exe" -m streamlit run web_app.py --server.address 127.0.0.1 --server.port 8501
pause
