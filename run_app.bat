@echo off
REM CAUTI Risk Prediction App Launcher
REM This script runs the Streamlit app from the streamlit_app directory

echo ========================================
echo  CAUTI Risk Prediction System
echo ========================================
echo.

REM Get the directory where this batch file is located
cd /d "%~dp0"

REM Change to streamlit_app directory
cd streamlit_app

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Streamlit is not installed!
    echo Please install it using: pip install streamlit
    pause
    exit /b 1
)

REM Run the Streamlit app
echo Starting Streamlit app...
echo.
streamlit run app.py

pause

