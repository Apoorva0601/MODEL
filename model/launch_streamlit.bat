@echo off
echo Starting Clinical NLP Streamlit Application...
echo.

:: Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

:: Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo Streamlit is not installed. Installing requirements...
    echo.
    python -m pip install -r requirements_streamlit.txt
    echo.
)

:: Start the Streamlit application
echo Starting Streamlit application...
echo.
echo The application will open in your default web browser.
echo If it doesn't open automatically, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application.
echo.

streamlit run streamlit_app.py --server.port 8501 --server.headless false

pause
