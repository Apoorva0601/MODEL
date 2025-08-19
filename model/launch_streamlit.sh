#!/bin/bash

echo "Starting Clinical NLP Streamlit Application..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Python is not installed or not in PATH"
        echo "Please install Python 3.8 or higher"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check if streamlit is installed
$PYTHON_CMD -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Streamlit is not installed. Installing requirements..."
    echo
    $PYTHON_CMD -m pip install -r requirements_streamlit.txt
    echo
fi

# Start the Streamlit application
echo "Starting Streamlit application..."
echo
echo "The application will open in your default web browser."
echo "If it doesn't open automatically, go to: http://localhost:8501"
echo
echo "Press Ctrl+C to stop the application."
echo

streamlit run streamlit_app.py --server.port 8501 --server.headless false
