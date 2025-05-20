#!/bin/bash

# Run Streamlit app for Iridology Analysis System
# Author: Suyash Alekar
# Version: 1.0.0

# Set environment variables
export PYTHONPATH="$(pwd)"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  echo "Activating virtual environment..."
  source .venv/bin/activate
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
  echo "Streamlit not found. Installing required packages..."
  pip install -r requirements.txt
fi

# Run the Streamlit app
echo "Starting Streamlit app for Iridology Analysis..."
streamlit run src/streamlit_app.py "$@" 