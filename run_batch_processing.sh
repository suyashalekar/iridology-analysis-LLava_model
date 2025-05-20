#!/bin/bash

# Batch processing script for Iridology Analysis System
# Processes all images in the images directory and saves results to the results directory
# Author: Suyash Alekar

# Set environment variables
export PYTHONPATH="$(pwd)"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  echo "Activating virtual environment..."
  source .venv/bin/activate
fi

# Ensure results directory exists
mkdir -p results

# Run the batch processing script
echo "Starting batch processing of all images..."
echo "Results will be saved to the results directory."
python src/batch_process_images.py

echo "Batch processing complete."
echo "Check the results directory for analysis files." 