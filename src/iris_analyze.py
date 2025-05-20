#!/usr/bin/env python3
"""
Iridology Analysis System - Main Entry Point
Version: 1.0.0
"""

import os
import sys
import json
import time
from datetime import datetime
import argparse

# Add parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core functionality
from core.iris_analyze_hierarchical import (
    analyze_iris_image_hierarchical,
    load_hierarchical_data,
    enhanced_aggregate_marker_responses,
    resize_image_before_processing
)

def load_config():
    """Load configuration from config.json file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        print("Using default configuration")
        return {
            "version": "1.0.0",
            "model": {
                "name": "llava-v1.6-vicuna-13b",
                "api_url": "http://localhost:1234/v1/chat/completions"
            },
            "image_processing": {
                "resize_width": 512,
                "resize_height": 512
            },
            "analysis": {
                "max_markers_per_system": 3,
                "early_stop_threshold": 2,
                "priority_threshold": 2
            },
            "data": {
                "iridology_data_file": "iridology-clean.xlsx",
                "results_directory": "results",
                "images_directory": "images"
            }
        }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Iridology Analysis System")
    parser.add_argument("image_path", nargs="?", help="Path to the iris image to analyze")
    parser.add_argument("--batch", action="store_true", help="Run batch analysis on all images in the images directory")
    parser.add_argument("--output", help="Custom output file path")
    parser.add_argument("--max-markers", type=int, help="Maximum markers per body system")
    parser.add_argument("--early-stop", type=int, help="Number of positive findings to stop analysis of a body system")
    parser.add_argument("--priority", type=int, help="Maximum priority level to include (1=highest priority)")
    return parser.parse_args()

def analyze_single_image(image_path, config, output_path=None):
    """Analyze a single iris image and generate a report"""
    print(f"Analyzing image: {image_path}")
    print(f"Model: {config['model']['name']}")
    print(f"Version: {config['version']}")
    
    # Set parameters from config
    max_markers = config['analysis']['max_markers_per_system']
    early_stop = config['analysis']['early_stop_threshold']
    priority = config['analysis']['priority_threshold']
    resize_width = config['image_processing']['resize_width']
    resize_height = config['image_processing']['resize_height']
    data_file = config['data']['iridology_data_file']
    
    # Resize the image to reduce token usage
    original_image_path = image_path
    resized_image_path = resize_image_before_processing(
        image_path, 
        target_size=(resize_width, resize_height)
    )
    
    # Load hierarchical data
    print("\n--- Loading hierarchical iridology data ---")
    hierarchical_data = load_hierarchical_data(
        data_file,
        priority_threshold=priority,
        max_markers=max_markers
    )
    
    # Analyze the image
    marker_responses = analyze_iris_image_hierarchical(
        resized_image_path,
        hierarchical_data,
        max_markers_per_system=max_markers,
        early_stop_threshold=early_stop
    )
    
    # Generate report
    print("\n--- Creating Final Report ---")
    final_report = enhanced_aggregate_marker_responses(marker_responses)
    
    # Create output directory if it doesn't exist
    results_dir = config['data']['results_directory']
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate output filename with version and timestamp
    if not output_path:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{image_id}_analysis_{timestamp}_v{config['version']}.txt"
        output_path = os.path.join(results_dir, output_filename)
    
    # Save the report
    with open(output_path, "w") as f:
        f.write(f"Iridology Analysis Report\n")
        f.write(f"Version: {config['version']}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Image: {original_image_path} (resized to {resize_width}x{resize_height})\n\n")
        f.write(final_report)
    
    print(f"\nAnalysis complete!")
    print(f"Report saved to: {output_path}")
    
    # Clean up temporary file
    if resized_image_path != original_image_path:
        try:
            os.remove(resized_image_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file - {e}")
    
    return output_path

def main():
    """Main entry point for the program"""
    args = parse_args()
    config = load_config()
    
    # Override config with command line args if provided
    if args.max_markers:
        config['analysis']['max_markers_per_system'] = args.max_markers
    if args.early_stop:
        config['analysis']['early_stop_threshold'] = args.early_stop
    if args.priority:
        config['analysis']['priority_threshold'] = args.priority
    
    # Determine what to analyze
    if args.batch:
        # Run batch processing (not implemented here - use batch_analyze_iris.py)
        print("Batch processing feature not implemented in this script.")
        print("Please use src/utils/batch_analyze_iris.py for batch processing.")
    elif args.image_path:
        # Analyze single image
        analyze_single_image(args.image_path, config, args.output)
    else:
        # No image specified
        print("Error: Please provide an image path or use --batch option.")
        print("Usage: python iris_analyze.py [image_path] [options]")
        print("For help: python iris_analyze.py --help")

if __name__ == "__main__":
    main() 