#!/usr/bin/env python3
"""
Batch Image Processing Script for Iridology Analysis
Processes all images in the images directory and saves analysis results to the results directory.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core functionality
from core.iris_analyze_hierarchical import (
    analyze_iris_image_hierarchical,
    load_hierarchical_data,
    enhanced_aggregate_marker_responses,
    resize_image_before_processing
)

def batch_process_all_images():
    """
    Process all images in the images directory and save analysis results.
    """
    # Configuration
    images_dir = 'images'
    results_dir = 'results'
    data_file = 'iridology-clean.xlsx'
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                  and os.path.isfile(os.path.join(images_dir, f))]
    
    # Sort images by name (assuming names are numerical like 1.jpg, 2.jpg, etc.)
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0]))) if any(c.isdigit() for c in x) else float('inf'))
    
    print(f"Found {len(image_files)} images to process")
    
    # Load hierarchical data
    print("\n--- Loading hierarchical iridology data ---")
    hierarchical_data = load_hierarchical_data(
        data_file,
        priority_threshold=2,  # Include priority 1-2 (high and medium)
        max_markers=5          # Up to 5 markers per body system
    )
    
    # Process each image
    start_time = time.time()
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        print(f"\n\n[{i+1}/{len(image_files)}] Processing image: {image_file}")
        
        # Calculate estimated time remaining if not the first image
        if i > 0:
            elapsed_time = time.time() - start_time
            avg_time_per_image = elapsed_time / i
            remaining_images = len(image_files) - i
            est_remaining_time = avg_time_per_image * remaining_images
            est_completion_time = datetime.now() + timedelta(seconds=est_remaining_time)
            
            print(f"Elapsed time: {elapsed_time:.1f} seconds")
            print(f"Estimated time remaining: {est_remaining_time:.1f} seconds")
            print(f"Estimated completion time: {est_completion_time.strftime('%H:%M:%S')}")
        
        # Process the image
        image_start_time = time.time()
        
        try:
            # Resize the image before processing
            original_image_path = image_path
            resized_image_path = resize_image_before_processing(image_path)
            
            # Analyze using hierarchical approach
            print("\n--- Analyzing iris image with hierarchical approach ---")
            marker_responses = analyze_iris_image_hierarchical(
                resized_image_path, 
                hierarchical_data,
                max_markers_per_system=3,  # Analyze up to 3 markers per system
                early_stop_threshold=2     # Stop after finding 2 positive markers in a system
            )
            
            # Generate enhanced report
            print("\n--- Creating analysis report ---")
            final_report = enhanced_aggregate_marker_responses(marker_responses)
            
            # Calculate processing time
            image_processing_time = time.time() - image_start_time
            
            # Create output file path
            output_file = os.path.join(results_dir, f"iris_{image_file.split('.')[0]}_analysis.txt")
            
            # Save the report with debug information at the top
            with open(output_file, "w") as f:
                f.write(f"Analysis of iris image: {image_path}\n")
                f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Processing Time: {image_processing_time:.2f} seconds\n\n")
                
                # Add any debugging information here
                f.write("--- Processing Debug Information ---\n")
                f.write(f"Image size: {os.path.getsize(original_image_path)} bytes\n")
                f.write(f"Resized for processing: {original_image_path != resized_image_path}\n")
                f.write(f"Markers analyzed: {len(marker_responses)}\n")
                f.write(f"Analysis approach: Hierarchical\n\n")
                
                # Add the main report
                f.write(final_report)
            
            print(f"Analysis complete for {image_file} in {image_processing_time:.2f} seconds")
            print(f"Results saved to: {output_file}")
            
            # Clean up the temporary resized image
            if resized_image_path != original_image_path and os.path.exists(resized_image_path):
                try:
                    os.remove(resized_image_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {resized_image_path}: {e}")
                    
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate total processing time
    total_time = time.time() - start_time
    avg_time = total_time / len(image_files) if image_files else 0
    
    print(f"\n--- Batch Processing Complete ---")
    print(f"Processed {len(image_files)} images in {total_time:.2f} seconds")
    print(f"Average processing time per image: {avg_time:.2f} seconds")

if __name__ == "__main__":
    print("Starting batch processing of all images...")
    batch_process_all_images() 