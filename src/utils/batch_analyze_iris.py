#!/usr/bin/env python3
"""
Batch Iris Image Analysis using LLaVA-MLX Client
Processes all iris images in the images directory
"""

import os
import sys
import time
from datetime import datetime
from llava_mlx_client import query_llava, resize_image_if_needed, check_server_status

# LM Studio API settings
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "llava-v1.6-vicuna-13b"
IMAGES_DIR = "images"
OUTPUT_DIR = "iris_analysis_results"

# Prompt template adapted from iris_analyzer_backend.py
IRIDOLOGY_PROMPT = """
Please analyze this eye image as an iridology expert.

IMPORTANT CONTEXT: Iridology examines specific patterns, colors, and structural features in the iris to assess health. The iris is divided into zones that correspond to different body systems:
1. Pupillary zone (inner 1/3): Digestive system, stomach
2. Ciliary zone (middle 1/3): Major organs and circulation
3. Autonomic nerve wreath: Junction between zones - nerve system
4. Iris edge/limbus (outer edge): Skin, lymphatics, circulation

When analyzing this iris, carefully examine:
- COLOR changes (white, yellow, brown, blue) in different zones
- STRUCTURAL features (rings, spots, fibers, lacunae, crypts)
- PATTERN changes (radial lines, dark spots, cloudiness, discoloration)

Check for these common iridology markers:
- Stress rings: White concentric rings showing nervous system stress
- Lymphatic rosary: Small white dots around the outer iris edge
- Scurf rim: Dark rim at the outer edge (skin issues)
- Autonomic nerve wreath: The boundary between pupillary and ciliary zones
- Lacunae: Closed, usually darker areas (potential lesions)
- Radii Solaris: Spoke-like lines radiating outward
- Stomach ring: Ring around the pupil
- Cholesterol ring: White/yellowish ring in the outer iris

Respond with:
1. OVERVIEW: Brief description of the entire iris
2. KEY FINDINGS: List the significant markers observed with their locations
3. ANALYSIS: Potential health implications according to iridology
4. RECOMMENDATIONS: Suggested lifestyle or health considerations based on findings
"""

def batch_analyze_iris_images():
    """Process all iris images in the images directory"""
    # Check if the server is running
    if not check_server_status(API_URL):
        print(f"Error: Cannot connect to LM Studio at {API_URL}")
        print("Make sure LM Studio is running and the model is loaded.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all jpg files in the image directory
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')]
    image_files.sort()  # Sort by filename
    
    print(f"Found {len(image_files)} images to process in {IMAGES_DIR}")
    
    # Create summary report file
    summary_file = os.path.join(OUTPUT_DIR, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_file, "w") as f:
        f.write(f"IRIS ANALYSIS BATCH SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Using model: {MODEL_NAME}\n")
        f.write(f"=================================\n\n")
    
    # Track batch processing stats
    batch_start_time = time.time()
    
    # Process each image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(IMAGES_DIR, image_file)
        
        # Calculate estimated time
        if i > 0:
            elapsed_time = time.time() - batch_start_time
            avg_time_per_image = elapsed_time / i
            remaining_images = len(image_files) - i
            est_remaining_time = avg_time_per_image * remaining_images
            est_completion_time = datetime.now().fromtimestamp(time.time() + est_remaining_time)
            
            print(f"\n\n[{i+1}/{len(image_files)}] Processing image: {image_file}")
            print(f"Elapsed time: {elapsed_time:.1f} sec | Est. completion: {est_completion_time.strftime('%H:%M:%S')}")
        else:
            print(f"\n\n[{i+1}/{len(image_files)}] Processing image: {image_file}")
        
        # Skip if file not found or not readable
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            continue
        
        try:
            # Start timing this image
            image_start_time = time.time()
            
            # Resize image if needed
            resized_image = resize_image_if_needed(image_path, max_size=1024)
            
            # Send to LLaVA for analysis
            print(f"Sending image to LLaVA for analysis...")
            response = query_llava(
                image_path=resized_image,
                prompt=IRIDOLOGY_PROMPT,
                api_url=API_URL,
                model=MODEL_NAME,
                max_tokens=1500,
                temperature=0.1
            )
            
            # Calculate processing time
            image_processing_time = time.time() - image_start_time
            
            # Save individual report
            output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(image_file)[0]}_analysis.txt")
            with open(output_file, "w") as f:
                f.write(f"Analysis of iris image: {image_file}\n")
                f.write(f"Processing time: {image_processing_time:.2f} seconds\n")
                f.write(f"Analyzed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(response)
            
            # Extract quick summary for batch report
            overview = "No overview available"
            if "OVERVIEW:" in response:
                overview_section = response.split("OVERVIEW:")[1].split("\n", 1)[1]
                if "KEY FINDINGS:" in overview_section:
                    overview = overview_section.split("KEY FINDINGS:")[0].strip()
                else:
                    overview = overview_section.split("\n\n")[0].strip()
            
            # Update summary report
            with open(summary_file, "a") as f:
                f.write(f"Image {image_file}:\n")
                f.write(f"• Processing time: {image_processing_time:.1f} seconds\n")
                f.write(f"• Overview: {overview[:150]}...\n")
                f.write(f"• Full analysis saved to: {output_file}\n\n")
            
            print(f"Completed analysis in {image_processing_time:.2f} seconds")
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            # Update summary with error
            with open(summary_file, "a") as f:
                f.write(f"Image {image_file}: ERROR - {str(e)}\n\n")
    
    # Finalize summary report
    total_time = time.time() - batch_start_time
    with open(summary_file, "a") as f:
        f.write("\nBATCH PROCESSING SUMMARY:\n")
        f.write(f"• Total images processed: {len(image_files)}\n")
        f.write(f"• Total processing time: {total_time:.1f} seconds\n")
        f.write(f"• Average time per image: {total_time/len(image_files) if image_files else 0:.1f} seconds\n")
    
    print(f"\nBatch processing complete!")
    print(f"Processed {len(image_files)} images in {total_time:.2f} seconds")
    print(f"Summary report saved to {summary_file}")

if __name__ == "__main__":
    batch_analyze_iris_images() 