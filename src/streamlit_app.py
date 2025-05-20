#!/usr/bin/env python3
"""
Iridology Analysis System - Streamlit Web Interface
Version: 1.0.0

A user-friendly web interface for the hierarchical iridology analysis system.
"""

import os
import sys
import json
import time
import base64
from datetime import datetime
from pathlib import Path
import tempfile
import streamlit as st
from PIL import Image
import pandas as pd

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core functionality
from core.iris_analyze_hierarchical import (
    analyze_iris_image_hierarchical,
    load_hierarchical_data,
    enhanced_aggregate_marker_responses,
    resize_image_before_processing
)

# Set page configuration
st.set_page_config(
    page_title="Iridology Analysis System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_data
def load_config():
    """Load configuration from config.json file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        st.error(f"Error loading config file: {e}")
        return {
            "version": "1.0.0",
            "model": {
                "name": "llava-v1.6-vicuna-13b",
                "api_url": "http://localhost:1234/v1/chat/completions"
            },
            "image_processing": {
                "resize_width": 512,
                "resize_height": 512,
                "jpg_quality": 85
            },
            "analysis": {
                "max_markers_per_system": 3,
                "early_stop_threshold": 2,
                "priority_threshold": 2,
                "token_limits": {
                    "marker_analysis": 200,
                    "summary": 1024,
                    "final_report": 1500
                }
            },
            "data": {
                "iridology_data_file": "iridology-clean.xlsx",
                "results_directory": "results",
                "images_directory": "images"
            }
        }

# Get list of existing images
@st.cache_data
def get_image_list(config):
    """Get list of images in the images directory"""
    images_dir = config['data']['images_directory']
    image_files = []
    
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                      and os.path.isfile(os.path.join(images_dir, f))]
    
    return sorted(image_files)

# Get list of existing analysis results
@st.cache_data
def get_results_list(config):
    """Get list of analysis results files"""
    results_dir = config['data']['results_directory']
    result_files = []
    
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) 
                      if f.lower().endswith('.txt') 
                      and os.path.isfile(os.path.join(results_dir, f))]
    
    return sorted(result_files, reverse=True)  # Most recent first

# Get body systems from the data
@st.cache_data
def get_body_systems(config):
    """Get available body systems from the data file"""
    data_file = config['data']['iridology_data_file']
    
    try:
        df = pd.read_excel(data_file)
        if 'Body System' in df.columns:
            return sorted(df['Body System'].unique())
        else:
            # Default body systems if not found
            return [
                "Digestive", "Circulatory", "Nervous", "Endocrine", 
                "Lymphatic", "Respiratory", "Urinary", "Structural", "Other"
            ]
    except Exception as e:
        st.warning(f"Could not read body systems from data file: {e}")
        # Default body systems
        return [
            "Digestive", "Circulatory", "Nervous", "Endocrine", 
            "Lymphatic", "Respiratory", "Urinary", "Structural", "Other"
        ]

# Analyze iris image
def analyze_iris(image_path, config, selected_systems=None):
    """Analyze an iris image using hierarchical approach"""
    max_markers = config['analysis']['max_markers_per_system']
    early_stop = config['analysis']['early_stop_threshold']
    priority = config['analysis']['priority_threshold']
    resize_width = config['image_processing']['resize_width']
    resize_height = config['image_processing']['resize_height']
    data_file = config['data']['iridology_data_file']
    
    # Start timing
    start_time = time.time()
    
    # Display progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Resize image
    status_text.info("Preprocessing image...")
    progress_bar.progress(10)
    
    original_image_path = image_path
    resized_image_path = resize_image_before_processing(
        image_path,
        target_size=(resize_width, resize_height)
    )
    
    resize_time = time.time() - start_time
    status_text.info(f"Preprocessing image... ({resize_time:.2f}s)")
    
    # Step 2: Load hierarchical data
    status_text.info(f"Loading hierarchical iridology data... ({resize_time:.2f}s)")
    progress_bar.progress(20)
    
    hierarchical_data = load_hierarchical_data(
        data_file,
        priority_threshold=priority,
        max_markers=max_markers
    )
    
    load_data_time = time.time() - start_time
    status_text.info(f"Loading hierarchical iridology data... ({load_data_time:.2f}s)")
    
    # Filter hierarchical data by selected systems if specified
    if selected_systems and selected_systems != ["All"]:
        hierarchical_data = {system: data for system, data in hierarchical_data.items() 
                            if system in selected_systems}
    
    # Step 3: Analyze image
    status_text.info(f"Analyzing iris image with LLaVA model... ({load_data_time:.2f}s)")
    progress_bar.progress(30)
    
    marker_responses = analyze_iris_image_hierarchical(
        resized_image_path,
        hierarchical_data,
        max_markers_per_system=max_markers,
        early_stop_threshold=early_stop
    )
    
    analysis_time = time.time() - start_time
    status_text.info(f"Analyzing iris image with LLaVA model... ({analysis_time:.2f}s)")
    
    # Step 4: Generate report
    status_text.info(f"Creating analysis report... ({analysis_time:.2f}s)")
    progress_bar.progress(90)
    
    final_report = enhanced_aggregate_marker_responses(marker_responses)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    results_dir = config['data']['results_directory']
    os.makedirs(results_dir, exist_ok=True)
    
    output_filename = f"iris_{image_id}_analysis_{timestamp}_v{config['version']}.txt"
    output_path = os.path.join(results_dir, output_filename)
    
    # Calculate total processing time
    total_time = time.time() - start_time
    
    with open(output_path, "w") as f:
        f.write(f"Analysis of iris image: {image_path}\n")
        f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Version: {config['version']}\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Processing Time: {total_time:.2f} seconds\n\n")
        
        # Add detailed timing
        f.write("--- Processing Timing ---\n")
        f.write(f"Image preprocessing: {resize_time:.2f} seconds\n")
        f.write(f"Data loading: {load_data_time - resize_time:.2f} seconds\n")
        f.write(f"Analysis: {analysis_time - load_data_time:.2f} seconds\n")
        f.write(f"Report generation: {total_time - analysis_time:.2f} seconds\n")
        f.write(f"Total processing time: {total_time:.2f} seconds\n\n")
        
        # Add the main report
        f.write(final_report)
    
    # Clean up temporary file
    if resized_image_path != original_image_path and os.path.exists(resized_image_path):
        try:
            os.remove(resized_image_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {resized_image_path}: {e}")
    
    # Complete progress
    progress_bar.progress(100)
    status_text.success(f"Analysis complete in {total_time:.2f} seconds")
    
    return final_report, output_path, total_time

def parse_report_for_visualization(marker_responses):
    """Parse marker responses for visualization."""
    # Simplified for now - returns counts by body system and status
    body_systems = {}
    
    for resp in marker_responses:
        system = resp.get('body_system', 'Other')
        is_positive = resp.get('is_positive', False)
        
        if system not in body_systems:
            body_systems[system] = {'positive': 0, 'negative': 0, 'total': 0}
        
        body_systems[system]['total'] += 1
        if is_positive:
            body_systems[system]['positive'] += 1
        else:
            body_systems[system]['negative'] += 1
    
    return body_systems

def get_download_link(text, filename, link_text):
    """Generate a download link for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def display_header():
    """Display the application header and description"""
    col1, col2 = st.columns([1, 5])
    
    with col1:
        st.image("https://img.icons8.com/ios-filled/100/000000/iris-scan.png", width=60)
    
    with col2:
        st.title("Iridology Analysis System")
    
    st.markdown("""
    This application uses LLaVA-1.6-Vicuna-13B with a hierarchical analysis approach to detect health markers in iris images. The hierarchical approach organizes markers by body system for more efficient analysis and better organized results.
    """)
    
    # Display version and model info in sidebar
    config = load_config()
    st.sidebar.markdown(f"**Version:** {config['version']}")
    st.sidebar.markdown(f"**Model:** {config['model']['name']}")

def main():
    # Load configuration
    config = load_config()
    
    # Display header
    display_header()
    
    # Sidebar options
    st.sidebar.title("Settings")
    
    # Analysis mode selection (single image only now)
    st.sidebar.subheader("Analysis Mode")
    analysis_mode = "Single Image"
    st.sidebar.radio(
        "Analysis Mode",
        ["Single Image"],
        index=0,
        key="analysis_mode",
        disabled=True,
        label_visibility="collapsed"
    )
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    
    # Body systems selection
    body_systems = get_body_systems(config)
    all_systems = ["All"] + body_systems
    selected_systems = st.sidebar.multiselect(
        "Body Systems to Analyze",
        all_systems,
        default=["All"],
        key="body_systems"
    )
    
    # Update config parameters with sliders
    priority_threshold = st.sidebar.slider(
        "Priority Threshold",
        min_value=1, max_value=3, value=config['analysis']['priority_threshold'],
        help="Maximum priority level to include (1=highest priority, 3=lowest priority)"
    )
    
    early_stop_threshold = st.sidebar.slider(
        "Early Stop Threshold",
        min_value=1, max_value=5, value=config['analysis']['early_stop_threshold'],
        help="Number of positive findings needed to stop checking a body system"
    )
    
    max_markers = st.sidebar.slider(
        "Max Markers per System",
        min_value=1, max_value=10, value=config['analysis']['max_markers_per_system'],
        help="Maximum number of markers to check per body system"
    )
    
    # Update config with selected values
    config['analysis']['priority_threshold'] = priority_threshold
    config['analysis']['early_stop_threshold'] = early_stop_threshold
    config['analysis']['max_markers_per_system'] = max_markers
    
    # Main content area - Single Image Analysis
    if analysis_mode == "Single Image":
        st.header("Single Image Analysis")
        
        # Option to upload a new image or select an existing one
        upload_tab, existing_tab = st.tabs(["Upload New Image", "Select Existing Image"])
        
        with upload_tab:
            uploaded_file = st.file_uploader("Upload an iris image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file:
                # Save uploaded file to disk
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(temp_path, caption="Uploaded Iris Image", use_container_width=True)
                
                with col2:
                    # Add analysis button
                    if st.button("Analyze Iris Image", key="analyze_uploaded_image"):
                        with st.spinner("Analyzing..."):
                            # Perform analysis with timing
                            final_report, output_path, total_time = analyze_iris(temp_path, config, selected_systems)
                            
                            # Display the results
                            st.markdown(f"### Analysis Results (completed in {total_time:.2f} seconds)")
                            st.text_area("Analysis Report", final_report, height=400)
                            st.success(f"Analysis saved to: {output_path}")
                            st.balloons()
        
        with existing_tab:
            # Get list of existing images
            image_files = get_image_list(config)
            
            if not image_files:
                st.warning("No images found in the images directory.")
            else:
                # Image selection
                selected_image = st.selectbox("Select an image to analyze", image_files)
                
                if selected_image:
                    image_path = os.path.join(config['data']['images_directory'], selected_image)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(image_path, caption=selected_image, use_container_width=True)
                    
                    with col2:
                        # Add analysis button
                        if st.button("Analyze Iris Image", key="analyze_existing_image"):
                            with st.spinner("Analyzing..."):
                                # Perform analysis with timing
                                final_report, output_path, total_time = analyze_iris(image_path, config, selected_systems)
                                
                                # Display the results
                                st.markdown(f"### Analysis Results (completed in {total_time:.2f} seconds)")
                                st.text_area("Analysis Report", final_report, height=400)
                                st.success(f"Analysis saved to: {output_path}")
                                st.balloons()
    
    # Display results viewer (always available)
    st.header("View Analysis Results")
    
    # Get list of existing analysis results
    result_files = get_results_list(config)
    
    if not result_files:
        st.info("No analysis results found. Run an analysis first.")
    else:
        selected_result = st.selectbox("Select a result to view", result_files)
        
        if selected_result:
            result_path = os.path.join(config['data']['results_directory'], selected_result)
            
            with open(result_path, 'r') as f:
                result_content = f.read()
            
            st.text_area("Analysis Report", result_content, height=500)
    
    # Footer
    st.markdown("---")
    st.markdown("Iridology Analysis System v1.1.0 | Hierarchical Approach")

if __name__ == "__main__":
    main() 