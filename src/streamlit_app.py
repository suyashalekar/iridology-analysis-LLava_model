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
    
    # Display progress
    progress_bar = st.progress(0)
    
    # Step 1: Resize image
    st.info("Preprocessing image...")
    progress_bar.progress(10)
    
    original_image_path = image_path
    resized_image_path = resize_image_before_processing(
        image_path,
        target_size=(resize_width, resize_height)
    )
    
    # Step 2: Load hierarchical data
    st.info("Loading hierarchical iridology data...")
    progress_bar.progress(20)
    
    hierarchical_data = load_hierarchical_data(
        data_file,
        priority_threshold=priority,
        max_markers=max_markers
    )
    
    # Filter hierarchical data by selected systems if specified
    if selected_systems and selected_systems != ["All"]:
        hierarchical_data = {system: data for system, data in hierarchical_data.items() 
                            if system in selected_systems}
    
    # Step 3: Analyze image
    st.info("Analyzing iris image with LLaVA model...")
    progress_bar.progress(30)
    
    marker_responses = analyze_iris_image_hierarchical(
        resized_image_path,
        hierarchical_data,
        max_markers_per_system=max_markers,
        early_stop_threshold=early_stop
    )
    
    # Step 4: Generate report
    st.info("Creating analysis report...")
    progress_bar.progress(90)
    
    final_report = enhanced_aggregate_marker_responses(marker_responses)
    
    # Save report if requested
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    results_dir = config['data']['results_directory']
    os.makedirs(results_dir, exist_ok=True)
    
    output_filename = f"{image_id}_analysis_{timestamp}_v{config['version']}.txt"
    output_path = os.path.join(results_dir, output_filename)
    
    with open(output_path, "w") as f:
        f.write(f"Iridology Analysis Report\n")
        f.write(f"Version: {config['version']}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Image: {original_image_path} (resized to {resize_width}x{resize_height})\n\n")
        f.write(final_report)
    
    # Clean up temporary file
    if resized_image_path != original_image_path:
        try:
            os.remove(resized_image_path)
        except Exception as e:
            st.warning(f"Warning: Could not delete temporary file - {e}")
    
    progress_bar.progress(100)
    
    return marker_responses, final_report, output_path

# Parse the report for visualization
def parse_report_for_visualization(marker_responses):
    """Parse marker responses into a format for visualization"""
    # Organize by body system
    body_system_findings = {}
    
    for resp in marker_responses:
        body_system = resp.get("body_system", "Other")
        
        if body_system not in body_system_findings:
            body_system_findings[body_system] = {
                "definitive": [],
                "possible": [],
                "normal": [],
                "uncertain": []
            }
        
        # Determine category
        category = "uncertain"
        if resp.get("is_generic", False):
            category = "uncertain"
        elif resp.get("is_positive", False):
            if "yes" in resp.get("response", "").lower():
                category = "definitive"
            else:
                category = "possible"
        elif "not observed" in resp.get("response", "").lower():
            category = "normal"
            
        # Add to appropriate category
        body_system_findings[body_system][category].append(resp)
    
    return body_system_findings

# Function to create a downloadable link
def get_download_link(text, filename, link_text):
    """Generate a download link for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# App header and title
def display_header():
    """Display the app header"""
    st.title("🔍 Iridology Analysis System")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        This application uses LLaVA-1.6-Vicuna-13B with a hierarchical analysis approach 
        to detect health markers in iris images. The hierarchical approach organizes markers 
        by body system for more efficient analysis and better organized results.
        """)
    
    with col2:
        config = load_config()
        st.markdown(f"**Version:** {config['version']}")
        st.markdown(f"**Model:** {config['model']['name']}")

# Main application
def main():
    # Load configuration
    config = load_config()
    
    # Display header
    display_header()
    
    # Sidebar - Configuration
    st.sidebar.title("Settings")
    
    # Analysis mode selection
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Single Image", "Compare Images"]
    )
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    
    # Get available body systems
    all_body_systems = get_body_systems(config)
    
    # Body system selection
    selected_systems = st.sidebar.multiselect(
        "Body Systems to Analyze",
        ["All"] + all_body_systems,
        default=["All"]
    )
    
    # If "All" is selected, ignore other selections
    if "All" in selected_systems and len(selected_systems) > 1:
        selected_systems = ["All"]
    
    # Priority threshold
    priority_threshold = st.sidebar.slider(
        "Priority Threshold",
        min_value=1,
        max_value=3,
        value=config['analysis']['priority_threshold'],
        help="Maximum priority level to include (1=highest priority only)"
    )
    
    # Early stopping threshold
    early_stop_threshold = st.sidebar.slider(
        "Early Stop Threshold",
        min_value=1,
        max_value=5,
        value=config['analysis']['early_stop_threshold'],
        help="Number of positive findings needed to stop checking a body system"
    )
    
    # Maximum markers per system
    max_markers_per_system = st.sidebar.slider(
        "Max Markers per System",
        min_value=1,
        max_value=10,
        value=config['analysis']['max_markers_per_system'],
        help="Maximum number of markers to check per body system"
    )
    
    # Update config with user settings
    config['analysis']['priority_threshold'] = priority_threshold
    config['analysis']['early_stop_threshold'] = early_stop_threshold
    config['analysis']['max_markers_per_system'] = max_markers_per_system
    
    # Single Image Analysis
    if analysis_mode == "Single Image":
        st.header("Single Image Analysis")
        
        # Image Selection
        image_selection_method = st.radio(
            "Select Image",
            ["Upload New Image", "Choose Existing Image"]
        )
        
        if image_selection_method == "Upload New Image":
            uploaded_file = st.file_uploader("Upload an iris image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    image_path = tmp_file.name
                
                # Display the uploaded image
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image_path, caption="Uploaded Iris Image", use_column_width=True)
                    
                with col2:
                    # Analyze button
                    if st.button("Analyze Iris Image"):
                        with st.spinner("Analyzing..."):
                            marker_responses, report, output_path = analyze_iris(image_path, config, selected_systems)
                            
                            # Store results in session state
                            st.session_state.marker_responses = marker_responses
                            st.session_state.report = report
                            st.session_state.output_path = output_path
                            st.session_state.analyzed_image_path = image_path
        
        else:  # Choose Existing Image
            # Get list of existing images
            image_files = get_image_list(config)
            
            if not image_files:
                st.warning(f"No images found in {config['data']['images_directory']} directory")
            else:
                selected_image = st.selectbox("Select an image", image_files)
                image_path = os.path.join(config['data']['images_directory'], selected_image)
                
                # Display the selected image
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image_path, caption=f"Selected Image: {selected_image}", use_column_width=True)
                    
                with col2:
                    # Analyze button
                    if st.button("Analyze Iris Image"):
                        with st.spinner("Analyzing..."):
                            marker_responses, report, output_path = analyze_iris(image_path, config, selected_systems)
                            
                            # Store results in session state
                            st.session_state.marker_responses = marker_responses
                            st.session_state.report = report
                            st.session_state.output_path = output_path
                            st.session_state.analyzed_image_path = image_path
        
        # Display results if available
        if hasattr(st.session_state, 'report') and st.session_state.report:
            st.header("Analysis Results")
            
            # Create tabs for different view types
            tab1, tab2, tab3 = st.tabs(["Body System View", "Full Report", "Download Options"])
            
            with tab1:
                # Parse the report for visualization
                body_system_findings = parse_report_for_visualization(st.session_state.marker_responses)
                
                # Create a tab for each body system
                if body_system_findings:
                    system_tabs = st.tabs(list(body_system_findings.keys()))
                    
                    for i, system in enumerate(body_system_findings.keys()):
                        with system_tabs[i]:
                            findings = body_system_findings[system]
                            
                            # Display definitive findings
                            if findings["definitive"]:
                                st.subheader("✅ Definitive Findings")
                                for finding in findings["definitive"]:
                                    with st.expander(finding["marker"].split('[')[0].strip()):
                                        if "response" in finding:
                                            st.text_area("Details", finding["response"], height=150)
                            
                            # Display possible findings
                            if findings["possible"]:
                                st.subheader("⚠️ Possible Findings")
                                for finding in findings["possible"]:
                                    with st.expander(finding["marker"].split('[')[0].strip()):
                                        if "response" in finding:
                                            st.text_area("Details", finding["response"], height=150)
                            
                            # Display if no findings
                            if not findings["definitive"] and not findings["possible"]:
                                st.info("No significant findings for this body system")
                else:
                    st.warning("No body system findings available")
            
            with tab2:
                # Display full report
                st.text_area("Full Analysis Report", st.session_state.report, height=500)
            
            with tab3:
                # Download options
                st.subheader("Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download as text
                    st.markdown(get_download_link(
                        st.session_state.report,
                        f"iridology_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "Download as Text File"
                    ), unsafe_allow_html=True)
                
                with col2:
                    # View saved file location
                    st.info(f"Report saved to: {st.session_state.output_path}")
    
    # Compare Images Analysis
    else:
        st.header("Compare Analysis Results")
        
        # Get list of existing results
        result_files = get_results_list(config)
        
        if not result_files:
            st.warning(f"No analysis results found in {config['data']['results_directory']} directory")
        else:
            # Select results to compare
            result1 = st.selectbox("Select first analysis result", result_files, index=0)
            remaining_results = [f for f in result_files if f != result1]
            
            if remaining_results:
                result2 = st.selectbox("Select second analysis result", remaining_results, index=0)
                
                # Display comparison
                if st.button("Compare Results"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"Result 1: {result1}")
                        result1_path = os.path.join(config['data']['results_directory'], result1)
                        with open(result1_path, 'r') as f:
                            result1_content = f.read()
                        st.text_area("Contents", result1_content, height=500)
                    
                    with col2:
                        st.subheader(f"Result 2: {result2}")
                        result2_path = os.path.join(config['data']['results_directory'], result2)
                        with open(result2_path, 'r') as f:
                            result2_content = f.read()
                        st.text_area("Contents", result2_content, height=500)
            else:
                st.warning("Need at least two result files to compare")
    
    # Footer
    st.markdown("---")
    st.caption(f"Iridology Analysis System v{config['version']} | Hierarchical Approach")

if __name__ == "__main__":
    main() 