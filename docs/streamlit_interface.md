# Streamlit Interface for Iridology Analysis

The Streamlit interface provides a user-friendly way to interact with the hierarchical iridology analysis system. This guide explains how to use the interface and its features.

## Getting Started

### Running the Interface

You can start the Streamlit interface in two ways:

1. Using the helper script:
   ```bash
   ./run_streamlit_app.sh
   ```

2. Directly with Streamlit:
   ```bash
   cd src
   streamlit run streamlit_app.py
   ```

The interface will automatically open in your default web browser at http://localhost:8501.

## Interface Features

### Main Components

The interface is divided into several key areas:

1. **Sidebar**: Contains all configuration and analysis settings
2. **Main Area**: Displays the image selection, analysis results, and comparison tools
3. **Header**: Shows the application title and version information

### Analysis Modes

The application supports two primary modes:

1. **Single Image Analysis**: Analyze a single iris image
2. **Compare Images**: Compare the results from two previous analyses

### Configuration Options

In the sidebar, you can configure the following settings:

- **Body Systems to Analyze**: Select specific body systems or choose "All"
- **Priority Threshold**: Set the maximum priority level of markers to include (1=highest priority only)
- **Early Stop Threshold**: Number of positive findings needed to stop checking a body system
- **Max Markers per System**: Maximum number of markers to check per body system

### Single Image Analysis

In single image analysis mode, you can:

1. **Select an Image**:
   - Upload a new image from your computer
   - Choose an existing image from the images directory

2. **Analyze the Image**:
   - Click the "Analyze Iris Image" button to start the analysis
   - Watch the progress bar as the analysis runs

3. **View Results**:
   - **Body System View**: Results organized by body system with expandable findings
   - **Full Report**: Complete text report of all findings
   - **Download Options**: Save the analysis as a text file

### Results Organization

The analysis results are organized in the following way:

- **Definitive Findings**: Strong evidence of markers (✅)
- **Possible Findings**: Potential evidence, but not conclusive (⚠️)
- **System Recommendations**: Specific advice for each body system

### Comparing Results

In compare mode, you can:

1. Select two previous analysis results from the dropdown menus
2. Click "Compare Results" to view them side by side
3. Identify differences in findings between analyses

## Tips for Best Results

1. **Image Quality**: Use clear, well-lit images of the iris
2. **Focus on Specific Systems**: If you're interested in particular health aspects, select only those body systems
3. **Save Important Results**: Download and save important analyses for later reference
4. **Adjusting Parameters**: 
   - Lower the priority threshold for more comprehensive analysis
   - Increase the early stop threshold for more thorough checking of each system
   - Adjust the max markers per system based on your available time/resources

## Troubleshooting

If you encounter issues:

1. **Images not showing**: Ensure the image is in the correct format (JPG, JPEG, PNG)
2. **Analysis taking too long**: Reduce the number of body systems or increase the priority threshold
3. **Results not appearing**: Check the console for any error messages
4. **Connection errors**: Ensure the LLaVA model is running properly at the configured endpoint 