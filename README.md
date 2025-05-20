# Iridology Analysis System with LLaVA-1.6

A machine learning based iridology analysis system that uses LLaVA-1.6-Vicuna-13B with MLX for Mac to analyze iris images and detect health markers.

## Key Features

- **Hierarchical Analysis**: Organizes markers by body system for more efficient use of context window
- **Priority-Based Evaluation**: Analyzes high-priority markers first, stopping early when sufficient evidence is found
- **Body System Classification**: Groups findings by major body systems (Digestive, Nervous, Circulatory, etc.)
- **Enhanced Reporting**: Provides system-specific recommendations based on findings
- **Streamlit Interface**: User-friendly web interface for analyzing iris images
- **Batch Processing**: Process all images in a directory with time tracking
- **Performance Metrics**: Detailed timing information for each step of the analysis process

## Streamlit Interface

![Streamlit Interface Screenshot](docs/images/streamlit_interface.png)

The Streamlit interface provides an intuitive way to interact with the iridology analysis system, allowing users to:

- Upload iris images or select from existing ones
- Configure analysis parameters
- View results organized by body system
- Track processing time for each analysis step

For detailed instructions, see [Streamlit Interface Documentation](docs/streamlit_interface.md).

## Project Structure

```
.
├── config/             # Configuration files
├── docs/               # Documentation
├── images/             # Test iris images
├── results/            # Analysis results
├── src/                # Source code
│   ├── core/           # Core analysis functionality
│   ├── models/         # Model interfaces
│   ├── utils/          # Utility scripts
│   └── streamlit_app.py # Web interface
└── tests/              # Test scripts
```

## Using the Streamlit Interface

The system includes a user-friendly Streamlit web interface for analyzing iris images and viewing results.

### Features:

1. **Single Image Analysis**:
   - Upload new images or select from existing images
   - Configure analysis parameters (body systems, priority thresholds)
   - View results with detailed timing information
   - Download analysis reports

2. **Result Viewer**:
   - View any previously generated analysis results
   - Access detailed diagnostics and timing information

### Running the Interface:

```bash
./run_streamlit_app.sh
```

The interface will open in your default web browser at http://localhost:8501.

## Batch Processing

To process all images in the images directory in a single batch:

```bash
./run_batch_processing.sh
```

This will:
- Process all images in the `images/` directory
- Save detailed analysis results to the `results/` directory
- Display progress with estimated completion time
- Provide detailed timing information for each image

### Batch Processing Features:

- **Progress Tracking**: Shows progress through all images with estimated completion time
- **Detailed Timing**: Records processing time for each image and each analysis step
- **Fault Tolerance**: Continues processing if an individual image analysis fails
- **Summary Report**: Provides total processing time and average time per image

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure LLaVA-1.6-Vicuna-13B is running (via MLX or other backend)
4. Update the configuration in `config/config.json` to point to your model

## Usage

For command-line usage:

```bash
python src/iris_analyze.py images/example.jpg
```

For web interface:

```bash
./run_streamlit_app.sh
```

For batch processing:

```bash
./run_batch_processing.sh
```

## Configuration

Edit `config/config.json` to configure:
- Model parameters
- Analysis settings
- File paths

## Documentation

See the `docs/` directory for detailed documentation on:
- Hierarchical approach (`hierarchical_approach.md`)
- Analysis methods
- System design

## Development

### Version Control

This project uses Git for version control. Major versions are tagged (e.g., v1.0.0).

- `main` branch: Stable, production-ready code
- Feature branches: Development of new features

### Configuration

Configuration values are stored in `config/config.json`:

- Model parameters
- Analysis settings
- Image processing settings

### Results

Analysis results are saved in the `results` directory with a timestamp and version number:
- `iris_<image_id>_analysis_<timestamp>_v<version>.txt`

Each result file includes:
- Detailed timing information for each processing step
- Debug information for troubleshooting
- Complete analysis report

## How It Works

The hierarchical approach is more efficient than linear analysis:

1. **Body System Organization**: Markers are organized by body system
2. **Priority-Based Analysis**: High-priority markers are checked first
3. **Early Stopping**: Analysis stops when sufficient evidence is found for a body system
4. **Adaptive Focus**: Focuses analysis on body systems with most positive findings

## License

MIT License - See LICENSE file for details

## Credits

Developed for Datascience project - Semester 4 