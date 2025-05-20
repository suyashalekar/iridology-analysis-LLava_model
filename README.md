# Iridology Analysis System with LLaVA-1.6

A machine learning based iridology analysis system that uses LLaVA-1.6-Vicuna-13B with MLX for Mac to analyze iris images and detect health markers.

## Key Features

- **Hierarchical Analysis**: Organizes markers by body system for more efficient use of context window
- **Priority-Based Evaluation**: Analyzes high-priority markers first, stopping early when sufficient evidence is found
- **Body System Classification**: Groups findings by major body systems (Digestive, Nervous, Circulatory, etc.)
- **Enhanced Reporting**: Provides system-specific recommendations based on findings
- **Streamlit Interface**: User-friendly web interface for analyzing and comparing iris images

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
   - View results organized by body system
   - Download analysis reports

2. **Result Comparison**:
   - Compare two previous analysis results side by side
   - Track changes between analyses

### Running the Interface:

```bash
cd src
streamlit run streamlit_app.py
```

The interface will open in your default web browser at http://localhost:8501.

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
streamlit run src/streamlit_app.py
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
- `<image_id>_analysis_<timestamp>_v<version>.txt`

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