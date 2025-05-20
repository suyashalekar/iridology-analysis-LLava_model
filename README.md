# Iridology Analysis System with LLaVA-1.6

A machine learning based iridology analysis system that uses LLaVA-1.6-Vicuna-13B with MLX for Mac to analyze iris images and detect health markers.

## Key Features

- **Hierarchical Analysis**: Organizes markers by body system for more efficient use of context window
- **Priority-Based Evaluation**: Analyzes high-priority markers first, stopping early when sufficient evidence is found
- **Body System Classification**: Groups findings by major body systems (Digestive, Nervous, Circulatory, etc.)
- **Enhanced Reporting**: Provides system-specific recommendations based on findings

## Project Structure

```
.
├── config/             # Configuration files
├── docs/               # Documentation
├── images/             # Test iris images
├── results/            # Analysis results
├── src/                # Source code
│   ├── core/           # Core analysis functions
│   ├── utils/          # Utility functions
│   └── models/         # Model interface code
└── tests/              # Test code
```

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure LLaVA-1.6-Vicuna-13B is running via LM Studio on port 1234

## Usage

### Hierarchical Analysis

Run the hierarchical analysis on an iris image:

```bash
python src/iris_analyze_hierarchical.py images/16.jpg
```

### Batch Processing

Process multiple images:

```bash
python src/batch_analyze_iris.py
```

### Web Interface

Run the Streamlit web interface:

```bash
sh run_streamlit_app.sh
```

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