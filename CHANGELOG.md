# Changelog

All notable changes to the Iridology Analysis System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2023-05-23

### Added
- Hierarchical analysis structure to improve context window utilization
- Body system categorization for markers (Digestive, Nervous, Circulatory, etc.)
- Priority-based marker evaluation to check high-priority markers first
- Early stopping mechanism to avoid wasting tokens on less relevant markers
- Enhanced reporting with body system-specific recommendations
- Configuration system with external config.json file
- Version control with Git repository structure
- Improved documentation including README and CHANGELOG

### Changed
- Restructured codebase with organized directory structure
- Refactored marker analysis to process by body system instead of linearly
- Modified result format to organize findings by body system and confidence level

### Fixed
- Context window limitations by using hierarchical approach
- Issue with false or uncertain responses by focusing on high-priority markers first
- Problem with unstructured recommendations by organizing them by body system

## [0.2.0] - 2023-05-15

### Added
- Support for LLaVA-1.6-Vicuna-13B model
- Image preprocessing to reduce token consumption
- Enhanced marker response aggregation
- Batch processing capability for multiple images
- More detailed visual analysis in reports

### Changed
- Upgraded from previous LLM to LLaVA for better visual analysis
- Improved prompt templates for more specific analysis
- Better handling of uncertain or generic responses

### Fixed
- Various issues with response parsing and formatting
- Image preprocessing to ensure consistent results

## [0.1.0] - 2023-05-01

### Added
- Initial implementation of iridology analysis system
- Basic marker detection capability
- Simple reporting format
- Support for analyzing single iris images
- Integration with LM Studio API 