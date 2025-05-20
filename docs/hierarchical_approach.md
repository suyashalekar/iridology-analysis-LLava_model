# Hierarchical Approach to Iridology Analysis

## Introduction

The hierarchical approach to iridology analysis is designed to address key challenges in automated iris analysis:

1. **Context window limitations**: LLM models have a fixed context window size that limits the amount of analysis that can be performed at once
2. **Focus on relevant markers**: Not all markers are equally important or likely to be present
3. **Organization of findings**: Results need to be organized logically by body system for better understanding

## How It Works

The hierarchical approach operates through these key mechanisms:

### 1. Body System Organization

Markers are grouped by body systems:
- Digestive System
- Nervous System
- Circulatory System
- Lymphatic System
- Respiratory System
- Endocrine System
- Structural System
- Urinary System

This enables more focused analysis and better organized results.

### 2. Priority-Based Analysis

Markers are assigned priority levels (1-3):
- Priority 1: Highest importance or most commonly found markers
- Priority 2: Medium importance markers
- Priority 3: Less common or less critical markers

The system first analyzes high-priority markers before moving to lower-priority ones.

### 3. Early Stopping Mechanism

When sufficient evidence is found for a body system (configurable threshold of positive markers), analysis stops for that system to save computational resources.

```python
# Example of early stopping logic
if system_positive_count >= early_stop_threshold:
    print(f"Found {system_positive_count} positive markers in {system} system. "
          f"Stopping analysis for this system.")
    break
```

### 4. Adaptive Focus

Analysis time is concentrated on body systems showing the most evidence of issues, rather than spending equal time on all systems.

## Implementation Details

### Core Components

1. **`load_hierarchical_data()`**: Loads markers from Excel/CSV and organizes them by body system and priority
2. **`analyze_iris_image_hierarchical()`**: Performs analysis in hierarchical order
3. **`enhanced_aggregate_marker_responses()`**: Aggregates findings with body system organization

### Configuration

The hierarchical approach is configurable through several parameters:

- `max_markers_per_system`: Maximum number of markers to check per body system
- `early_stop_threshold`: Number of positive findings needed to stop checking a body system
- `priority_threshold`: Maximum priority level to include (1=highest priority only, 2=high+medium, etc.)

## Benefits

The hierarchical approach offers significant advantages:

1. **More efficient token usage**: By focusing on high-priority markers and stopping early, the system uses tokens more efficiently
2. **Better organized results**: Findings are logically grouped by body system
3. **More targeted recommendations**: Recommendations are tailored to the affected body systems
4. **Adaptability**: The system can adapt its focus based on what it finds
5. **Improved accuracy**: By focusing on the most relevant markers first, the system produces more accurate overall analysis

## Future Enhancements

Potential improvements to the hierarchical approach:

1. **Dynamic priority adjustment**: Adjust marker priorities based on findings in related body systems
2. **Confidence weighting**: Weight findings by confidence level when determining body system scores
3. **Cross-system pattern detection**: Identify patterns that span multiple body systems
4. **Personalized analysis paths**: Adapt analysis based on patient history or previous findings 