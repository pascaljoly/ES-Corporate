# Configuration System Guide

This guide explains how to use the new configuration system that replaces hardcoded parameters throughout the Energy Score Tool.

## Overview

The configuration system centralizes all configurable parameters in a single YAML file (`config.yaml`), making it easy to:
- Update model performance profiles without changing code
- Adjust scoring weights and thresholds
- Configure hardware and task support
- Customize output formats and logging
- Override settings via environment variables

## Configuration File Structure

The `config.yaml` file is organized into logical sections:

### 1. Scoring System Configuration
```yaml
scoring:
  star_rating:
    min_stars: 1.0
    max_stars: 5.0
    default_score: 3.0
  
  metric_weights:
    energy_efficiency: 0.4
    co2_efficiency: 0.3
    performance: 0.2
    speed: 0.1
  
  validation:
    max_coefficient_of_variation: 0.15
    default_coefficient_of_variation: 0.03
    variation_range: 0.05
```

### 2. Model Profiles
```yaml
model_profiles:
  defaults:
    energy_per_1k_wh: 5.0
    co2_per_1k_g: 2.5
    samples_per_second: 50
    duration_seconds: 20.0
    size_mb: 100
    architecture: "Unknown"
  
  text_generation:
    "distilgpt2":
      energy_per_1k_wh: 2.1
      co2_per_1k_g: 1.0
      samples_per_second: 120
      duration_seconds: 8.3
      size_mb: 82
      architecture: "GPT-2"
```

### 3. Hardware and Task Support
```yaml
hardware:
  supported_types:
    "T4": "NVIDIA Tesla T4 16GB"
    "V100": "NVIDIA Tesla V100 32GB"
    "CPU": "CPU-only (no GPU)"

tasks:
  supported:
    - "text-classification"
    - "image-classification"
    - "text-generation"
```

### 4. CodeCarbon Settings
```yaml
codecarbon:
  pue: 1.2
  measure_power_secs: 1
  project_name_pattern: "{model_path}_{task}"
```

### 5. Measurement Configuration
```yaml
measurement:
  default_samples: 100
  default_runs: 3
  progress_interval_percent: 10
  hardware_detection_timeout: 5
```

## Using the Configuration System

### Basic Usage

```python
from config_loader import get_config

# Load configuration
config = get_config()

# Access configuration values
metric_weights = config.get_metric_weights()
supported_hardware = config.get_supported_hardware()
model_profile = config.get_model_profile("distilgpt2", "text_generation")

# Get nested values using dot notation
max_stars = config.get("scoring.star_rating.max_stars", 5.0)
```

### Using Configuration-Aware Scorer

```python
from energy_compare.scorer.config_aware_scorer import ConfigAwareModelScorer

# Create scorer that uses configuration profiles
scorer = ConfigAwareModelScorer()

# Score a model using profiles from config.yaml
result = scorer.score(
    model="distilgpt2",
    task="text_generation",
    n_samples=100,
    runs=3
)
```

### Using Configuration-Aware Comparator

```python
from energy_compare.comparator import ModelComparator
from energy_compare.scorer.config_aware_scorer import ConfigAwareModelScorer

# Create comparator with configuration-aware scorer
scorer = ConfigAwareModelScorer()
comparator = ModelComparator(scorer=scorer)

# Compare models using configuration profiles
result = comparator.compare_models([
    ("distilgpt2", "text_generation"),
    ("gpt2", "text_generation"),
    ("gpt2-medium", "text_generation")
])
```

## Environment Variable Overrides

You can override configuration values using environment variables:

```bash
# Override metric weights
export ENERGY_SCORE_SCORING_METRIC_WEIGHTS_ENERGY_EFFICIENCY=0.6
export ENERGY_SCORE_SCORING_METRIC_WEIGHTS_CO2_EFFICIENCY=0.4

# Override star rating range
export ENERGY_SCORE_SCORING_STAR_RATING_MAX_STARS=10.0

# Override CodeCarbon settings
export ENERGY_SCORE_CODECARBON_PUE=1.5
```

Environment variables use the format: `ENERGY_SCORE_{SECTION}_{SUBSECTION}_{KEY}`

## Adding New Model Profiles

### Method 1: Edit config.yaml directly

Add new models to the appropriate section in `config.yaml`:

```yaml
model_profiles:
  text_generation:
    "my-custom-model":
      energy_per_1k_wh: 1.8
      co2_per_1k_g: 0.9
      samples_per_second: 150
      duration_seconds: 6.7
      size_mb: 67
      architecture: "Custom Architecture"
```

### Method 2: Add programmatically

```python
from energy_compare.scorer.config_aware_scorer import ConfigAwareModelScorer

scorer = ConfigAwareModelScorer()

new_profile = {
    "energy_per_1k_wh": 1.8,
    "co2_per_1k_g": 0.9,
    "samples_per_second": 150,
    "duration_seconds": 6.7,
    "size_mb": 67,
    "architecture": "Custom Architecture"
}

# Add profile (in-memory only)
scorer.add_model_profile("my-custom-model", new_profile, "text_generation")

# Validate the profile
is_valid, error = scorer.validate_model_profile(new_profile)
```

## Configuration Validation

The system automatically validates configuration:

```python
from config_loader import get_config

config = get_config()

try:
    config.validate_config()
    print("Configuration is valid")
except ConfigError as e:
    print(f"Configuration error: {e}")
```

Validation checks:
- Metric weights sum to 1.0
- Star rating min < max
- Required sections are present
- Model profiles have required fields

## Migration from Hardcoded Values

### Before (Hardcoded)
```python
# Old hardcoded approach
model_profiles = {
    "distilgpt2": {
        "energy_per_1k_wh": 2.1,
        "co2_per_1k_g": 1.0,
        "samples_per_second": 120,
        # ... more hardcoded values
    }
}

metric_weights = {
    ComparisonMetric.ENERGY_EFFICIENCY: 0.4,
    ComparisonMetric.CO2_EFFICIENCY: 0.3,
    # ... more hardcoded weights
}
```

### After (Configuration-based)
```python
# New configuration-based approach
from config_loader import get_config

config = get_config()
model_profiles = config.get_model_profile("distilgpt2", "text_generation")
metric_weights = config.get_metric_weights()
```

## Best Practices

1. **Keep config.yaml in version control** - This ensures all team members use the same configuration
2. **Use environment variables for deployment-specific overrides** - Don't modify config.yaml for different environments
3. **Validate configuration** - Always call `config.validate_config()` after loading
4. **Document custom model profiles** - Add comments in config.yaml explaining model characteristics
5. **Use meaningful model IDs** - Choose descriptive names that match actual model identifiers

## Troubleshooting

### Common Issues

1. **Configuration file not found**
   - Ensure `config.yaml` exists in the project root
   - Check file permissions

2. **Invalid YAML syntax**
   - Use a YAML validator to check syntax
   - Ensure proper indentation (spaces, not tabs)

3. **Metric weights don't sum to 1.0**
   - Check that all weights are positive numbers
   - Ensure the sum equals 1.0 (within floating-point precision)

4. **Model profile not found**
   - Check model ID spelling
   - Verify the model is in the correct task category
   - Check that the task category exists in config.yaml

### Debug Mode

Enable debug logging to see configuration loading details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from config_loader import get_config
config = get_config()  # Will show detailed loading information
```

## Example Usage

See `tests/configuration/example_config_usage.py` for a complete demonstration of the configuration system features.

## Testing

Run the comprehensive test suite to verify the configuration system:

```bash
cd /path/to/EStool
source energy-compare/venv/bin/activate
python tests/configuration/test_configuration_system.py
```

## API Reference

### ConfigLoader Class

- `get(key, default=None)` - Get configuration value using dot notation
- `get_model_profile(model_id, task_type=None)` - Get model performance profile
- `get_metric_weights()` - Get scoring metric weights
- `get_supported_hardware()` - Get supported hardware types
- `get_supported_tasks()` - Get supported task types
- `validate_config()` - Validate configuration
- `reload()` - Reload configuration from file

### ConfigAwareModelScorer Class

- `score(model, task, n_samples=100, runs=3)` - Score model using configuration profiles
- `get_available_models(task_category=None)` - Get all available model profiles
- `add_model_profile(model_id, profile, task_category=None)` - Add new model profile
- `validate_model_profile(profile)` - Validate model profile structure
