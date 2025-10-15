# Energy Compare Framework

A comprehensive framework for comparing machine learning models based on their energy consumption, CO2 emissions, and performance metrics.

## 🎯 Overview

The Energy Compare framework provides tools to:
- **Score individual models** using the `ModelEnergyScorer` (with configuration-based profiles)
- **Compare multiple models** using the `ModelComparator`
- **Analyze energy efficiency** across different metrics
- **Generate detailed reports** with rankings and statistics

> **Note**: The current implementation uses configuration-based model profiles for scoring. For real energy measurement, use the `ml_energy_score.measure.measure_model_energy()` function which accepts actual datasets and performs live energy measurement.

## 🏗️ Architecture

### Core Components

1. **Scorer** (`scorer/`) - Measures energy consumption of individual models
2. **Comparator** (`comparator/`) - Compares multiple models and generates rankings
3. **Tests** (`tests/`) - Comprehensive test suite for validation

### Key Classes

- `ModelEnergyScorer` - Measures energy consumption and CO2 emissions
- `ModelComparator` - Compares multiple models based on various metrics
- `ComparisonResult` - Contains comparison results with rankings and statistics
- `ComparisonMetric` - Enum of available comparison metrics

## 🎯 Supported Model Types

The framework supports a wide range of ML model types:

### Text Models
- **Text Generation**: GPT-2, DistilGPT-2, GPT-2 Medium
- **Text Classification**: BERT, DistilBERT, RoBERTa
- **Sentiment Analysis**: Various transformer models

### Computer Vision Models
- **Object Detection**: YOLOv8 (nano, small, medium), BiRefNet
- **Image Classification**: ResNet-50, EfficientNet, Vision Transformer
- **Salient Object Detection**: U2Net, PoolNet

### Supported Tasks
- `text-generation` - Text generation models
- `text-classification` - Text classification models  
- `image-classification` - Image classification models
- `object_detection` - Object detection models
- `sentiment-analysis` - Sentiment analysis models
- `question-answering` - Question answering models
- `summarization` - Text summarization models
- `translation` - Machine translation models

## 🔧 Scoring Approaches

The framework supports two approaches for model scoring:

### 1. Configuration-Based Scoring (Default)
- Uses predefined model profiles from `config.yaml`
- Fast and consistent for comparison purposes
- No actual energy measurement required
- Perfect for model selection and comparison workflows

### 2. Live Energy Measurement
- Uses `ml_energy_score.measure.measure_model_energy()` function
- Performs actual energy measurement with CodeCarbon
- Requires real datasets and model execution
- Suitable for production energy monitoring

```python
# Configuration-based scoring (fast, for comparison)
from scorer.config_aware_scorer import ConfigAwareModelScorer
scorer = ConfigAwareModelScorer()
result = scorer.score("distilgpt2", "text-generation")

# Live energy measurement (real measurement)
from ml_energy_score.measure import measure_model_energy
result = measure_model_energy(
    model_path="distilgpt2",
    task="text-generation", 
    dataset=your_dataset,  # Real dataset required
    hardware="CPU"
)
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For development
pip install -e .
```

### Basic Usage

```python
from comparator import ModelComparator, ComparisonMetric
from scorer.core import ModelEnergyScorer

# Initialize the comparator
comparator = ModelComparator()

# Define models to compare
model_specs = [
    ("gpt2", "text-generation"),
    ("gpt2-medium", "text-generation"),
    ("distilgpt2", "text-generation")
]

# Compare models
result = comparator.compare_models(
    model_specs=model_specs,
    n_samples=100,
    runs=3
)

# Get results
winner = result.get_winner()
print(f"Winner: {winner.model_id}")
print(f"Score: {winner.score:.3f}")

# View rankings
for model in result.get_rankings():
    print(f"Rank {model.rank}: {model.model_id} (Score: {model.score:.3f})")
```

### Computer Vision Example

```python
# Compare computer vision models
cv_model_specs = [
    ("YOLOv8n", "object_detection"),
    ("YOLOv8s", "object_detection"),
    ("BiRefNet", "object_detection")
]

cv_result = comparator.compare_models(
    model_specs=cv_model_specs,
    n_samples=50,
    runs=2
)

print(f"CV Winner: {cv_result.get_winner().model_id}")
```

## 📊 Comparison Metrics

The framework supports multiple comparison metrics:

| Metric | Description | Better When |
|--------|-------------|-------------|
| `ENERGY_EFFICIENCY` | Energy consumption per 1000 queries | Lower |
| `CO2_EFFICIENCY` | CO2 emissions per 1000 queries | Lower |
| `PERFORMANCE` | Throughput (samples per second) | Higher |
| `SPEED` | Inverse of duration | Higher |
| `COST_EFFECTIVENESS` | Performance per unit energy | Higher |

### Custom Weights

You can customize the importance of each metric:

```python
custom_weights = {
    ComparisonMetric.ENERGY_EFFICIENCY: 0.5,  # 50% weight
    ComparisonMetric.CO2_EFFICIENCY: 0.3,     # 30% weight
    ComparisonMetric.PERFORMANCE: 0.2         # 20% weight
}

result = comparator.compare_models(
    model_specs=model_specs,
    custom_weights=custom_weights
)
```

## 🔧 Advanced Usage

### Using Pre-computed Results

If you already have scoring results, you can compare them directly:

```python
from scorer.core import ScoringResult

# Create scoring results
results = [
    ScoringResult(
        model_id="model1",
        task="text_generation",
        measurements={
            'energy_per_1k_wh': 4.0,
            'co2_per_1k_g': 2.0,
            'samples_per_second': 100,
            'duration_seconds': 10.0
        },
        hardware={'gpu': 'RTX 4090'},
        metadata={'timestamp': '2025-10-03'}
    ),
    # ... more results
]

# Compare pre-computed results
result = comparator.compare_models_from_results(results)
```

### Specific Metrics Only

Compare models on specific metrics only:

```python
# Compare only on energy efficiency
energy_only = [ComparisonMetric.ENERGY_EFFICIENCY]
result = comparator.compare_models(
    model_specs=model_specs,
    metrics=energy_only
)
```

### Save and Load Results

```python
# Save comparison results
comparator.save_comparison(result, "comparison_results.json")

# Load comparison results
loaded_result = comparator.load_comparison("comparison_results.json")
```

## 📈 Detailed Analysis

The framework provides comprehensive analysis capabilities:

```python
result = comparator.compare_models(model_specs)

# Summary statistics
summary = result.summary
print(f"Total models: {summary['total_models']}")
print(f"Winner: {summary['winner']}")
print(f"Score range: {summary['score_statistics']['min']:.3f} - {summary['score_statistics']['max']:.3f}")

# Energy and CO2 ranges
energy_range = summary['energy_range']
co2_range = summary['co2_range']
print(f"Energy range: {energy_range['min_kwh']:.1f} - {energy_range['max_kwh']:.1f} kWh/1k queries")
print(f"CO2 range: {co2_range['min_kg']:.1f} - {co2_range['max_kg']:.1f} kg CO2/1k queries")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_comparator.py -v
python -m pytest tests/test_scorer.py -v

# Run with coverage
python -m pytest tests/ --cov=comparator --cov=scorer
```

### Test Coverage

The test suite covers:
- ✅ Input validation and error handling
- ✅ Model comparison algorithms
- ✅ Metric score calculations
- ✅ Ranking and scoring logic
- ✅ Save/load functionality
- ✅ Edge cases and error conditions
- ✅ Integration with scorer

## 📋 Example Scenarios

### Scenario 1: Energy-Focused Comparison

```python
# Prioritize energy efficiency
energy_weights = {
    ComparisonMetric.ENERGY_EFFICIENCY: 0.7,
    ComparisonMetric.CO2_EFFICIENCY: 0.3
}

result = comparator.compare_models(
    model_specs=model_specs,
    custom_weights=energy_weights
)
```

### Scenario 2: Performance-Focused Comparison

```python
# Prioritize performance and speed
performance_weights = {
    ComparisonMetric.PERFORMANCE: 0.4,
    ComparisonMetric.SPEED: 0.4,
    ComparisonMetric.ENERGY_EFFICIENCY: 0.2
}

result = comparator.compare_models(
    model_specs=model_specs,
    custom_weights=performance_weights
)
```

### Scenario 3: Balanced Comparison

```python
# Use default balanced weights
result = comparator.compare_models(model_specs)
```

## 🔍 Understanding Results

### Scoring System

- **Composite Score**: Weighted combination of all metrics (0.0 - 1.0)
- **Rankings**: Models ranked by composite score (1 = best)
- **Individual Metrics**: Normalized scores for each metric

### Result Structure

```python
{
    "task": "text_generation",
    "models": [
        {
            "model_id": "distilgpt2",
            "rank": 1,
            "score": 0.95,
            "measurements": {
                "energy_per_1k_wh": 2.8,
                "co2_per_1k_g": 1.4,
                "samples_per_second": 95
            }
        }
    ],
    "summary": {
        "total_models": 3,
        "winner": "distilgpt2",
        "score_statistics": {...},
        "energy_range": {...},
        "co2_range": {...}
    }
}
```

## 🛠️ Development

### Project Structure

```
energy-compare/
├── comparator/
│   ├── __init__.py
│   └── core.py          # Main comparison logic
├── scorer/
│   ├── __init__.py
│   └── core.py          # Energy scoring logic
├── tests/
│   ├── conftest.py      # Test fixtures
│   ├── test_comparator.py
│   └── test_scorer.py
├── example_usage.py     # Usage examples
├── requirements.txt     # Dependencies
└── README.md           # This file
```

### Adding New Metrics

To add a new comparison metric:

1. Add to `ComparisonMetric` enum in `comparator/core.py`
2. Implement calculation in `_calculate_metric_score()`
3. Add to default weights in `ModelComparator.__init__()`
4. Add tests in `test_comparator.py`

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## 📚 API Reference

### ModelComparator

```python
class ModelComparator:
    def __init__(self, scorer: Optional[ModelEnergyScorer] = None)
    
    def compare_models(
        self,
        model_specs: List[Tuple[str, str]],
        metrics: Optional[List[ComparisonMetric]] = None,
        n_samples: int = 100,
        runs: int = 3,
        custom_weights: Optional[Dict[ComparisonMetric, float]] = None
    ) -> ComparisonResult
    
    def compare_models_from_results(
        self,
        scoring_results: List[ScoringResult],
        metrics: Optional[List[ComparisonMetric]] = None,
        custom_weights: Optional[Dict[ComparisonMetric, float]] = None
    ) -> ComparisonResult
    
    def save_comparison(self, result: ComparisonResult, output_path: Union[str, Path])
    def load_comparison(self, input_path: Union[str, Path]) -> ComparisonResult
```

### ComparisonResult

```python
class ComparisonResult:
    def get_winner(self) -> Optional[ModelComparison]
    def get_rankings(self) -> List[ModelComparison]
    def to_dict(self) -> Dict
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors**: Make sure you're in the correct directory and have installed dependencies
```bash
cd energy-compare
pip install -r requirements.txt
```

**Validation Errors**: Check that model specifications are valid and weights sum to 1.0

**Empty Results**: Ensure scoring results contain required measurement fields

### Getting Help

- Check the test files for usage examples
- Run `python example_usage.py` for comprehensive examples
- Review the API reference above

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of the ML Energy Score framework
- Uses CodeCarbon for energy measurement
- Integrates with HuggingFace model ecosystem

