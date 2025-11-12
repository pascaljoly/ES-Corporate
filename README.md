# ML Energy Score

A simple and efficient Python toolkit for measuring and scoring ML model energy consumption using CodeCarbon.

## Overview

ML Energy Score provides:
- Energy consumption measurement for any ML model
- Star-based efficiency scoring to compare models
- Support for multiple hardware platforms (CPU, GPU, Apple Silicon)
- Integration with CodeCarbon for accurate energy tracking
- JSON-formatted results for easy analysis

## Motivation

As AI models grow in size and complexity, their energy consumption and environmental impact increase significantly. ML Energy Score helps organizations make informed decisions about model selection by quantifying energy efficiency, enabling:

- **Sustainable AI Development**: Choose models that minimize environmental impact
- **Cost Optimization**: Reduce energy costs in production deployments
- **Transparency**: Provide clear metrics for AI sustainability reporting
- **Informed Decision-Making**: Balance model performance with energy efficiency

This tool empowers AI engineers and sustainability professionals to build more sustainable AI systems without sacrificing functionality.

## 🚀 Quick Start

```bash
# Install from GitHub
pip install git+https://github.com/pascaljoly/ml-energy-score.git

# Or install dependencies locally
pip install -r requirements.txt

# Measure energy consumption of a model
python energy_measurement/measure_energy.py

# Test the measurement function
python energy_measurement/test/test_dummy.py
python energy_measurement/test/test_pytorch.py

# Calculate energy efficiency scores
python energy_score/calculate_scores.py
```

## 📁 Project Structure

```
ml-energy-score/
├── utils/                       # Shared utilities
│   ├── __init__.py
│   └── security_utils.py        # Security and validation utilities
│
├── energy_measurement/          # Energy measurement package
│   ├── __init__.py
│   ├── measure_energy.py        # Main measurement function
│   ├── example_usage.py         # Usage examples
│   ├── test/                    # Measurement tests
│   │   ├── __init__.py
│   │   ├── test_dummy.py        # Test with dummy model
│   │   ├── test_pytorch.py      # Test with real PyTorch model
│   │   └── sample_dataset.py    # Dataset sampling utilities
│   └── README.md                # Detailed measurement documentation
│
├── energy_score/                # Energy scoring package
│   ├── __init__.py
│   ├── calculate_scores.py      # Star rating calculation
│   └── test/                    # Scoring tests
│       ├── __init__.py
│       └── test_scoring.py      # Test scoring functionality
│
├── archive/                     # Archived older functionality
├── results/                     # Energy measurement results (generated)
│   └── {task_name}/
│       └── {hardware}/
│           └── {model}_{timestamp}.json
├── scores/                      # Energy scoring results (generated)
│   └── {task_name}/
│       └── {task}_{hardware}_{timestamp}.json
├── setup.py                     # Package installation configuration
├── requirements.txt             # Core dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## 🔧 Core Features

### 1. Energy Measurement (`measure_energy.py`)

Measure energy consumption of any ML model:

```python
from energy_measurement.measure_energy import measure_energy

def my_inference_function(sample):
    # Your model inference code
    return processed_result

# Measure energy
results = measure_energy(
    inference_fn=my_inference_function,
    dataset=my_dataset,
    model_name="my_model",
    task_name="text-classification",
    hardware="CPU",
    num_samples=1000
)

print(f"Energy: {results['kwh_per_1000_queries']:.4f} kWh/1k queries")
```

**Features:**
- ✅ **CodeCarbon Integration**: Accurate energy tracking with PUE=1.2
- ✅ **Hardware Support**: CPU, T4, V100, A100, H100, M1, M2
- ✅ **Progress Tracking**: Shows progress every 100 samples
- ✅ **JSON Output**: Structured results with timestamps
- ✅ **Error Handling**: Validates hardware and datasets

### 2. Energy Scoring (`calculate_scores.py`)

Calculate star ratings for energy efficiency:

```python
from energy_score.calculate_scores import calculate_scores, print_scores

# Calculate scores and display
scores = calculate_scores('image-classification', 'CPU')
print_scores(scores)

# Calculate scores and save to file (recommended)
scores = calculate_scores('image-classification', 'CPU', output_dir='scores')
# Saves to: scores/image-classification/image-classification_CPU_20250111_143022.json
print_scores(scores)
```

**Features:**
- ✅ **Star Rating System**: Quintile-based scoring (5 stars = top 20%, 1 star = bottom 20%)
- ✅ **Auto-Save Results**: Optional output_dir parameter saves scores to timestamped JSON files
- ✅ **Organized Storage**: Files organized by task name with hardware and timestamp in filename
- ✅ **Percentile Rankings**: Each model gets a percentile ranking (0-100, lower is better)
- ✅ **Energy Statistics**: Min, max, and median energy consumption for the task

**Star Rating System:**
- ⭐⭐⭐⭐⭐ **5 stars**: Top 20% (most efficient)
- ⭐⭐⭐⭐☆ **4 stars**: Next 20%
- ⭐⭐⭐☆☆ **3 stars**: Middle 20%
- ⭐⭐☆☆☆ **2 stars**: Next 20%
- ⭐☆☆☆☆ **1 star**: Bottom 20% (least efficient)

**Saved Score Format:**
```json
{
  "task_name": "image-classification",
  "hardware": "CPU",
  "num_models": 10,
  "models": [
    {
      "model_name": "efficientnet-b0",
      "star_rating": 5,
      "kwh_per_1000_queries": 0.0250,
      "percentile": 0
    }
  ],
  "energy_range": {"min": 0.0250, "max": 0.1850, "median": 0.0680},
  "score_timestamp": "2025-01-11T14:30:22.123456"
}
```

## 🧪 Testing

### Dummy Model Test (No Dependencies)
```bash
python energy_measurement/test/test_dummy.py
```

### PyTorch Model Test (Real Neural Network)
```bash
python energy_measurement/test/test_pytorch.py
```

### Scoring Function Test
```bash
python energy_score/test/test_scoring.py
```

## 📊 Example Results

**CPU Models (10 models):**
```
⭐⭐⭐⭐⭐ efficientnet-b0: 0.0250 kWh (percentile: 0)
⭐⭐⭐⭐⭐ mobilenet-v2: 0.0420 kWh (percentile: 11)
⭐⭐⭐⭐☆ resnet18: 0.0680 kWh (percentile: 22)
⭐⭐⭐⭐☆ resnet50: 0.1250 kWh (percentile: 33)
⭐⭐⭐☆☆ vgg16: 0.1850 kWh (percentile: 44)
```

**GPU Models (5 models):**
```
⭐⭐⭐⭐⭐ efficientnet-b0-gpu: 0.0150 kWh (percentile: 0)
⭐⭐⭐⭐☆ resnet50-gpu: 0.0350 kWh (percentile: 25)
⭐⭐⭐☆☆ vgg16-gpu: 0.0550 kWh (percentile: 50)
```

## 📋 Supported Hardware

- **CPU**: CPU-only execution
- **T4**: NVIDIA Tesla T4 16GB
- **V100**: NVIDIA Tesla V100 32GB
- **A100**: NVIDIA A100 40GB/80GB
- **H100**: NVIDIA H100 80GB
- **M1/M2**: Apple Silicon (development)

## 📁 Output Format

Results are saved as JSON files:

```json
{
  "model_name": "mobilenet-v2",
  "task_name": "image-classification",
  "hardware": "CPU",
  "timestamp": "2025-10-25T17:56:08.882447",
  "num_samples": 100,
  "energy_kwh": 0.000008,
  "co2_kg": 0.000002,
  "duration_seconds": 3.16,
  "kwh_per_1000_queries": 0.000084
}
```

## 🔧 Requirements

- Python 3.8+
- CodeCarbon >= 2.4.0
- NumPy >= 1.20.0
- PyTorch >= 1.9.0 (optional, for PyTorch tests)

## 📚 Documentation

- **Main Documentation**: `energy_measurement/README.md`
- **API Reference**: See docstrings in source files
- **Examples**: `energy_measurement/example_usage.py`

## 🗂️ Archive

Older functionality has been moved to the `archive/` directory:
- `archive/energy-compare/`: Previous energy comparison framework
- `archive/ml_energy_score/`: Previous ML energy scoring system
- `archive/tests/`: Previous test suites
- `archive/`: Configuration system and demos

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

This is a focused energy measurement toolkit. For contributions, please ensure:
1. All tests pass
2. Code follows the existing style
3. New features include comprehensive tests
4. Documentation is updated

## Author

**Pascal Joly**
Sustainability Consulting
IT Climate Ed, LLC

## Acknowledgments

This project is inspired by and builds upon:
- **[Hugging Face AI Energy Score](https://huggingface.github.io/AIEnergyScore/)**: Methodology for measuring and comparing ML model energy consumption
- **[CodeCarbon](https://codecarbon.io/)**: Python library for tracking carbon emissions from computing

---

**Created**: October 25, 2025
**License:** MIT
**Status:** Open source tool for sustainable AI development
