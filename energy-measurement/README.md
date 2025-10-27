# Energy Measurement Script

A simple Python script for measuring energy consumption of ML models using CodeCarbon with flexible, reproducible sampling.

## Features

- **Simple API**: Easy-to-use function for measuring model energy consumption
- **Flexible Sampling**: Fully customizable sample sizes (10-5000+ samples)
- **Reproducible Comparisons**: Use same seed for fair model comparisons
- **CodeCarbon Integration**: Uses CodeCarbon for accurate energy tracking
- **Hardware Validation**: Validates against supported hardware types
- **Progress Tracking**: Shows progress every 100 samples
- **JSON Output**: Saves results in structured JSON format
- **Error Handling**: Graceful handling of edge cases

## Installation

```bash
pip install codecarbon torch
```

## Quick Start

```python
from measure_energy import measure_energy

def my_inference_function(sample):
    # Your model inference code here
    return processed_result

# Small dataset - use 100 samples
results = measure_energy(
    inference_fn=my_inference_function,
    dataset=small_dataset,
    model_name="my_model",
    task_name="text-classification",
    hardware="CPU",
    num_samples=100,  # Flexible sample size
    seed=42  # Reproducible sampling
)

# Standard benchmarking - 1000 samples
results = measure_energy(
    inference_fn=my_inference_function,
    dataset=large_dataset,
    model_name="my_model",
    task_name="text-classification",
    hardware="CPU",
    num_samples=1000,  # Default
    seed=42
)

# Compare model versions (use same seed!)
results_v1 = measure_energy(..., model_name="v1", seed=42)
results_v2 = measure_energy(..., model_name="v2", seed=42)

print(f"Energy consumption: {results['kwh_per_1000_queries']:.4f} kWh/1k queries")
```

## Usage Guide

### `measure_energy()`

Measure energy consumption of a model.

**Parameters:**
- `inference_fn` (Callable): Function that processes one sample
- `dataset` (Iterable): Dataset to process
- `model_name` (str): Model identifier
- `task_name` (str): Task description (e.g., "text-classification")
- `hardware` (str): Hardware type (must be in SUPPORTED_HARDWARE)
- `num_samples` (int, optional): Number of samples to process (default: 1000)
- `seed` (int, optional): Random seed for reproducible sampling (default: 42)
- `output_dir` (str, optional): Output directory (default: "results")

**Parameter Details:**
- **`inference_fn`**: Your model inference function that takes a sample and returns processed result
- **`dataset`**: Any iterable (list, HuggingFace dataset, generator, etc.)
- **`model_name`**: Unique identifier for your model (e.g., "resnet50", "gpt-2")
- **`task_name`**: Descriptive label for your ML task (any string)
- **`hardware`**: Hardware identifier (see Hardware Support below for CodeCarbon compatibility)
- **`num_samples`**: Fully flexible (10-5000+ samples). Use 100 for small datasets, 1000 for standard benchmarking
- **`seed`**: Critical for fair comparisons. Use same seed when comparing model versions
- **`output_dir`**: Directory where JSON results will be saved

**Returns:**
- `dict`: Measurement results with energy, CO2, and timing data

**Return Value Structure:**
```json
{
  "model_name": "resnet50",
  "task_name": "image-classification", 
  "hardware": "CPU",
  "timestamp": "2025-10-25T18:42:41.123456",
  "num_samples": 100,
  "seed": 42,
  "energy_kwh": 0.000008,
  "co2_kg": 0.000002,
  "duration_seconds": 3.16,
  "kwh_per_1000_queries": 0.000084
}
```

**Raises:**
- `ValueError`: If hardware is not supported or dataset is empty

## Flexible Sample Sizes

The `num_samples` parameter is fully customizable:

- **Small datasets (<1000 samples)**: Use what you have (e.g., `num_samples=100`)
- **Standard benchmarking**: Use 1000 samples (default)
- **Large-scale testing**: Use any value you need (e.g., `num_samples=5000`)

All results are normalized to "per 1000 queries" for fair comparison.

## Reproducible Comparisons

Always use the same `seed` when comparing model versions:

```python
# Test baseline model
results_baseline = measure_energy(
    ..., 
    model_name="baseline", 
    num_samples=500,  # Custom size
    seed=42  # Fixed seed
)

# Test optimized model (same seed!)
results_optimized = measure_energy(
    ..., 
    model_name="optimized", 
    num_samples=500,  # Same size
    seed=42  # Same seed = same samples
)

# Now fairly comparable
improvement = (results_baseline['kwh_per_1000_queries'] - 
               results_optimized['kwh_per_1000_queries']) / \
               results_baseline['kwh_per_1000_queries']
print(f"Energy improvement: {improvement*100:.1f}%")
```

## Task Name Parameter

The `task_name` is a descriptive label for your ML task. You can use any string that describes what your model does:

**Examples:**
- `"image-classification"` - Image classification
- `"text-generation"` - Text generation  
- `"speech-recognition"` - Speech recognition
- `"recommendation-system"` - Recommendation system
- `"anomaly-detection"` - Anomaly detection
- `"custom-model"` - Custom model
- `"experimental-task"` - Experimental task

**Key Points:**
- ✅ **Any string is valid** - No restrictions on task names
- ✅ **Descriptive purpose** - Just for organization and identification
- ✅ **Independent of functionality** - The tool doesn't validate task names
- ✅ **Use what makes sense** - Choose a name that describes your specific use case

## Hardware Support

The energy measurement tool uses [CodeCarbon](https://mlco2.github.io/codecarbon/) for hardware monitoring, which supports:

### **CPU Support:**
- **Linux**: Intel and AMD processors via Intel RAPL interface
- **Windows/macOS (Intel)**: Intel Power Gadget (discontinued, may have limitations)
- **Apple Silicon**: M1, M2 chips via `powermetrics` (requires `sudo` privileges)

### **GPU Support:**
- **NVIDIA GPUs**: All NVIDIA GPUs via `nvidia-ml-py` library
  - Tesla series (T4, V100)
  - Ampere series (A100, A100-80GB)
  - Hopper series (H100, H100-SXM)
  - RTX series (RTX 4090, RTX 4080, etc.)
  - Any other NVIDIA GPU with nvidia-ml-py support

### **RAM Support:**
- **Automatic estimation** based on RAM slots and architecture
- **Custom override** available via `force_ram_power` parameter

### **Platform-Specific Notes:**
- **Linux**: Requires access to `/sys/class/powercap/intel-rapl/` files
- **Apple Silicon**: Requires `sudo` privileges for `powermetrics`
- **Windows**: Intel Power Gadget limitations may apply

**Reference**: [CodeCarbon Methodology Documentation](https://mlco2.github.io/codecarbon/methodology.html)

## Output Format

Results are saved as JSON files in the following structure:

```
results/
├── task_name/
│   └── model_name_timestamp.json
```

**JSON Structure:**
```json
{
  "model_name": "my_model",
  "task_name": "text-classification",
  "hardware": "CPU",
  "timestamp": "2024-01-15T10:30:45.123456",
  "num_samples": 1000,
  "energy_kwh": 0.042,
  "co2_kg": 0.018,
  "duration_seconds": 125.3,
  "kwh_per_1000_queries": 0.042
}
```

## Examples

### Basic Usage

```python
from measure_energy import measure_energy

def simple_inference(sample):
    # Your inference logic
    return result

results = measure_energy(
    inference_fn=simple_inference,
    dataset=my_dataset,
    model_name="model_v1",
    task_name="classification",
    hardware="CPU"
)
```

### PyTorch Model

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return self.layers(x)

model = MyModel()
model.eval()

def pytorch_inference(sample):
    with torch.no_grad():
        return model(torch.tensor(sample))

results = measure_energy(
    inference_fn=pytorch_inference,
    dataset=dataset,
    model_name="pytorch_model",
    task_name="image-classification",
    hardware="CPU"
)
```

## Testing

Run the test suite:

```bash
python test/test_measure.py
```

Run examples:

```bash
python example_usage.py
```

## Error Handling

The script handles common error cases:

- **Invalid Hardware**: Raises `ValueError` for unsupported hardware
- **Empty Dataset**: Raises `ValueError` for empty datasets
- **Progress Tracking**: Shows progress every 100 samples
- **Graceful Cleanup**: Ensures CodeCarbon tracker is stopped

## Requirements

- Python 3.7+
- CodeCarbon
- PyTorch (optional, for examples)
- NumPy (optional, for examples)

## License

MIT License
