# Energy Measurement Script

A simple Python script for measuring energy consumption of ML models using CodeCarbon.

## Features

- **Simple API**: Easy-to-use function for measuring model energy consumption
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

# Measure energy
results = measure_energy(
    inference_fn=my_inference_function,
    dataset=my_dataset,
    model_name="my_model",
    task_name="text-classification",
    hardware="CPU",
    num_samples=1000
)

print(f"Energy consumption: {results['kwh_per_1000_queries']:.4f} kWh/1k queries")
```

## API Reference

### `measure_energy()`

Measure energy consumption of a model.

**Parameters:**
- `inference_fn` (Callable): Function that processes one sample
- `dataset` (Iterable): Dataset to process
- `model_name` (str): Model identifier
- `task_name` (str): Task description (e.g., "text-classification")
- `hardware` (str): Hardware type (must be in SUPPORTED_HARDWARE)
- `num_samples` (int, optional): Number of samples to process (default: 1000)
- `output_dir` (str, optional): Output directory (default: "results")

**Returns:**
- `dict`: Measurement results with energy, CO2, and timing data

**Raises:**
- `ValueError`: If hardware is not supported or dataset is empty

## Supported Hardware

- `CPU`: CPU-only execution
- `T4`: NVIDIA Tesla T4 16GB
- `V100`: NVIDIA Tesla V100 32GB
- `A100`: NVIDIA A100 40GB
- `A100-80GB`: NVIDIA A100 80GB
- `H100`: NVIDIA H100 80GB
- `H100-SXM`: NVIDIA H100 SXM 80GB
- `M1`: Apple M1 (development only)
- `M2`: Apple M2 (development only)

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
python test_measure.py
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
