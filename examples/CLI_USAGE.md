# CLI Usage Examples

This directory contains example files showing how to use the CLI wrapper.

## Quick Start

```bash
# From the ml-energy-score directory
python energy_measurement/run_measurement.py \
  --inference-source examples/my_model.py \
  --inference-fn predict \
  --dataset-source examples/my_data.py \
  --dataset-fn load_dataset \
  --model-name "my_classifier_v1" \
  --task "text-classification" \
  --hardware CPU \
  --num-samples 50
```

## Your Files

### 1. Create your model file (e.g., `my_model.py`):

```python
def predict(sample):
    # Your model inference code
    result = your_model(sample)
    return result
```

### 2. Create your data file (e.g., `my_data.py`):

```python
def load_dataset():
    # Your data loading code
    data = load_your_data()
    return data
```

### 3. Run the measurement:

```bash
python energy_measurement/run_measurement.py \
  --inference-source my_model.py \
  --inference-fn predict \
  --dataset-source my_data.py \
  --dataset-fn load_dataset \
  --model-name "my_model" \
  --task "my_task" \
  --hardware CPU
```

## CLI Options

```
Required:
  --inference-source    Path to file or module with inference function
  --inference-fn        Name of inference function
  --dataset-source      Path to file or module with dataset
  --dataset-fn          Name of function that returns dataset
  OR
  --dataset-var         Name of variable containing dataset
  --model-name          Name of your model
  --task                Task type (e.g., "text-classification")
  --hardware            Hardware type (CPU, T4, V100, A100, H100, M1, M2)

Optional:
  --num-samples         Number of samples to measure (default: 1000)
  --seed                Random seed (default: 42)
  --output-dir          Where to save results (default: results)
```

## Full Help

```bash
python energy_measurement/run_measurement.py --help
```
