# ML Energy Score

A comprehensive tool for measuring energy consumption and CO2 emissions of machine learning models during inference.

## 🚀 Features

- **Real Energy Measurement**: Uses CodeCarbon to track actual energy consumption
- **Multi-Task Support**: Text classification, image classification, text generation, Q&A, and more
- **Hardware Flexibility**: Supports CPU, GPU (NVIDIA), and Apple Silicon
- **HuggingFace Integration**: Easy loading of pre-trained models
- **Comprehensive Metrics**: Energy, CO2, throughput, and timing measurements
- **Organized Output**: Results saved in structured JSON format
- **Progress Tracking**: Real-time progress updates during inference

## 📦 Installation

```bash
# Install required dependencies
pip install codecarbon transformers torch datasets

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 Quick Start

```python
from ml_energy_score.measure import measure_model_energy
from datasets import load_dataset

# Load a dataset
dataset = load_dataset("imdb", split="test[:100]")

# Measure energy consumption
result = measure_model_energy(
    model_path="distilbert-base-uncased-finetuned-sst-2-english",
    task="text-classification",
    dataset=dataset,
    hardware="CPU",  # or "T4", "V100", "A100", etc.
    num_samples=50,
    output_dir="results"
)

print(f"Energy consumed: {result['energy_kwh']} kWh")
print(f"CO2 emissions: {result['co2_kg']} kg")
print(f"Throughput: {result['samples_per_second']} samples/sec")
```

## 🔧 Configuration

### Supported Hardware
- `CPU`: CPU-only inference
- `T4`: NVIDIA Tesla T4 16GB
- `V100`: NVIDIA Tesla V100 32GB  
- `A100`: NVIDIA A100 40GB/80GB
- `M1`/`M2`: Apple Silicon (development)

### Supported Tasks
- `text-classification`: Sentiment analysis, topic classification
- `image-classification`: Image recognition tasks
- `text-generation`: Text completion, chatbots
- `question-answering`: Q&A systems
- `sentiment-analysis`: Sentiment detection
- `token-classification`: NER, POS tagging
- `summarization`: Text summarization
- `translation`: Language translation

## 📊 Output Format

Results are saved as JSON files with the following structure:

```json
{
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "model_path": "distilbert-base-uncased-finetuned-sst-2-english",
  "task": "text-classification",
  "hardware": "CPU",
  "hardware_detected": "CPU: arm; Apple Silicon (M1/M2)",
  "timestamp": "2025-10-03T06:21:03.273566Z",
  "num_samples": 50,
  "energy_kwh": 0.000123,
  "co2_kg": 0.000056,
  "duration_seconds": 2.45,
  "kwh_per_1000_queries": 2.46,
  "samples_per_second": 20.4,
  "inference_results_sample": [...]
}
```

## 🏗️ Architecture

The tool consists of several key components:

### Core Functions
- `measure_model_energy()`: Main measurement function
- `_load_model()`: HuggingFace model loading
- `_run_inference()`: Inference execution with progress tracking
- `_detect_hardware()`: Automatic hardware detection
- `_save_results()`: Structured result saving

### Configuration
- `config.py`: Hardware and task definitions
- Customizable CodeCarbon settings (PUE, sampling frequency)

### Testing
- Comprehensive test suite with pytest
- Hardware validation tests
- Real model integration tests

## 🧪 Testing

```bash
# Run all tests
python -m pytest ml_energy_score/test_measure.py -v

# Test with real model
python ml_energy_score/test_real_model.py
```

## 📈 Advanced Usage

### Custom Datasets

```python
# Create custom dataset
class CustomDataset:
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __iter__(self):
        for text in self.texts:
            yield {"text": text}
    
    def select(self, indices):
        selected = CustomDataset([self.texts[i] for i in indices])
        return selected

# Use with energy measurement
dataset = CustomDataset(["Sample text 1", "Sample text 2"])
result = measure_model_energy(
    model_path="your-model",
    task="text-classification", 
    dataset=dataset,
    hardware="CPU"
)
```

### Batch Processing

```python
# Measure multiple models
models = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
]

results = []
for model in models:
    result = measure_model_energy(
        model_path=model,
        task="text-classification",
        dataset=dataset,
        hardware="CPU",
        num_samples=100
    )
    results.append(result)
```

## 🔍 Hardware Detection

The tool automatically detects:
- CPU architecture (Intel, AMD, Apple Silicon)
- GPU availability (NVIDIA via nvidia-smi)
- CUDA support (PyTorch integration)
- Memory specifications

## ⚡ Performance Tips

1. **Use GPU when available** for faster inference
2. **Batch processing** for multiple samples
3. **Model caching** to avoid reloading
4. **Smaller sample sizes** for quick testing
5. **Progress monitoring** for long runs

## 🐛 Troubleshooting

### Common Issues

**Import Error**: Install missing dependencies
```bash
pip install transformers torch codecarbon
```

**GPU Not Detected**: Check CUDA installation
```bash
nvidia-smi  # Should show GPU info
```

**Model Loading Error**: Verify model name/path
```python
from transformers import pipeline
model = pipeline("text-classification", model="your-model-name")
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality  
4. Run test suite
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [CodeCarbon](https://codecarbon.io/) for energy measurement
- [HuggingFace](https://huggingface.co/) for model ecosystem
- [PyTorch](https://pytorch.org/) for deep learning framework
