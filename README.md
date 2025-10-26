# AI Energy Score for Enterprise

- A comprehensive suite of tools for measuring and comparing energy consumption and CO2 emissions of machine learning models during inference. 
- Extending the capabilities of the Hugging Face Energy Score to internal models, beyond Transformer and Diffuser architecture, and for custom use cases. https://huggingface.github.io/AIEnergyScore/#documentation
- Producing a internal energy score for models running on the same hardware and for the same use case and dataset

> **Note**: Please raise questions with the Autodesk ESG team.

## 🏗️ Project Structure

This repository contains two main components:

### 1. ML Energy Score (`ml_energy_score/`)
A production-ready tool for measuring energy consumption and CO2 emissions of ML models during inference.

**Key Features:**
- Real energy measurement using CodeCarbon
- Multi-task support (text classification, image classification, text generation, Q&A, etc.)
- Hardware flexibility (CPU, GPU, Apple Silicon)
- HuggingFace integration
- Comprehensive metrics (energy, CO2, throughput, timing)
- Organized JSON output format

### 2. Energy Compare (`energy-compare/`)
A framework for comparing and scoring different ML models based on their energy efficiency.

**Key Features:**
- Model comparison framework
- Scoring algorithms for energy efficiency
- Configurable comparison metrics
- Test suite for validation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone git@git.autodesk.com:saas/ai_energy_score_corporate.git
cd ai_energy_score_corporate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

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
    hardware="CPU",
    num_samples=50,
    output_dir="results"
)

print(f"Energy consumed: {result['energy_kwh']} kWh")
print(f"CO2 emissions: {result['co2_kg']} kg")
print(f"Throughput: {result['samples_per_second']} samples/sec")
```

### Run Demo

```bash
# Quick demo with a small model
python ml_energy_score/run_demo.py

# Comprehensive testing
python ml_energy_score/test_real_model.py
```

## 📊 Supported Tasks

| Task | Example Models | Description |
|------|----------------|-------------|
| `text-classification` | `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment, topic classification |
| `sentiment-analysis` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment detection |
| `image-classification` | `google/vit-base-patch16-224` | Image recognition |
| `text-generation` | `gpt2`, `microsoft/DialoGPT-medium` | Text completion |
| `question-answering` | `distilbert-base-cased-distilled-squad` | Q&A systems |
| `token-classification` | `dbmdz/bert-large-cased-finetuned-conll03-english` | NER, POS tagging |
| `summarization` | `facebook/bart-large-cnn` | Text summarization |
| `translation` | `Helsinki-NLP/opus-mt-en-de` | Language translation |

## 🔧 Hardware Support

- **CPU**: CPU-only inference (works everywhere)
- **T4**: NVIDIA Tesla T4 16GB
- **V100**: NVIDIA Tesla V100 32GB  
- **A100**: NVIDIA A100 40GB/80GB
- **M1/M2**: Apple Silicon (development)

## 📁 Output Format

Results are saved as structured JSON files:

```json
{
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "task": "text-classification",
  "hardware": "CPU",
  "energy_kwh": 0.000123,
  "co2_kg": 0.000056,
  "duration_seconds": 2.45,
  "samples_per_second": 20.4,
  "kwh_per_1000_queries": 2.46
}
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest energy-compare/tests/ -v
python -m pytest ml_energy_score/test_measure.py -v

# Test with real models
python ml_energy_score/test_real_model.py
```

## 📚 Documentation

- **ML Energy Score**: See `ml_energy_score/README.md` for detailed documentation
- **User Guide**: See `ml_energy_score/USER_GUIDE.md` for testing with real models
- **Testing Summary**: See `ml_energy_score/TESTING_SUMMARY.md` for test results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [CodeCarbon](https://codecarbon.io/) for energy measurement
- [HuggingFace](https://huggingface.co/) for model ecosystem
- [PyTorch](https://pytorch.org/) for deep learning framework
