# 🚀 User Guide: Testing with Real Models

## 📋 Quick Answer: Model Support

**✅ YES! Your tool supports both:**
- **HuggingFace Models** (from Hub or local cache)
- **Local Models** (saved on your filesystem)

## 🎯 Quick Start - 3 Ways to Test

### 1. 🏃‍♂️ **Fastest Way - Run the Demo**
```bash
cd "/Users/Pascal/Documents/consulting - pro/autodesk/EStool"
python ml_energy_score/run_demo.py
```
This will:
- Use a small BERT model (quick download)
- Test with 5 samples
- Show complete energy measurements
- Save results to `demo_results/`

### 2. 🧪 **Real Model Test Script**
```bash
cd "/Users/Pascal/Documents/consulting - pro/autodesk/EStool"
python ml_energy_score/test_real_model.py
```
This runs comprehensive tests with real models.

### 3. 📝 **Custom Testing (Copy & Paste)**
```python
# Navigate to your project directory first
cd "/Users/Pascal/Documents/consulting - pro/autodesk/EStool"

# Then run this Python code:
python -c "
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from ml_energy_score.measure import measure_model_energy

# Create simple test data
class TestDataset:
    def __init__(self):
        self.data = [
            {'text': 'I love this movie! Great acting.'},
            {'text': 'Terrible film, waste of time.'},
            {'text': 'Pretty good, worth watching.'}
        ]
    def __len__(self): return len(self.data)
    def __iter__(self): return iter(self.data)
    def select(self, indices): 
        new_ds = TestDataset()
        new_ds.data = [self.data[i] for i in indices if i < len(self.data)]
        return new_ds

# Test with HuggingFace model
result = measure_model_energy(
    model_path='prajjwal1/bert-tiny',  # Small, fast model
    task='text-classification',
    dataset=TestDataset(),
    hardware='CPU',
    num_samples=3,
    output_dir='my_test_results'
)

print('✅ Success!')
print(f'Energy: {result[\"energy_kwh\"]} kWh')
print(f'Duration: {result[\"duration_seconds\"]} seconds')
print(f'Results saved to: my_test_results/')
"
```

## 🤖 Model Support Details

### **HuggingFace Models** ✅

#### **From HuggingFace Hub:**
```python
# Popular models that work out of the box:
models = [
    "distilbert-base-uncased-finetuned-sst-2-english",  # Sentiment
    "prajjwal1/bert-tiny",                              # Small BERT
    "cardiffnlp/twitter-roberta-base-sentiment-latest", # Twitter sentiment
    "microsoft/DialoGPT-medium",                        # Text generation
    "google/vit-base-patch16-224",                      # Image classification
]

result = measure_model_energy(
    model_path="distilbert-base-uncased-finetuned-sst-2-english",
    task="text-classification",
    dataset=your_dataset,
    hardware="CPU"
)
```

#### **From Local HuggingFace Cache:**
```python
# If you've downloaded models before, they're cached locally
# The tool automatically finds them
result = measure_model_energy(
    model_path="bert-base-uncased",  # Uses cached version
    task="text-classification",
    dataset=your_dataset,
    hardware="CPU"
)
```

### **Local Models** ✅

#### **Local Directory Path:**
```python
# Point to a local model directory
result = measure_model_energy(
    model_path="/path/to/your/local/model",  # Local filesystem path
    task="text-classification",
    dataset=your_dataset,
    hardware="CPU"
)
```

#### **Relative Path:**
```python
# Relative to current directory
result = measure_model_energy(
    model_path="./models/my_custom_model",
    task="text-classification", 
    dataset=your_dataset,
    hardware="CPU"
)
```

## 📊 Supported Tasks & Example Models

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

## 🔧 Hardware Options

```python
# Choose your hardware:
hardware_options = [
    "CPU",        # CPU-only (works everywhere)
    "T4",         # NVIDIA Tesla T4
    "V100",       # NVIDIA Tesla V100  
    "A100",       # NVIDIA A100 40GB
    "A100-80GB",  # NVIDIA A100 80GB
    "M1",         # Apple M1 (your system!)
    "M2"          # Apple M2
]

# Your system will auto-detect: "CPU: arm; Apple Silicon (M1/M2)"
```

## 📁 Where Results Are Saved

Results are organized in this structure:
```
output_dir/
├── text-classification/
│   ├── CPU/
│   │   ├── bert-tiny_20251003T123045.json
│   │   └── distilbert_20251003T124567.json
│   └── T4/
│       └── roberta_20251003T125678.json
├── image-classification/
│   └── CPU/
│       └── vit_20251003T126789.json
└── text-generation/
    └── CPU/
        └── gpt2_20251003T127890.json
```

## 🎯 Real-World Examples

### **Example 1: Compare Sentiment Models**
```python
from ml_energy_score.measure import measure_model_energy
from datasets import load_dataset

# Load real dataset
dataset = load_dataset("imdb", split="test[:100]")

# Test different models
models = [
    "prajjwal1/bert-tiny",
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
        num_samples=50,
        output_dir="sentiment_comparison"
    )
    results.append({
        'model': model,
        'energy_kwh': result['energy_kwh'],
        'samples_per_sec': result['samples_per_second']
    })

# Compare results
for r in results:
    print(f"{r['model']}: {r['energy_kwh']:.6f} kWh, {r['samples_per_sec']:.1f} samples/sec")
```

### **Example 2: Test Your Own Local Model**
```python
# If you have a local model directory
result = measure_model_energy(
    model_path="/Users/Pascal/my_models/custom_bert",
    task="text-classification",
    dataset=your_dataset,
    hardware="CPU",
    num_samples=100,
    output_dir="custom_model_results"
)
```

### **Example 3: Image Classification**
```python
from datasets import load_dataset

# Load image dataset
dataset = load_dataset("cifar10", split="test[:50]")

result = measure_model_energy(
    model_path="google/vit-base-patch16-224",
    task="image-classification",
    dataset=dataset,
    hardware="CPU",
    num_samples=20,
    output_dir="image_results"
)
```

## 🔍 Understanding Results

Each result file contains:
```json
{
  "model_name": "bert-tiny",
  "energy_kwh": 0.000123,           // Total energy consumed
  "co2_kg": 0.000056,               // CO2 emissions
  "duration_seconds": 2.45,         // Total time
  "samples_per_second": 20.4,       // Throughput
  "kwh_per_1000_queries": 2.46,     // Efficiency metric
  "inference_results_sample": [...] // Sample predictions
}
```

## 🚨 Troubleshooting

### **Model Not Found**
```
Error: Model 'my-model' not found
```
**Solution:** Check model name on HuggingFace Hub or verify local path

### **Task Not Supported**
```
Error: Task 'my-task' not supported
```
**Solution:** Use one of the 8 supported tasks listed above

### **Out of Memory**
```
Error: CUDA out of memory
```
**Solution:** Use smaller model or reduce `num_samples`

### **Slow Downloads**
```
Model downloading slowly...
```
**Solution:** Use smaller models like `prajjwal1/bert-tiny` for testing

## 📚 File Locations Summary

| File | Purpose | Location |
|------|---------|----------|
| **README.md** | Complete documentation | `ml_energy_score/README.md` |
| **run_demo.py** | Quick start demo | `ml_energy_score/run_demo.py` |
| **test_real_model.py** | Comprehensive testing | `ml_energy_score/test_real_model.py` |
| **USER_GUIDE.md** | This guide! | `ml_energy_score/USER_GUIDE.md` |
| **measure.py** | Main implementation | `ml_energy_score/measure.py` |

## 🎉 You're Ready!

Your energy measurement tool is **production-ready** and supports both HuggingFace and local models. Start with the demo script and then explore with your own models!
