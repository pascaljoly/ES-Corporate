"""Configuration for ML Energy Score."""

SUPPORTED_HARDWARE = {
    "T4": "NVIDIA Tesla T4 16GB",
    "V100": "NVIDIA Tesla V100 32GB",
    "A100": "NVIDIA A100 40GB",
    "A100-80GB": "NVIDIA A100 80GB",
    "CPU": "CPU-only (no GPU)",
    "M1": "Apple M1 (development only)",
    "M2": "Apple M2 (development only)",
}

SUPPORTED_TASKS = [
    "text-classification",
    "sentiment-analysis", 
    "image-classification",
    "text-generation",
    "question-answering",
    "token-classification",
    "summarization",
    "translation"
]

# CodeCarbon settings
PUE = 1.2  # Power Usage Effectiveness (20% infrastructure overhead)
MEASURE_POWER_SECS = 1  # Sampling frequency
