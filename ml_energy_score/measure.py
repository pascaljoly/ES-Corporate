"""Model energy measurement function."""

import json
import time
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Union, Any, Dict, List
from codecarbon import EmissionsTracker
from .config import SUPPORTED_HARDWARE, SUPPORTED_TASKS, PUE, MEASURE_POWER_SECS

# Import ML libraries with fallback handling
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. Install with: pip install torch")


def measure_model_energy(
    model_path: str,
    task: str,
    dataset,
    hardware: str,
    num_samples: int = 1000,
    output_dir: str = "results"
) -> dict:
    """
    Measure energy consumption of a model on a given task.
    
    Args:
        model_path: HuggingFace model name or local path
        task: Task name (e.g., 'image-classification')
        dataset: Dataset to run inference on
        hardware: Hardware type from SUPPORTED_HARDWARE
        num_samples: Number of samples to process (default 1000)
        output_dir: Directory to save results
        
    Returns:
        Dictionary with measurement results
        
    Raises:
        ValueError: If hardware not in SUPPORTED_HARDWARE
    """
    # Validate hardware
    if hardware not in SUPPORTED_HARDWARE:
        raise ValueError(
            f"Hardware '{hardware}' not supported. "
            f"Choose from: {list(SUPPORTED_HARDWARE.keys())}"
        )
    
    # Validate task
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Task '{task}' not supported. "
            f"Choose from: {SUPPORTED_TASKS}"
        )
    
    # TODO: Auto-detect hardware and warn if mismatch
    
    # Sample dataset
    if len(dataset) > num_samples:
        import random
        indices = random.sample(range(len(dataset)), num_samples)
        test_dataset = dataset.select(indices)
    else:
        test_dataset = dataset
        num_samples = len(dataset)
    
    # Load model
    model, tokenizer = _load_model(model_path, task, hardware)
    
    # Start energy tracking
    tracker = EmissionsTracker(
        project_name=f"{model_path.replace('/', '_')}_{task}",
        pue=PUE,
        measure_power_secs=MEASURE_POWER_SECS,
        save_to_file=False  # We'll handle file saving
    )
    
    # Start timing and energy tracking
    start_time = time.time()
    tracker.start()
    
    # Run inference on test_dataset
    print(f"Running inference on {num_samples} samples...")
    inference_results = _run_inference(model, tokenizer, test_dataset, task, num_samples)
    
    # Stop tracking
    emissions = tracker.stop()
    end_time = time.time()
    duration_seconds = end_time - start_time
    
    # Extract energy data from emissions tracker
    energy_kwh = getattr(emissions, 'energy_consumed', 0.0) if emissions else 0.0
    co2_kg = getattr(emissions, 'emissions', 0.0) if emissions else 0.0
    
    # Calculate metrics
    kwh_per_1000_queries = (energy_kwh / num_samples * 1000) if num_samples > 0 else 0.0
    
    # Prepare results
    results = {
        "model_name": Path(model_path).name,
        "model_path": model_path,
        "task": task,
        "hardware": hardware,
        "hardware_detected": _detect_hardware(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_samples": num_samples,
        "energy_kwh": energy_kwh,
        "co2_kg": co2_kg,
        "duration_seconds": duration_seconds,
        "kwh_per_1000_queries": kwh_per_1000_queries,
        "samples_per_second": num_samples / duration_seconds if duration_seconds > 0 else 0.0,
        "inference_results_sample": inference_results[:3] if inference_results else []  # First 3 for debugging
    }
    
    # Save results
    _save_results(results, output_dir)
    
    return results


def _detect_hardware() -> str:
    """Auto-detect hardware. Returns description string."""
    hardware_info = []
    
    # Detect CPU
    cpu_info = platform.processor() or platform.machine()
    if cpu_info:
        hardware_info.append(f"CPU: {cpu_info}")
    
    # Detect GPU (NVIDIA)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_names = result.stdout.strip().split('\n')
            for gpu in gpu_names:
                if gpu.strip():
                    hardware_info.append(f"GPU: {gpu.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Detect Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        hardware_info.append("Apple Silicon (M1/M2)")
    
    # Detect PyTorch CUDA availability
    if TORCH_AVAILABLE and torch.cuda.is_available():
        hardware_info.append(f"CUDA: {torch.cuda.get_device_name()}")
    
    return "; ".join(hardware_info) if hardware_info else "Unknown"


def _save_results(results: dict, output_dir: str):
    """Save results to JSON file organized by task/hardware."""
    # Create directory structure: results/task/hardware/
    output_path = Path(output_dir) / results["task"] / results["hardware"]
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = results["timestamp"].replace(":", "").replace("-", "")
    filename = f"{results['model_name']}_{timestamp}.json"
    
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {filepath}")


def _load_model(model_path: str, task: str, hardware: str) -> tuple[Any, Any]:
    """
    Load model and tokenizer based on task and model path.
    
    Returns:
        Tuple of (model, tokenizer) or (pipeline, None) for HuggingFace models
    """
    if not HF_AVAILABLE:
        raise RuntimeError("transformers library not available. Install with: pip install transformers")
    
    print(f"Loading model: {model_path}")
    
    # Determine device
    device = _get_device(hardware)
    
    try:
        # Try to load as HuggingFace pipeline first (easiest)
        if task in ["text-classification", "sentiment-analysis"]:
            model = pipeline("text-classification", model=model_path, device=device)
            return model, None
        elif task in ["image-classification"]:
            model = pipeline("image-classification", model=model_path, device=device)
            return model, None
        elif task in ["text-generation"]:
            model = pipeline("text-generation", model=model_path, device=device)
            return model, None
        elif task in ["question-answering"]:
            model = pipeline("question-answering", model=model_path, device=device)
            return model, None
        else:
            # Fallback: load raw model and tokenizer
            print(f"Loading raw model for task: {task}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            
            if TORCH_AVAILABLE and device != -1:
                model = model.to(f"cuda:{device}" if device >= 0 else "cpu")
            
            return model, tokenizer
            
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        raise


def _get_device(hardware: str) -> int:
    """Determine device based on hardware configuration."""
    if hardware == "CPU":
        return -1  # CPU device for transformers
    elif TORCH_AVAILABLE and torch.cuda.is_available():
        return 0  # First GPU
    else:
        return -1  # Fallback to CPU


def _run_inference(model: Any, tokenizer: Any, dataset: Any, task: str, num_samples: int) -> List[Dict]:
    """
    Run inference on the dataset.
    
    Args:
        model: Loaded model (pipeline or raw model)
        tokenizer: Tokenizer (None for pipelines)
        dataset: Dataset to run inference on
        task: Task type
        num_samples: Number of samples to process
        
    Returns:
        List of inference results
    """
    results = []
    
    print(f"Starting inference for {num_samples} samples...")
    
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
            
        # Show progress every 10% or every 100 samples, whichever is smaller
        progress_interval = max(1, min(100, num_samples // 10))
        if i % progress_interval == 0:
            print(f"Progress: {i}/{num_samples} ({i/num_samples*100:.1f}%)")
        
        try:
            # Handle different task types
            if task in ["text-classification", "sentiment-analysis"]:
                text = _extract_text_from_sample(sample)
                if tokenizer is None:  # This means we have a pipeline
                    result = model(text)
                else:  # Raw model - use pipeline approach
                    result = {"prediction": "raw_model_placeholder"}
                results.append(result)
                
            elif task in ["image-classification"]:
                image = _extract_image_from_sample(sample)
                if image is not None:
                    if tokenizer is None:  # Pipeline
                        result = model(image)
                    else:  # Raw model
                        result = {"prediction": "raw_model_placeholder"}
                    results.append(result)
                    
            elif task in ["text-generation"]:
                text = _extract_text_from_sample(sample)
                if tokenizer is None:  # Pipeline
                    result = model(text, max_length=50, do_sample=False)
                else:  # Raw model
                    result = {"generated_text": "raw_model_placeholder"}
                results.append(result)
                
            else:
                # Generic handling - just simulate work
                time.sleep(0.001)  # Small delay to simulate processing
                results.append({"prediction": f"generic_result_{i}"})
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            results.append({"error": str(e)})
    
    print(f"Inference completed: {len(results)} results")
    return results


def _extract_text_from_sample(sample: Dict) -> str:
    """Extract text from a dataset sample."""
    # Common text fields in datasets
    text_fields = ['text', 'sentence', 'content', 'input', 'question', 'review']
    
    for field in text_fields:
        if field in sample and sample[field]:
            return str(sample[field])
    
    # Fallback: convert entire sample to string
    return str(sample)


def _extract_image_from_sample(sample: Dict) -> Any:
    """Extract image from a dataset sample."""
    # Common image fields in datasets
    image_fields = ['image', 'img', 'picture', 'photo']
    
    for field in image_fields:
        if field in sample and sample[field] is not None:
            return sample[field]
    
    return None
