#!/usr/bin/env python3
"""
Simple Energy Measurement Script

This module provides a simple function to measure energy consumption
of ML models using CodeCarbon for tracking.
"""

import json
import time
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Any
from codecarbon import EmissionsTracker
from test.sample_dataset import sample_dataset
from utils.security_utils import (
    sanitize_path_component,
    validate_input_length,
    sanitize_and_validate_path
)

# Import config from parent project
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ml_energy_score.config import SUPPORTED_HARDWARE, PUE
except ImportError:
    # Fallback if config not available
    SUPPORTED_HARDWARE = {
        "CPU": "CPU-only (no GPU)",
        "T4": "NVIDIA Tesla T4 16GB",
        "A10": "NVIDIA A10 24GB",
        "V100": "NVIDIA Tesla V100 32GB",
        "A100": "NVIDIA A100 40GB",
        "A100-80GB": "NVIDIA A100 80GB",
        "H100": "NVIDIA H100 80GB",
        "H100-SXM": "NVIDIA H100 SXM 80GB",
        "M1": "Apple M1 (development only)",
        "M2": "Apple M2 (development only)"
    }
    PUE = 1.2


def measure_energy(
    inference_fn: Callable[[Any], Any],
    dataset: Iterable[Any],
    model_name: str,
    task_name: str,
    hardware: str,
    num_samples: int = 1000,
    seed: int = 42,
    output_dir: str = "results"
) -> dict:
    """
    Measure energy consumption for any inference function.
    
    Args:
        inference_fn: User-provided function that takes a sample and runs inference
        dataset: Any iterable (list, HF dataset, custom data)
        model_name: Model identifier (e.g., "resnet50")
        task_name: Task name (e.g., "image-classification")
        hardware: Hardware type from SUPPORTED_HARDWARE
        num_samples: Number of samples to process (default 1000).
                    FULLY FLEXIBLE - adjust based on dataset size:
                    - Small datasets: use 100 samples
                    - Standard benchmarking: use 1000 samples  
                    - Large-scale testing: use any value you need
        seed: Random seed for dataset sampling (default 42).
              Use same seed when comparing model versions to ensure
              fair comparison on identical samples.
        output_dir: Directory to save results
        
    Returns:
        Dict with energy measurements, normalized to "per 1000 queries"
        
    Raises:
        ValueError: If hardware is not supported or dataset is empty
        
    Examples:
        >>> # Small dataset - use 100 samples
        >>> results = measure_energy(
        >>>     inference_fn=my_fn,
        >>>     dataset=small_dataset,
        >>>     model_name="model-v1",
        >>>     task_name="classification",
        >>>     hardware="T4",
        >>>     num_samples=100,  # Adjust to dataset size
        >>>     seed=42
        >>> )
        >>> 
        >>> # Standard benchmarking - 1000 samples
        >>> results = measure_energy(
        >>>     inference_fn=my_fn,
        >>>     dataset=large_dataset,
        >>>     model_name="model-v1",
        >>>     task_name="classification",
        >>>     hardware="T4",
        >>>     num_samples=1000,  # Default
        >>>     seed=42
        >>> )
        >>> 
        >>> # Compare model versions (use same seed!)
        >>> results_v1 = measure_energy(..., model_name="v1", seed=42)
        >>> results_v2 = measure_energy(..., model_name="v2", seed=42)
    """
    
    # Validate input lengths
    validate_input_length(model_name, "model_name")
    validate_input_length(task_name, "task_name")
    validate_input_length(hardware, "hardware")
    validate_input_length(output_dir, "output_dir", max_length=500)  # Allow longer paths for output_dir
    
    # Validate hardware
    if hardware not in SUPPORTED_HARDWARE:
        raise ValueError(f"Hardware '{hardware}' not supported. Available: {list(SUPPORTED_HARDWARE.keys())}")
    
    # Validate num_samples
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    
    if num_samples > 1000000:  # Reasonable upper limit
        raise ValueError(f"num_samples ({num_samples}) exceeds maximum of 1,000,000")
    
    # Use reproducible sampling
    samples = sample_dataset(dataset, num_samples=num_samples, seed=seed)
    actual_samples = len(samples)
    
    print(f"Processing {actual_samples} samples for model '{model_name}' on {hardware}")
    
    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name=f"{model_name}_{task_name}",
        pue=PUE,
        measure_power_secs=1
    )
    
    # Start tracking
    start_time = time.time()
    tracker.start()
    
    try:
        # Process samples
        for i, sample in enumerate(samples):
            # Run inference
            inference_fn(sample)
            
            # Print progress every 100 samples
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples...")
    
    finally:
        # Stop tracking
        tracker.stop()
        end_time = time.time()
    
    # Extract results
    duration_seconds = end_time - start_time
    
    # Get energy from tracker (CO2 calculation removed as it requires carbon intensity configuration)
    try:
        # Try to get energy from the tracker's final emissions data
        if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data:
            emissions_data = tracker.final_emissions_data
            energy_kwh = getattr(emissions_data, 'energy_consumed', 0.0)
        else:
            # Fallback to direct attributes
            energy_kwh = getattr(tracker, 'energy_consumed', 0.0)
    except (AttributeError, KeyError) as e:
        # Expected errors from tracker - log warning but continue
        warnings.warn(f"Could not extract energy data from tracker: {e}. Using default values.")
        energy_kwh = 0.0
    except Exception as e:
        # Unexpected errors - re-raise with context
        raise RuntimeError(
            f"Unexpected error extracting energy data from tracker: {e}"
        ) from e
    
    # Calculate normalized metrics (ALWAYS normalize to per 1000 queries)
    kwh_per_1000_queries = (energy_kwh / actual_samples) * 1000 if actual_samples > 0 else 0.0
    
    # Create results
    results = {
        "model_name": model_name,
        "task_name": task_name,
        "hardware": hardware,
        "timestamp": datetime.now().isoformat(),
        "num_samples": actual_samples,  # Actual samples processed
        "seed": seed,  # Record the seed used
        "energy_kwh": round(energy_kwh, 6),
        "duration_seconds": round(duration_seconds, 2),
        "kwh_per_1000_queries": round(kwh_per_1000_queries, 6)
    }
    
    # Save results
    save_results(results, output_dir)
    
    print(f"✅ Measurement complete!")
    print(f"   Energy: {energy_kwh:.4f} kWh")
    print(f"   Duration: {duration_seconds:.1f} seconds")
    print(f"   kWh per 1000 queries: {kwh_per_1000_queries:.4f}")
    
    return results


def save_results(results: dict, output_dir: str) -> None:
    """
    Save results to JSON file with path sanitization.
    
    Args:
        results: Results dictionary to save
        output_dir: Base output directory (will be sanitized)
        
    Raises:
        ValueError: If path components are invalid
        OSError: If file cannot be written
    """
    # Sanitize and validate paths
    sanitized_output_dir = sanitize_path_component(output_dir, max_length=500)
    sanitized_task_name = sanitize_path_component(results["task_name"])
    sanitized_model_name = sanitize_path_component(results["model_name"], max_length=200)
    
    # Create safe directory path
    task_dir = sanitize_and_validate_path(sanitized_output_dir, sanitized_task_name, create=True)
    
    # Create filename with timestamp (timestamp is safe as it's generated)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{sanitized_model_name}_{timestamp}.json"
    
    # Validate filename doesn't contain unsafe characters
    if any(char in filename for char in ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']):
        raise ValueError(f"Generated filename contains unsafe characters: {filename}")
    
    filepath = task_dir / filename
    
    # Save JSON
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    except OSError as e:
        raise OSError(f"Failed to write results file {filepath}: {e}") from e
    
    print(f"📁 Results saved to: {filepath}")
