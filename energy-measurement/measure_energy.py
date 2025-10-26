#!/usr/bin/env python3
"""
Simple Energy Measurement Script

This module provides a simple function to measure energy consumption
of ML models using CodeCarbon for tracking.
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Any
from codecarbon import EmissionsTracker
from sample_dataset import sample_dataset

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
    
    # Validate hardware
    if hardware not in SUPPORTED_HARDWARE:
        raise ValueError(f"Hardware '{hardware}' not supported. Available: {list(SUPPORTED_HARDWARE.keys())}")
    
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
    
    # Get energy and CO2 from tracker
    try:
        # Try to get energy from the tracker's final emissions data
        if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data:
            emissions_data = tracker.final_emissions_data
            energy_kwh = getattr(emissions_data, 'energy_consumed', 0.0)
            co2_kg = getattr(emissions_data, 'emissions', 0.0)
        else:
            # Fallback to direct attributes
            energy_kwh = getattr(tracker, 'energy_consumed', 0.0)
            co2_kg = getattr(tracker, 'emissions', 0.0)
    except Exception as e:
        print(f"Warning: Could not extract energy data: {e}")
        energy_kwh = 0.0
        co2_kg = 0.0
    
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
        "co2_kg": round(co2_kg, 6),
        "duration_seconds": round(duration_seconds, 2),
        "kwh_per_1000_queries": round(kwh_per_1000_queries, 6)
    }
    
    # Save results
    save_results(results, output_dir)
    
    print(f"✅ Measurement complete!")
    print(f"   Energy: {energy_kwh:.4f} kWh")
    print(f"   CO2: {co2_kg:.4f} kg")
    print(f"   Duration: {duration_seconds:.1f} seconds")
    print(f"   kWh per 1000 queries: {kwh_per_1000_queries:.4f}")
    
    return results


def save_results(results: dict, output_dir: str) -> None:
    """Save results to JSON file."""
    
    # Create output directory structure
    task_dir = Path(output_dir) / results["task_name"]
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results['model_name']}_{timestamp}.json"
    filepath = task_dir / filename
    
    # Save JSON
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Results saved to: {filepath}")


if __name__ == "__main__":
    # Example usage
    def dummy_inference(sample):
        """Dummy inference function for testing."""
        time.sleep(0.01)  # Simulate processing
        return sample
    
    # Create dummy dataset
    dummy_dataset = [{"text": f"Sample {i}"} for i in range(100)]
    
    # Measure energy
    results = measure_energy(
        inference_fn=dummy_inference,
        dataset=dummy_dataset,
        model_name="dummy_model",
        task_name="text-classification",
        hardware="CPU",
        num_samples=50
    )
    
    print(f"\nResults: {json.dumps(results, indent=2)}")
