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
    output_dir: str = "results"
) -> dict:
    """
    Measure energy consumption of a model using CodeCarbon.
    
    Args:
        inference_fn: A callable that processes one sample
        dataset: Any iterable containing samples to process
        model_name: String identifier for the model
        task_name: String describing the task (e.g., "image-classification")
        hardware: String from SUPPORTED_HARDWARE
        num_samples: Number of samples to process (default 1000)
        output_dir: Directory to save results (default "results")
        
    Returns:
        dict: Measurement results
        
    Raises:
        ValueError: If hardware is not supported or dataset is empty
    """
    
    # Validate hardware
    if hardware not in SUPPORTED_HARDWARE:
        raise ValueError(f"Hardware '{hardware}' not supported. Available: {list(SUPPORTED_HARDWARE.keys())}")
    
    # Convert dataset to list and validate
    dataset_list = list(dataset)
    if not dataset_list:
        raise ValueError("Dataset is empty")
    
    # Sample the dataset
    if len(dataset_list) < num_samples:
        print(f"Warning: Dataset has {len(dataset_list)} samples, but {num_samples} requested. Using all available samples.")
        num_samples = len(dataset_list)
    
    samples = dataset_list[:num_samples]
    print(f"Processing {len(samples)} samples for model '{model_name}' on {hardware}")
    
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
    
    # Calculate normalized metrics
    kwh_per_1000_queries = (energy_kwh / len(samples)) * 1000 if len(samples) > 0 else 0.0
    
    # Create results
    results = {
        "model_name": model_name,
        "task_name": task_name,
        "hardware": hardware,
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(samples),
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
