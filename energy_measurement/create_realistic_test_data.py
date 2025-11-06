#!/usr/bin/env python3
"""
Create realistic test data for energy scoring with varied energy consumption values.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import random


def create_realistic_test_data():
    """Create realistic measurement results with varied energy consumption."""
    
    # Realistic energy consumption values (kWh per 1000 queries)
    # Based on typical ML model energy consumption patterns
    realistic_models = [
        {
            "model_name": "efficientnet-b0",
            "task_name": "image-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:30:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0025,
            "co2_kg": 0.001,
            "duration_seconds": 45.2,
            "kwh_per_1000_queries": 0.025  # Very efficient
        },
        {
            "model_name": "mobilenet-v2",
            "task_name": "image-classification", 
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:31:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0042,
            "co2_kg": 0.0017,
            "duration_seconds": 52.8,
            "kwh_per_1000_queries": 0.042
        },
        {
            "model_name": "resnet18",
            "task_name": "image-classification",
            "hardware": "CPU", 
            "timestamp": "2025-01-15T10:32:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0068,
            "co2_kg": 0.0027,
            "duration_seconds": 68.1,
            "kwh_per_1000_queries": 0.068
        },
        {
            "model_name": "resnet50",
            "task_name": "image-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:33:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0125,
            "co2_kg": 0.005,
            "duration_seconds": 125.3,
            "kwh_per_1000_queries": 0.125
        },
        {
            "model_name": "vgg16",
            "task_name": "image-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:34:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0185,
            "co2_kg": 0.0074,
            "duration_seconds": 185.7,
            "kwh_per_1000_queries": 0.185
        },
        {
            "model_name": "inception-v3",
            "task_name": "image-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:35:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0225,
            "co2_kg": 0.009,
            "duration_seconds": 225.0,
            "kwh_per_1000_queries": 0.225
        },
        {
            "model_name": "densenet121",
            "task_name": "image-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:36:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0285,
            "co2_kg": 0.0114,
            "duration_seconds": 285.2,
            "kwh_per_1000_queries": 0.285
        },
        {
            "model_name": "resnet101",
            "task_name": "image-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:37:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0350,
            "co2_kg": 0.014,
            "duration_seconds": 350.5,
            "kwh_per_1000_queries": 0.350
        },
        {
            "model_name": "vgg19",
            "task_name": "image-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:38:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0425,
            "co2_kg": 0.017,
            "duration_seconds": 425.8,
            "kwh_per_1000_queries": 0.425
        },
        {
            "model_name": "resnet152",
            "task_name": "image-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:39:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0525,
            "co2_kg": 0.021,
            "duration_seconds": 525.0,
            "kwh_per_1000_queries": 0.525
        }
    ]
    
    # Create directory structure
    results_dir = Path("realistic_results")
    task_dir = results_dir / "image-classification" / "CPU"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each result to a JSON file
    for result in realistic_models:
        timestamp_str = result['timestamp'].replace(':', '-').replace('T', '_')
        filename = f"{result['model_name']}_{timestamp_str}.json"
        filepath = task_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(f"✅ Created {len(realistic_models)} realistic measurement results in {task_dir}")
    return results_dir


def create_gpu_test_data():
    """Create test data for GPU hardware."""
    
    # GPU models with different energy consumption patterns
    gpu_models = [
        {
            "model_name": "efficientnet-b0-gpu",
            "task_name": "image-classification",
            "hardware": "T4",
            "timestamp": "2025-01-15T11:30:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0015,  # GPU is more efficient
            "co2_kg": 0.0006,
            "duration_seconds": 12.5,
            "kwh_per_1000_queries": 0.015
        },
        {
            "model_name": "resnet50-gpu",
            "task_name": "image-classification",
            "hardware": "T4",
            "timestamp": "2025-01-15T11:31:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0035,
            "co2_kg": 0.0014,
            "duration_seconds": 18.2,
            "kwh_per_1000_queries": 0.035
        },
        {
            "model_name": "vgg16-gpu",
            "task_name": "image-classification",
            "hardware": "T4",
            "timestamp": "2025-01-15T11:32:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0055,
            "co2_kg": 0.0022,
            "duration_seconds": 22.8,
            "kwh_per_1000_queries": 0.055
        },
        {
            "model_name": "inception-v3-gpu",
            "task_name": "image-classification",
            "hardware": "T4",
            "timestamp": "2025-01-15T11:33:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0085,
            "co2_kg": 0.0034,
            "duration_seconds": 28.5,
            "kwh_per_1000_queries": 0.085
        },
        {
            "model_name": "resnet152-gpu",
            "task_name": "image-classification",
            "hardware": "T4",
            "timestamp": "2025-01-15T11:34:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.0125,
            "co2_kg": 0.005,
            "duration_seconds": 35.2,
            "kwh_per_1000_queries": 0.125
        }
    ]
    
    # Create directory structure
    results_dir = Path("realistic_results")
    task_dir = results_dir / "image-classification" / "T4"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each result to a JSON file
    for result in gpu_models:
        timestamp_str = result['timestamp'].replace(':', '-').replace('T', '_')
        filename = f"{result['model_name']}_{timestamp_str}.json"
        filepath = task_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(f"✅ Created {len(gpu_models)} GPU measurement results in {task_dir}")
    return results_dir


def main():
    """Create realistic test data."""
    print("🎯 Creating Realistic Test Data for Energy Scoring")
    print("=" * 55)
    
    # Clean up any existing test data
    if Path("realistic_results").exists():
        shutil.rmtree("realistic_results")
    
    # Create CPU test data
    print("\n📊 Creating CPU Test Data...")
    create_realistic_test_data()
    
    # Create GPU test data  
    print("\n🎮 Creating GPU Test Data...")
    create_gpu_test_data()
    
    print("\n✅ Realistic test data created successfully!")
    print("\n📁 Directory structure:")
    print("realistic_results/")
    print("├── image-classification/")
    print("│   ├── CPU/ (10 models)")
    print("│   └── T4/ (5 models)")
    
    print("\n🧪 Test the scoring function:")
    print("python -c \"from calculate_scores import calculate_scores, print_scores; scores = calculate_scores('image-classification', 'CPU', 'realistic_results'); print_scores(scores)\"")


if __name__ == "__main__":
    main()
