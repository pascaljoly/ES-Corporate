#!/usr/bin/env python3
"""
Energy scoring function for measured models.

This module provides functionality to calculate star ratings for energy efficiency
based on measurement results from the energy measurement script.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def calculate_scores(task_name: str, hardware: str, results_dir: str = "results") -> dict:
    """
    Calculate energy star ratings for all models in a task + hardware category.
    
    This function reads all measurement results for a specific task and hardware
    combination, then assigns star ratings based on energy efficiency quintiles.
    Models with lower energy consumption get higher star ratings.
    
    Args:
        task_name: Task name (e.g., "image-classification", "text-generation")
        hardware: Hardware type (e.g., "T4", "CPU", "A100")
        results_dir: Directory containing measurement results (default: "results")
        
    Returns:
        Dictionary containing:
        {
            "task_name": "image-classification",
            "hardware": "T4", 
            "num_models": 10,
            "models": [
                {
                    "model_name": "resnet50",
                    "energy_kwh": 0.05,
                    "kwh_per_1000_queries": 0.05,
                    "star_rating": 5,
                    "percentile": 10,  # Lower is better
                    "timestamp": "2025-01-15T10:30:45.123456",
                    "duration_seconds": 125.3,
                    "co2_kg": 0.018
                },
                ...
            ],
            "energy_range": {
                "min": 0.05,
                "max": 0.25,
                "median": 0.12
            }
        }
        
    Raises:
        ValueError: If no valid measurement results are found
        FileNotFoundError: If the results directory doesn't exist
        
    Example:
        >>> scores = calculate_scores("image-classification", "CPU")
        >>> print(f"Found {scores['num_models']} models")
        >>> for model in scores['models']:
        ...     stars = "⭐" * model['star_rating'] + "☆" * (5 - model['star_rating'])
        ...     print(f"{stars} {model['model_name']}: {model['kwh_per_1000_queries']:.4f} kWh")
    """
    
    # Construct path to results directory
    results_path = Path(results_dir) / task_name / hardware
    
    # Check if directory exists
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    
    # Find all JSON files in the directory
    json_files = list(results_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {results_path}")
    
    # Load and parse all measurement results
    models = []
    invalid_files = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Validate required fields
            required_fields = ['model_name', 'kwh_per_1000_queries', 'energy_kwh', 'timestamp']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                warnings.warn(f"Skipping {json_file.name}: missing fields {missing_fields}")
                invalid_files.append(json_file.name)
                continue
            
            # Extract model data
            model_data = {
                'model_name': data['model_name'],
                'energy_kwh': data['energy_kwh'],
                'kwh_per_1000_queries': data['kwh_per_1000_queries'],
                'timestamp': data['timestamp'],
                'duration_seconds': data.get('duration_seconds', 0.0),
                'co2_kg': data.get('co2_kg', 0.0),
                'num_samples': data.get('num_samples', 0)
            }
            
            models.append(model_data)
            
        except (json.JSONDecodeError, KeyError) as e:
            warnings.warn(f"Skipping invalid JSON file {json_file.name}: {e}")
            invalid_files.append(json_file.name)
            continue
    
    # Check if we have any valid results
    if not models:
        raise ValueError(f"No valid measurement results found in {results_path}")
    
    # Warn about invalid files
    if invalid_files:
        warnings.warn(f"Skipped {len(invalid_files)} invalid files: {invalid_files}")
    
    # Sort models by energy efficiency (lower is better)
    models.sort(key=lambda x: x['kwh_per_1000_queries'])
    
    # Calculate star ratings using quintiles
    num_models = len(models)
    
    # Handle edge cases for small numbers of models
    if num_models < 5:
        warnings.warn(f"Only {num_models} models found. Star distribution may not be optimal.")
    
    # Calculate quintile boundaries
    quintile_size = max(1, num_models // 5)  # At least 1 model per quintile
    
    # Assign star ratings
    for i, model in enumerate(models):
        # Calculate which quintile this model falls into (0-4)
        quintile = min(4, i // quintile_size)
        
        # Convert quintile to star rating (5 stars for best, 1 star for worst)
        model['star_rating'] = 5 - quintile
        
        # Calculate percentile (0-100, lower is better)
        model['percentile'] = int((i / (num_models - 1)) * 100) if num_models > 1 else 0
    
    # Calculate energy range statistics
    energy_values = [model['kwh_per_1000_queries'] for model in models]
    energy_range = {
        'min': min(energy_values),
        'max': max(energy_values),
        'median': sorted(energy_values)[len(energy_values) // 2]
    }
    
    # Sort models by star rating (descending) then by energy (ascending)
    models.sort(key=lambda x: (-x['star_rating'], x['kwh_per_1000_queries']))
    
    # Return results
    return {
        'task_name': task_name,
        'hardware': hardware,
        'num_models': num_models,
        'models': models,
        'energy_range': energy_range,
        'invalid_files': invalid_files
    }


def print_scores(scores: dict) -> None:
    """
    Print formatted energy scores to console.
    
    Args:
        scores: Results from calculate_scores()
    """
    print(f"\n=== Energy Scores for {scores['task_name']} on {scores['hardware']} ===")
    print(f"Models evaluated: {scores['num_models']}")
    print(f"\nEnergy range: {scores['energy_range']['min']:.4f} - {scores['energy_range']['max']:.4f} kWh")
    print(f"Median: {scores['energy_range']['median']:.4f} kWh")
    
    if scores['invalid_files']:
        print(f"\n⚠️  Skipped {len(scores['invalid_files'])} invalid files")
    
    print("\nModel Rankings:")
    for model in scores['models']:
        stars = "⭐" * model['star_rating'] + "☆" * (5 - model['star_rating'])
        print(f"{stars} {model['model_name']}: {model['kwh_per_1000_queries']:.4f} kWh (percentile: {model['percentile']})")


if __name__ == "__main__":
    # Example usage
    try:
        scores = calculate_scores("test-classification", "CPU")
        print_scores(scores)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
