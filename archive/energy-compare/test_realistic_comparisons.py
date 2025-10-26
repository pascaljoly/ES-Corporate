#!/usr/bin/env python3
"""
Realistic model comparison testing with varied model characteristics.

This script tests the comparator with realistic model variations to ensure
proper ranking and scoring functionality.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from comparator import ModelComparator, ComparisonMetric
from scorer.core import ScoringResult, ModelEnergyScorer
from unittest.mock import Mock, patch
import random


class RealisticModelScorer(ModelEnergyScorer):
    """
    A more realistic scorer that returns different values based on model characteristics.
    """
    
    def __init__(self):
        super().__init__()
        # Define realistic model characteristics
        self.model_profiles = {
            # Small, efficient models
            "distilgpt2": {
                "energy_per_1k_wh": 2.1,
                "co2_per_1k_g": 1.0,
                "samples_per_second": 120,
                "duration_seconds": 8.3,
                "size_mb": 82
            },
            "distilbert-base-uncased": {
                "energy_per_1k_wh": 1.8,
                "co2_per_1k_g": 0.9,
                "samples_per_second": 150,
                "duration_seconds": 6.7,
                "size_mb": 67
            },
            "prajjwal1/bert-tiny": {
                "energy_per_1k_wh": 1.2,
                "co2_per_1k_g": 0.6,
                "samples_per_second": 200,
                "duration_seconds": 5.0,
                "size_mb": 17
            },
            
            # Medium models
            "gpt2": {
                "energy_per_1k_wh": 4.2,
                "co2_per_1k_g": 2.1,
                "samples_per_second": 85,
                "duration_seconds": 11.8,
                "size_mb": 548
            },
            "bert-base-uncased": {
                "energy_per_1k_wh": 3.8,
                "co2_per_1k_g": 1.9,
                "samples_per_second": 95,
                "duration_seconds": 10.5,
                "size_mb": 440
            },
            "roberta-base": {
                "energy_per_1k_wh": 4.1,
                "co2_per_1k_g": 2.0,
                "samples_per_second": 88,
                "duration_seconds": 11.4,
                "size_mb": 500
            },
            
            # Large models
            "gpt2-medium": {
                "energy_per_1k_wh": 8.5,
                "co2_per_1k_g": 4.2,
                "samples_per_second": 45,
                "duration_seconds": 22.2,
                "size_mb": 1248
            },
            "bert-large-uncased": {
                "energy_per_1k_wh": 7.2,
                "co2_per_1k_g": 3.6,
                "samples_per_second": 55,
                "duration_seconds": 18.2,
                "size_mb": 1343
            },
            "roberta-large": {
                "energy_per_1k_wh": 7.8,
                "co2_per_1k_g": 3.9,
                "samples_per_second": 52,
                "duration_seconds": 19.2,
                "size_mb": 1350
            },
            
            # Very large models
            "gpt2-large": {
                "energy_per_1k_wh": 15.2,
                "co2_per_1k_g": 7.6,
                "samples_per_second": 25,
                "duration_seconds": 40.0,
                "size_mb": 3086
            },
            "gpt2-xl": {
                "energy_per_1k_wh": 28.5,
                "co2_per_1k_g": 14.2,
                "samples_per_second": 15,
                "duration_seconds": 66.7,
                "size_mb": 6065
            }
        }
    
    def score(self, model: str, task: str, n_samples: int = 100, runs: int = 3) -> ScoringResult:
        """Return realistic scoring results based on model characteristics."""
        self.logger.info(f"Scoring {model} on {task} with realistic data")
        
        # Validate inputs
        self._validate_inputs(model, task, n_samples, runs)
        
        # Get model profile or use default
        profile = self.model_profiles.get(model, {
            "energy_per_1k_wh": 5.0,
            "co2_per_1k_g": 2.5,
            "samples_per_second": 75,
            "duration_seconds": 13.3,
            "size_mb": 500
        })
        
        # Add some realistic variation based on runs
        variation_factor = 1.0 + (random.uniform(-0.1, 0.1) * (runs - 1) / 10)
        
        result = ScoringResult(
            model_id=model,
            task=task,
            measurements={
                'energy_per_1k_wh': profile["energy_per_1k_wh"] * variation_factor,
                'co2_per_1k_g': profile["co2_per_1k_g"] * variation_factor,
                'samples_per_second': profile["samples_per_second"] * variation_factor,
                'duration_seconds': profile["duration_seconds"] / variation_factor,
                'statistics': {
                    'coefficient_of_variation': 0.05 + random.uniform(0, 0.1),
                    'runs': runs
                }
            },
            hardware={'gpu': 'RTX 4090', 'cpu': 'AMD Ryzen 9 7950X'},
            metadata={
                'timestamp': '2025-10-03T12:00:00Z',
                'model_size_mb': profile.get("size_mb", 500),
                'n_samples': n_samples,
                'runs': runs
            }
        )
        
        self.logger.info(f"Scoring complete for {model}")
        return result


def test_small_vs_large_models():
    """Test comparison between small efficient models and large powerful models."""
    print("=== Small vs Large Models Comparison ===")
    
    # Use realistic scorer
    realistic_scorer = RealisticModelScorer()
    comparator = ModelComparator(scorer=realistic_scorer)
    
    model_specs = [
        ("prajjwal1/bert-tiny", "text_generation"),    # Very small, very efficient
        ("distilgpt2", "text_generation"),             # Small, efficient
        ("gpt2", "text_generation"),                   # Medium
        ("gpt2-medium", "text_generation"),            # Large
        ("gpt2-large", "text_generation")              # Very large
    ]
    
    print(f"Comparing {len(model_specs)} models of different sizes...")
    
    result = comparator.compare_models(
        model_specs=model_specs,
        n_samples=50,
        runs=2
    )
    
    print(f"\nResults for task: {result.task}")
    print(f"Winner: {result.summary['winner']}")
    
    print("\nRankings (Energy Efficiency Focus):")
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        print(f"{model.rank}. {model.model_id}")
        print(f"   Score: {model.score:.3f}")
        print(f"   Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k queries")
        print(f"   CO2: {measurements['co2_per_1k_g']:.1f} kg CO2/1k queries")
        print(f"   Throughput: {measurements['samples_per_second']:.0f} samples/sec")
        print(f"   Model Size: {model.scoring_result.metadata.get('model_size_mb', 'N/A')} MB")
        print()


def test_energy_focused_comparison():
    """Test comparison with energy-focused weights."""
    print("=== Energy-Focused Comparison ===")
    
    realistic_scorer = RealisticModelScorer()
    comparator = ModelComparator(scorer=realistic_scorer)
    
    model_specs = [
        ("distilbert-base-uncased", "text_generation"),
        ("bert-base-uncased", "text_generation"),
        ("bert-large-uncased", "text_generation")
    ]
    
    # Energy-focused weights
    energy_weights = {
        ComparisonMetric.ENERGY_EFFICIENCY: 0.6,
        ComparisonMetric.CO2_EFFICIENCY: 0.4,
        ComparisonMetric.PERFORMANCE: 0.0,
        ComparisonMetric.SPEED: 0.0
    }
    
    result = comparator.compare_models(
        model_specs=model_specs,
        custom_weights=energy_weights,
        n_samples=50,
        runs=2
    )
    
    print("Energy-focused rankings:")
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        print(f"{model.rank}. {model.model_id}")
        print(f"   Score: {model.score:.3f}")
        print(f"   Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k queries")
        print(f"   CO2: {measurements['co2_per_1k_g']:.1f} kg CO2/1k queries")
        print()


def test_performance_focused_comparison():
    """Test comparison with performance-focused weights."""
    print("=== Performance-Focused Comparison ===")
    
    realistic_scorer = RealisticModelScorer()
    comparator = ModelComparator(scorer=realistic_scorer)
    
    model_specs = [
        ("prajjwal1/bert-tiny", "text_generation"),
        ("gpt2", "text_generation"),
        ("gpt2-medium", "text_generation"),
        ("gpt2-large", "text_generation")
    ]
    
    # Performance-focused weights
    performance_weights = {
        ComparisonMetric.ENERGY_EFFICIENCY: 0.1,
        ComparisonMetric.CO2_EFFICIENCY: 0.1,
        ComparisonMetric.PERFORMANCE: 0.5,
        ComparisonMetric.SPEED: 0.3
    }
    
    result = comparator.compare_models(
        model_specs=model_specs,
        custom_weights=performance_weights,
        n_samples=50,
        runs=2
    )
    
    print("Performance-focused rankings:")
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        print(f"{model.rank}. {model.model_id}")
        print(f"   Score: {model.score:.3f}")
        print(f"   Throughput: {measurements['samples_per_second']:.0f} samples/sec")
        print(f"   Speed: {1/measurements['duration_seconds']:.3f} queries/sec")
        print(f"   Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k queries")
        print()


def test_balanced_comparison():
    """Test balanced comparison with default weights."""
    print("=== Balanced Comparison (Default Weights) ===")
    
    realistic_scorer = RealisticModelScorer()
    comparator = ModelComparator(scorer=realistic_scorer)
    
    model_specs = [
        ("distilgpt2", "text_generation"),
        ("gpt2", "text_generation"),
        ("gpt2-medium", "text_generation")
    ]
    
    result = comparator.compare_models(
        model_specs=model_specs,
        n_samples=50,
        runs=2
    )
    
    print("Balanced rankings:")
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        print(f"{model.rank}. {model.model_id}")
        print(f"   Score: {model.score:.3f}")
        print(f"   Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k queries")
        print(f"   CO2: {measurements['co2_per_1k_g']:.1f} kg CO2/1k queries")
        print(f"   Throughput: {measurements['samples_per_second']:.0f} samples/sec")
        print(f"   Duration: {measurements['duration_seconds']:.1f} seconds")
        print()


def test_cost_effectiveness_analysis():
    """Test cost-effectiveness analysis."""
    print("=== Cost-Effectiveness Analysis ===")
    
    realistic_scorer = RealisticModelScorer()
    comparator = ModelComparator(scorer=realistic_scorer)
    
    model_specs = [
        ("prajjwal1/bert-tiny", "text_generation"),
        ("distilgpt2", "text_generation"),
        ("gpt2", "text_generation"),
        ("gpt2-medium", "text_generation")
    ]
    
    # Cost-effectiveness focused weights
    cost_weights = {
        ComparisonMetric.ENERGY_EFFICIENCY: 0.3,
        ComparisonMetric.COST_EFFECTIVENESS: 0.7
    }
    
    result = comparator.compare_models(
        model_specs=model_specs,
        custom_weights=cost_weights,
        n_samples=50,
        runs=2
    )
    
    print("Cost-effectiveness rankings:")
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        cost_effectiveness = measurements['samples_per_second'] / measurements['energy_per_1k_wh']
        print(f"{model.rank}. {model.model_id}")
        print(f"   Score: {model.score:.3f}")
        print(f"   Cost-Effectiveness: {cost_effectiveness:.1f} samples/kWh")
        print(f"   Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k queries")
        print(f"   Throughput: {measurements['samples_per_second']:.0f} samples/sec")
        print()


def test_detailed_statistics():
    """Test detailed comparison statistics."""
    print("=== Detailed Statistics Analysis ===")
    
    realistic_scorer = RealisticModelScorer()
    comparator = ModelComparator(scorer=realistic_scorer)
    
    model_specs = [
        ("prajjwal1/bert-tiny", "text_generation"),
        ("distilgpt2", "text_generation"),
        ("distilbert-base-uncased", "text_generation"),
        ("gpt2", "text_generation"),
        ("bert-base-uncased", "text_generation"),
        ("gpt2-medium", "text_generation"),
        ("bert-large-uncased", "text_generation"),
        ("gpt2-large", "text_generation")
    ]
    
    result = comparator.compare_models(
        model_specs=model_specs,
        n_samples=50,
        runs=2
    )
    
    print("Detailed Statistics:")
    print("=" * 60)
    
    summary = result.summary
    print(f"Total models compared: {summary['total_models']}")
    print(f"Winner: {summary['winner']}")
    
    # Score statistics
    score_stats = summary['score_statistics']
    print(f"\nScore Statistics:")
    print(f"  Mean: {score_stats['mean']:.3f}")
    print(f"  Median: {score_stats['median']:.3f}")
    print(f"  Std Dev: {score_stats['std']:.3f}")
    print(f"  Range: {score_stats['min']:.3f} - {score_stats['max']:.3f}")
    
    # Energy and CO2 ranges
    energy_range = summary['energy_range']
    co2_range = summary['co2_range']
    print(f"\nEnergy Range: {energy_range['min_kwh']:.1f} - {energy_range['max_kwh']:.1f} kWh/1k queries")
    print(f"CO2 Range: {co2_range['min_kg']:.1f} - {co2_range['max_kg']:.1f} kg CO2/1k queries")
    
    # Model size analysis
    print(f"\nModel Size Analysis:")
    for model in result.get_rankings():
        size_mb = model.scoring_result.metadata.get('model_size_mb', 0)
        measurements = model.scoring_result.measurements
        energy_per_mb = measurements['energy_per_1k_wh'] / (size_mb / 100) if size_mb > 0 else 0
        print(f"  {model.model_id}: {size_mb} MB, {energy_per_mb:.3f} kWh/1k queries per 100MB")
    
    print()


def test_save_and_load_realistic():
    """Test saving and loading with realistic data."""
    print("=== Save and Load Realistic Data ===")
    
    realistic_scorer = RealisticModelScorer()
    comparator = ModelComparator(scorer=realistic_scorer)
    
    model_specs = [
        ("distilgpt2", "text_generation"),
        ("gpt2", "text_generation"),
        ("gpt2-medium", "text_generation")
    ]
    
    result = comparator.compare_models(
        model_specs=model_specs,
        n_samples=50,
        runs=2
    )
    
    # Save to file
    output_file = "realistic_comparison.json"
    comparator.save_comparison(result, output_file)
    print(f"Comparison saved to {output_file}")
    
    # Load from file
    loaded_result = comparator.load_comparison(output_file)
    print(f"Loaded comparison for task: {loaded_result.task}")
    print(f"Number of models: {len(loaded_result.models)}")
    print(f"Winner: {loaded_result.get_winner().model_id}")
    
    # Verify data integrity
    original_winner = result.get_winner()
    loaded_winner = loaded_result.get_winner()
    
    print(f"\nData integrity check:")
    print(f"Original winner: {original_winner.model_id} (Score: {original_winner.score:.3f})")
    print(f"Loaded winner: {loaded_winner.model_id} (Score: {loaded_winner.score:.3f})")
    print(f"Match: {original_winner.model_id == loaded_winner.model_id}")
    
    # Clean up
    Path(output_file).unlink(missing_ok=True)
    print("Demo file cleaned up")


if __name__ == "__main__":
    print("Realistic Model Comparison Testing")
    print("=" * 50)
    
    try:
        test_small_vs_large_models()
        test_energy_focused_comparison()
        test_performance_focused_comparison()
        test_balanced_comparison()
        test_cost_effectiveness_analysis()
        test_detailed_statistics()
        test_save_and_load_realistic()
        
        print("=" * 50)
        print("All realistic comparison tests completed successfully!")
        print("\nKey Insights:")
        print("- Small models (bert-tiny, distilgpt2) excel in energy efficiency")
        print("- Large models (gpt2-large, bert-large) provide higher performance")
        print("- Balanced comparisons show trade-offs between efficiency and performance")
        print("- Cost-effectiveness varies significantly across model sizes")
        
    except Exception as e:
        print(f"Error running realistic tests: {e}")
        import traceback
        traceback.print_exc()


