#!/usr/bin/env python3
"""
Example usage of the Energy Compare framework.

This script demonstrates how to use the ModelComparator to compare
multiple ML models based on their energy efficiency and performance.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from comparator import ModelComparator, ComparisonMetric
from scorer.core import ModelEnergyScorer, ScoringResult


def example_basic_comparison():
    """Example of basic model comparison"""
    print("=== Basic Model Comparison Example ===")
    
    # Initialize the comparator
    comparator = ModelComparator()
    
    # Define models to compare (model_id, task)
    model_specs = [
        ("gpt2", "text_generation"),
        ("gpt2-medium", "text_generation"),
        ("distilgpt2", "text_generation")
    ]
    
    print(f"Comparing {len(model_specs)} models...")
    
    # Compare models (this will run actual energy measurements)
    try:
        result = comparator.compare_models(
            model_specs=model_specs,
            n_samples=50,  # Small sample for demo
            runs=2         # Few runs for demo
        )
        
        # Display results
        print(f"\nComparison Results for task: {result.task}")
        print(f"Total models compared: {result.summary['total_models']}")
        print(f"Winner: {result.summary['winner']}")
        
        print("\nRankings:")
        for i, model in enumerate(result.get_rankings(), 1):
            print(f"{i}. {model.model_id}")
            print(f"   Score: {model.score:.3f}")
            print(f"   Energy: {model.scoring_result.measurements.get('energy_per_1k_wh', 'N/A')} kWh/1k queries")
            print(f"   CO2: {model.scoring_result.measurements.get('co2_per_1k_g', 'N/A')} kg CO2/1k queries")
            print()
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        print("This is expected in demo mode - the scorer returns mock data")


def example_custom_weights():
    """Example of comparison with custom metric weights"""
    print("\n=== Custom Weights Comparison Example ===")
    
    # Create mock scoring results for demonstration
    mock_results = [
        ScoringResult(
            model_id="efficient_model",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 2.0,  # Very efficient
                'co2_per_1k_g': 1.0,
                'samples_per_second': 80,  # Slower
                'duration_seconds': 12.5
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        ),
        ScoringResult(
            model_id="fast_model",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 5.0,  # Less efficient
                'co2_per_1k_g': 2.5,
                'samples_per_second': 150,  # Faster
                'duration_seconds': 6.7
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        ),
        ScoringResult(
            model_id="balanced_model",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 3.5,  # Moderate efficiency
                'co2_per_1k_g': 1.75,
                'samples_per_second': 120,  # Moderate speed
                'duration_seconds': 8.3
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        )
    ]
    
    comparator = ModelComparator()
    
    # Compare with default weights (balanced)
    print("Comparison with default weights:")
    result_default = comparator.compare_models_from_results(mock_results)
    print(f"Winner: {result_default.get_winner().model_id}")
    
    # Compare with energy-focused weights
    energy_focused_weights = {
        ComparisonMetric.ENERGY_EFFICIENCY: 0.7,
        ComparisonMetric.CO2_EFFICIENCY: 0.3,
        ComparisonMetric.PERFORMANCE: 0.0,
        ComparisonMetric.SPEED: 0.0
    }
    
    print("\nComparison with energy-focused weights:")
    result_energy = comparator.compare_models_from_results(
        mock_results,
        custom_weights=energy_focused_weights
    )
    print(f"Winner: {result_energy.get_winner().model_id}")
    
    # Compare with performance-focused weights
    performance_focused_weights = {
        ComparisonMetric.ENERGY_EFFICIENCY: 0.2,
        ComparisonMetric.CO2_EFFICIENCY: 0.1,
        ComparisonMetric.PERFORMANCE: 0.5,
        ComparisonMetric.SPEED: 0.2
    }
    
    print("\nComparison with performance-focused weights:")
    result_performance = comparator.compare_models_from_results(
        mock_results,
        custom_weights=performance_focused_weights
    )
    print(f"Winner: {result_performance.get_winner().model_id}")


def example_specific_metrics():
    """Example of comparison using specific metrics only"""
    print("\n=== Specific Metrics Comparison Example ===")
    
    # Create mock results
    mock_results = [
        ScoringResult(
            model_id="model_a",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 3.0,
                'co2_per_1k_g': 1.5,
                'samples_per_second': 100,
                'duration_seconds': 10.0
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        ),
        ScoringResult(
            model_id="model_b",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 4.0,
                'co2_per_1k_g': 2.0,
                'samples_per_second': 120,
                'duration_seconds': 8.3
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        )
    ]
    
    comparator = ModelComparator()
    
    # Compare only on energy efficiency
    energy_only_metrics = [ComparisonMetric.ENERGY_EFFICIENCY]
    result_energy = comparator.compare_models_from_results(
        mock_results,
        metrics=energy_only_metrics
    )
    
    print("Energy efficiency only comparison:")
    print(f"Winner: {result_energy.get_winner().model_id}")
    print(f"Metrics used: {[m.value for m in result_energy.comparison_metrics]}")
    
    # Compare only on performance
    performance_only_metrics = [ComparisonMetric.PERFORMANCE]
    result_performance = comparator.compare_models_from_results(
        mock_results,
        metrics=performance_only_metrics
    )
    
    print("\nPerformance only comparison:")
    print(f"Winner: {result_performance.get_winner().model_id}")
    print(f"Metrics used: {[m.value for m in result_performance.comparison_metrics]}")


def example_save_and_load():
    """Example of saving and loading comparison results"""
    print("\n=== Save and Load Example ===")
    
    # Create mock results
    mock_results = [
        ScoringResult(
            model_id="saved_model",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 3.5,
                'co2_per_1k_g': 1.75,
                'samples_per_second': 110,
                'duration_seconds': 9.1
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        )
    ]
    
    comparator = ModelComparator()
    result = comparator.compare_models_from_results(mock_results)
    
    # Save to file
    output_file = "comparison_results.json"
    comparator.save_comparison(result, output_file)
    print(f"Comparison saved to {output_file}")
    
    # Load from file
    loaded_result = comparator.load_comparison(output_file)
    print(f"Loaded comparison for task: {loaded_result.task}")
    print(f"Number of models: {len(loaded_result.models)}")
    print(f"Winner: {loaded_result.get_winner().model_id}")
    
    # Clean up
    Path(output_file).unlink(missing_ok=True)
    print("Demo file cleaned up")


def example_detailed_analysis():
    """Example of detailed comparison analysis"""
    print("\n=== Detailed Analysis Example ===")
    
    # Create comprehensive mock results
    mock_results = [
        ScoringResult(
            model_id="bert-base",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 4.2,
                'co2_per_1k_g': 2.1,
                'samples_per_second': 85,
                'duration_seconds': 11.8
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        ),
        ScoringResult(
            model_id="distilbert",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 2.8,
                'co2_per_1k_g': 1.4,
                'samples_per_second': 95,
                'duration_seconds': 10.5
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        ),
        ScoringResult(
            model_id="roberta-base",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 5.1,
                'co2_per_1k_g': 2.55,
                'samples_per_second': 75,
                'duration_seconds': 13.3
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        )
    ]
    
    comparator = ModelComparator()
    result = comparator.compare_models_from_results(mock_results)
    
    print("Detailed Comparison Analysis:")
    print("=" * 50)
    
    # Summary statistics
    summary = result.summary
    print(f"Task: {result.task}")
    print(f"Total models: {summary['total_models']}")
    print(f"Metrics used: {', '.join(summary['metrics_used'])}")
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
    
    # Detailed model breakdown
    print(f"\nModel Rankings:")
    print("-" * 50)
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        print(f"Rank {model.rank}: {model.model_id}")
        print(f"  Composite Score: {model.score:.3f}")
        print(f"  Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k queries")
        print(f"  CO2: {measurements['co2_per_1k_g']:.1f} kg CO2/1k queries")
        print(f"  Throughput: {measurements['samples_per_second']:.0f} samples/sec")
        print(f"  Duration: {measurements['duration_seconds']:.1f} seconds")
        print()


if __name__ == "__main__":
    print("Energy Compare Framework - Example Usage")
    print("=" * 50)
    
    try:
        example_basic_comparison()
        example_custom_weights()
        example_specific_metrics()
        example_save_and_load()
        example_detailed_analysis()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nTo use with real models:")
        print("1. Ensure you have the required dependencies installed")
        print("2. Replace mock data with actual model scoring")
        print("3. Adjust n_samples and runs based on your needs")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("This is expected if dependencies are not installed")
