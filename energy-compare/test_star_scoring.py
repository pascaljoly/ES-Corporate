#!/usr/bin/env python3
"""
Test the new 1-5 star scoring system.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from comparator import ModelComparator, ComparisonMetric
from scorer.core import ScoringResult


def test_star_scoring_system():
    """Test that the scoring system now uses 1-5 stars"""
    print("=== Testing 1-5 Star Scoring System ===")
    
    # Create test results with clear differences
    results = [
        ScoringResult(
            model_id="efficient_model",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 1.0,  # Very efficient
                'co2_per_1k_g': 0.5,
                'samples_per_second': 200,  # High performance
                'duration_seconds': 5.0
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        ),
        ScoringResult(
            model_id="balanced_model",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 3.0,  # Moderate efficiency
                'co2_per_1k_g': 1.5,
                'samples_per_second': 100,  # Moderate performance
                'duration_seconds': 10.0
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        ),
        ScoringResult(
            model_id="inefficient_model",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 5.0,  # Less efficient
                'co2_per_1k_g': 2.5,
                'samples_per_second': 50,   # Lower performance
                'duration_seconds': 20.0
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        )
    ]
    
    comparator = ModelComparator()
    
    # Test with default balanced weights
    result = comparator.compare_models_from_results(results)
    
    print("Star Ratings (1-5 scale, 5 = best):")
    print("-" * 50)
    
    for model in result.get_rankings():
        star_rating = ModelComparator.format_star_rating(model.score)
        measurements = model.scoring_result.measurements
        
        print(f"Rank {model.rank}: {model.model_id}")
        print(f"  Overall Score: {star_rating}")
        print(f"  Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k queries")
        print(f"  CO2: {measurements['co2_per_1k_g']:.1f} kg CO2/1k queries")
        print(f"  Throughput: {measurements['samples_per_second']:.0f} samples/sec")
        print()
    
    # Verify all scores are in 1-5 range
    all_scores = [model.score for model in result.models]
    min_score = min(all_scores)
    max_score = max(all_scores)
    
    print(f"Score Range: {min_score:.1f} - {max_score:.1f} stars")
    
    if min_score >= 1.0 and max_score <= 5.0:
        print("✅ PASSED: All scores are in 1-5 star range")
    else:
        print(f"❌ FAILED: Scores outside 1-5 range (min: {min_score}, max: {max_score})")
    
    # Test summary statistics
    summary = result.summary
    print(f"\nSummary Statistics:")
    print(f"  Scoring System: {summary['scoring_system']}")
    print(f"  Winner: {summary['winner']} ({summary['winner_stars']} stars)")
    print(f"  Score Range: {summary['score_statistics']['min']} - {summary['score_statistics']['max']} stars")
    print(f"  Average: {summary['score_statistics']['mean']} stars")


def test_energy_focused_star_ratings():
    """Test star ratings with energy-focused weights"""
    print("\n=== Energy-Focused Star Ratings ===")
    
    results = [
        ScoringResult(
            model_id="very_efficient",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 0.5,  # Very low energy
                'co2_per_1k_g': 0.25,
                'samples_per_second': 80,   # Lower performance
                'duration_seconds': 12.5
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        ),
        ScoringResult(
            model_id="high_performance",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 4.0,  # Higher energy
                'co2_per_1k_g': 2.0,
                'samples_per_second': 200,  # High performance
                'duration_seconds': 5.0
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        )
    ]
    
    comparator = ModelComparator()
    
    # Energy-focused weights
    energy_weights = {
        ComparisonMetric.ENERGY_EFFICIENCY: 0.7,
        ComparisonMetric.CO2_EFFICIENCY: 0.3
    }
    
    result = comparator.compare_models_from_results(
        results,
        custom_weights=energy_weights
    )
    
    print("Energy-focused rankings:")
    for model in result.get_rankings():
        star_rating = ModelComparator.format_star_rating(model.score)
        measurements = model.scoring_result.measurements
        
        print(f"  {model.model_id}: {star_rating}")
        print(f"    Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k queries")
        print(f"    Performance: {measurements['samples_per_second']:.0f} samples/sec")
        print()


def test_performance_focused_star_ratings():
    """Test star ratings with performance-focused weights"""
    print("=== Performance-Focused Star Ratings ===")
    
    results = [
        ScoringResult(
            model_id="very_efficient",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 0.5,  # Very low energy
                'co2_per_1k_g': 0.25,
                'samples_per_second': 80,   # Lower performance
                'duration_seconds': 12.5
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        ),
        ScoringResult(
            model_id="high_performance",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 4.0,  # Higher energy
                'co2_per_1k_g': 2.0,
                'samples_per_second': 200,  # High performance
                'duration_seconds': 5.0
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        )
    ]
    
    comparator = ModelComparator()
    
    # Performance-focused weights
    performance_weights = {
        ComparisonMetric.PERFORMANCE: 0.5,
        ComparisonMetric.SPEED: 0.3,
        ComparisonMetric.ENERGY_EFFICIENCY: 0.2
    }
    
    result = comparator.compare_models_from_results(
        results,
        custom_weights=performance_weights
    )
    
    print("Performance-focused rankings:")
    for model in result.get_rankings():
        star_rating = ModelComparator.format_star_rating(model.score)
        measurements = model.scoring_result.measurements
        
        print(f"  {model.model_id}: {star_rating}")
        print(f"    Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k queries")
        print(f"    Performance: {measurements['samples_per_second']:.0f} samples/sec")
        print()


def test_edge_cases_star_ratings():
    """Test edge cases with star ratings"""
    print("=== Edge Cases with Star Ratings ===")
    
    # Test identical values
    identical_results = [
        ScoringResult(
            model_id="model1",
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
            model_id="model2",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 3.0,  # Identical values
                'co2_per_1k_g': 1.5,
                'samples_per_second': 100,
                'duration_seconds': 10.0
            },
            hardware={'gpu': 'RTX 4090'},
            metadata={'timestamp': '2025-10-03'}
        )
    ]
    
    comparator = ModelComparator()
    result = comparator.compare_models_from_results(identical_results)
    
    print("Identical values test:")
    for model in result.get_rankings():
        star_rating = ModelComparator.format_star_rating(model.score)
        print(f"  {model.model_id}: {star_rating}")
    
    # Test single model
    single_result = [identical_results[0]]
    single_comparison = comparator.compare_models_from_results(single_result)
    
    print("\nSingle model test:")
    model = single_comparison.models[0]
    star_rating = ModelComparator.format_star_rating(model.score)
    print(f"  {model.model_id}: {star_rating}")


if __name__ == "__main__":
    print("1-5 Star Scoring System Testing")
    print("=" * 50)
    
    try:
        test_star_scoring_system()
        test_energy_focused_star_ratings()
        test_performance_focused_star_ratings()
        test_edge_cases_star_ratings()
        
        print("\n" + "=" * 50)
        print("✅ All star scoring tests completed!")
        print("Scoring system now uses 1-5 stars (5 = best)")
        print("Aligns with HuggingFace energy score format")
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()

