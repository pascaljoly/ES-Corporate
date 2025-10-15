#!/usr/bin/env python3
"""
End-to-End Energy Score Tool Demo for Management

This demo showcases the complete scorer + comparator solution:
1. Model Energy Scoring - Measure energy consumption of ML models
2. Model Comparison - Compare multiple models and rank them
3. Configuration System - Easy customization without code changes

Perfect for demonstrating the business value and technical capabilities.
"""

import sys
from pathlib import Path
import time

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "energy-compare"))

from config_loader import get_config
from scorer.config_aware_scorer import ConfigAwareModelScorer
from comparator import ModelComparator, ComparisonMetric


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_section(title):
    """Print a formatted section header"""
    print(f"\n📊 {title}")
    print("-" * 40)


def demo_model_scoring():
    """Demonstrate individual model energy scoring"""
    print_section("Model Energy Scoring")
    
    # Create scorer
    scorer = ConfigAwareModelScorer()
    
    # Score different types of models
    models_to_score = [
        ("distilgpt2", "text-generation", "Small, efficient text generator"),
        ("gpt2", "text-generation", "Standard GPT-2 model"),
        ("BiRefNet", "object_detection", "Computer vision model for object detection"),
        ("YOLOv8n", "object_detection", "Lightweight object detection model")
    ]
    
    print("Scoring individual models...")
    results = []
    
    for model_id, task, description in models_to_score:
        print(f"\n  🔍 Scoring: {model_id}")
        print(f"     Description: {description}")
        
        # Score the model
        result = scorer.score(
            model=model_id,
            task=task,
            n_samples=50,
            runs=2
        )
        
        # Display results
        measurements = result.measurements
        metadata = result.metadata
        
        print(f"     ⚡ Energy: {measurements['energy_per_1k_wh']:.2f} kWh/1k queries")
        print(f"     🌍 CO2: {measurements['co2_per_1k_g']:.2f} kg CO2/1k queries")
        print(f"     🚀 Throughput: {measurements['samples_per_second']:.0f} samples/sec")
        print(f"     📦 Model Size: {metadata.get('model_size_mb', 'N/A')} MB")
        print(f"     🏗️  Architecture: {metadata.get('architecture', 'Unknown')}")
        
        results.append((model_id, result, description))
    
    return results


def demo_model_comparison():
    """Demonstrate model comparison and ranking"""
    print_section("Model Comparison & Ranking")
    
    # Create scorer and comparator
    scorer = ConfigAwareModelScorer()
    comparator = ModelComparator(scorer=scorer)
    
    # Compare text generation models
    print("Comparing Text Generation Models...")
    text_models = [
        ("distilgpt2", "text-generation"),
        ("gpt2", "text-generation"),
        ("gpt2-medium", "text-generation")
    ]
    
    text_result = comparator.compare_models(
        model_specs=text_models,
        n_samples=50,
        runs=2
    )
    
    print(f"\n  🏆 Winner: {text_result.summary['winner']} ({text_result.summary['winner_stars']} stars)")
    print(f"  📈 Score Range: {text_result.summary['score_statistics']['min']:.1f} - {text_result.summary['score_statistics']['max']:.1f} stars")
    
    print("\n  📋 Rankings:")
    for model in text_result.get_rankings():
        star_rating = ModelComparator.format_star_rating(model.score)
        measurements = model.scoring_result.measurements
        metadata = model.scoring_result.metadata
        
        print(f"    {model.rank}. {model.model_id} - {star_rating}")
        print(f"       Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k | "
              f"CO2: {measurements['co2_per_1k_g']:.1f} kg | "
              f"Speed: {measurements['samples_per_second']:.0f} samples/sec")
    
    # Compare computer vision models
    print("\n\nComparing Computer Vision Models...")
    cv_models = [
        ("YOLOv8n", "object_detection"),
        ("YOLOv8s", "object_detection"),
        ("BiRefNet", "object_detection")
    ]
    
    cv_result = comparator.compare_models(
        model_specs=cv_models,
        n_samples=50,
        runs=2
    )
    
    print(f"\n  🏆 Winner: {cv_result.summary['winner']} ({cv_result.summary['winner_stars']} stars)")
    
    print("\n  📋 Rankings:")
    for model in cv_result.get_rankings():
        star_rating = ModelComparator.format_star_rating(model.score)
        measurements = model.scoring_result.measurements
        metadata = model.scoring_result.metadata
        
        print(f"    {model.rank}. {model.model_id} - {star_rating}")
        print(f"       Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k | "
              f"CO2: {measurements['co2_per_1k_g']:.1f} kg | "
              f"Speed: {measurements['samples_per_second']:.0f} samples/sec")
    
    return text_result, cv_result


def demo_custom_scoring_weights():
    """Demonstrate custom scoring weights for different priorities"""
    print_section("Custom Scoring Weights")
    
    scorer = ConfigAwareModelScorer()
    comparator = ModelComparator(scorer=scorer)
    
    models = [
        ("distilgpt2", "text-generation"),
        ("gpt2", "text-generation"),
        ("gpt2-medium", "text-generation")
    ]
    
    # Energy-focused comparison
    print("Energy-Focused Comparison (60% energy, 40% CO2):")
    energy_weights = {
        ComparisonMetric.ENERGY_EFFICIENCY: 0.6,
        ComparisonMetric.CO2_EFFICIENCY: 0.4,
        ComparisonMetric.PERFORMANCE: 0.0,
        ComparisonMetric.SPEED: 0.0
    }
    
    energy_result = comparator.compare_models(
        model_specs=models,
        custom_weights=energy_weights,
        n_samples=50,
        runs=2
    )
    
    print(f"  🏆 Winner: {energy_result.summary['winner']} ({energy_result.summary['winner_stars']} stars)")
    
    # Performance-focused comparison
    print("\nPerformance-Focused Comparison (40% energy, 30% CO2, 30% performance):")
    performance_weights = {
        ComparisonMetric.ENERGY_EFFICIENCY: 0.4,
        ComparisonMetric.CO2_EFFICIENCY: 0.3,
        ComparisonMetric.PERFORMANCE: 0.3,
        ComparisonMetric.SPEED: 0.0
    }
    
    perf_result = comparator.compare_models(
        model_specs=models,
        custom_weights=performance_weights,
        n_samples=50,
        runs=2
    )
    
    print(f"  🏆 Winner: {perf_result.summary['winner']} ({perf_result.summary['winner_stars']} stars)")
    
    return energy_result, perf_result


def demo_configuration_system():
    """Demonstrate the configuration system benefits"""
    print_section("Configuration System")
    
    config = get_config()
    
    print("Current Configuration:")
    print(f"  📊 Supported Tasks: {len(config.get_supported_tasks())} types")
    print(f"  🖥️  Supported Hardware: {len(config.get_supported_hardware())} types")
    print(f"  ⚖️  Metric Weights: {config.get_metric_weights()}")
    print(f"  ⭐ Star Rating Range: {config.get('scoring.star_rating.min_stars')} - {config.get('scoring.star_rating.max_stars')}")
    
    # Show model profiles
    print(f"\n  🤖 Model Profiles Available:")
    text_models = config.get("model_profiles.text_generation", {})
    cv_models = config.get("model_profiles.computer_vision", {})
    print(f"     Text Generation: {len(text_models)} models")
    print(f"     Computer Vision: {len(cv_models)} models")
    
    # Demonstrate environment variable override
    print(f"\n  🔧 Environment Variable Override Demo:")
    print(f"     Current energy efficiency weight: {config.get('scoring.metric_weights.energy_efficiency')}")
    
    # Simulate override
    config._set_nested_value("scoring.metric_weights.energy_efficiency", 0.7)
    config._set_nested_value("scoring.metric_weights.co2_efficiency", 0.3)
    config._set_nested_value("scoring.metric_weights.performance", 0.0)
    config._set_nested_value("scoring.metric_weights.speed", 0.0)
    
    print(f"     After override: {config.get_metric_weights()}")
    
    return config


def demo_business_value():
    """Demonstrate business value and use cases"""
    print_section("Business Value & Use Cases")
    
    print("🎯 Key Business Benefits:")
    print("  • Cost Optimization: Choose energy-efficient models to reduce cloud costs")
    print("  • Sustainability: Track and minimize CO2 emissions from ML workloads")
    print("  • Performance Trade-offs: Balance energy consumption with model performance")
    print("  • Vendor Comparison: Compare models across different providers")
    print("  • Compliance: Meet environmental reporting requirements")
    
    print("\n💼 Real-World Use Cases:")
    print("  • Model Selection: Choose between GPT-2 variants based on energy efficiency")
    print("  • Infrastructure Planning: Estimate energy costs for production deployments")
    print("  • Green AI: Select models that meet sustainability goals")
    print("  • Cost Analysis: Compare total cost of ownership (energy + compute)")
    print("  • Performance Benchmarking: Standardized energy efficiency metrics")
    
    print("\n🔧 Technical Advantages:")
    print("  • No Code Changes: Update model profiles via configuration files")
    print("  • Environment Flexibility: Override settings for different deployments")
    print("  • Standardized Metrics: Consistent 1-5 star rating system")
    print("  • Extensible: Easy to add new models and metrics")
    print("  • Validated: Automatic configuration validation prevents errors")


def main():
    """Run the complete end-to-end demo"""
    print_header("Energy Score Tool - End-to-End Demo")
    print("Demonstrating: Model Scoring → Comparison → Ranking → Configuration")
    
    try:
        # Demo 1: Individual model scoring
        scoring_results = demo_model_scoring()
        
        # Demo 2: Model comparison and ranking
        text_result, cv_result = demo_model_comparison()
        
        # Demo 3: Custom scoring weights
        energy_result, perf_result = demo_custom_scoring_weights()
        
        # Demo 4: Configuration system
        config = demo_configuration_system()
        
        # Demo 5: Business value
        demo_business_value()
        
        # Summary
        print_header("Demo Summary")
        print("✅ Successfully demonstrated:")
        print("  • Individual model energy scoring")
        print("  • Multi-model comparison and ranking")
        print("  • Custom scoring weights for different priorities")
        print("  • Configuration system flexibility")
        print("  • Business value and use cases")
        
        print(f"\n📊 Demo Results:")
        print(f"  • Scored {len(scoring_results)} individual models")
        print(f"  • Compared {len(text_result.models)} text generation models")
        print(f"  • Compared {len(cv_result.models)} computer vision models")
        print(f"  • Demonstrated {len(config.get_supported_tasks())} supported task types")
        
        print(f"\n🎉 The Energy Score Tool is ready for production use!")
        print(f"   All models can be scored, compared, and ranked automatically.")
        print(f"   Configuration system enables easy customization without code changes.")
        
    except Exception as e:
        print(f"\n❌ Demo Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
