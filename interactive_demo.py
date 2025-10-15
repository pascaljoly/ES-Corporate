#!/usr/bin/env python3
"""
Interactive Demo for Energy Score Tool

This demo allows step-by-step presentation:
1. Score individual models one by one
2. Compare models and show rankings
3. Demonstrate different scoring weights
4. Show configuration system

Perfect for live presentations and manager demos.
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


def wait_for_user(message="Press Enter to continue..."):
    """Wait for user to press Enter"""
    input(f"\n{message}")


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_section(title):
    """Print a formatted section header"""
    print(f"\n📊 {title}")
    print("-" * 40)


def step_1_individual_scoring():
    """Step 1: Score individual models one by one"""
    print_header("Step 1: Individual Model Energy Scoring")
    
    print("Let's start by scoring individual models to see their energy consumption.")
    wait_for_user("Ready to score the first model?")
    
    # Create scorer
    scorer = ConfigAwareModelScorer()
    
    # Models to score
    models_to_score = [
        ("distilgpt2", "text-generation", "Small, efficient text generator"),
        ("gpt2", "text-generation", "Standard GPT-2 model"),
        ("gpt2-medium", "text-generation", "Larger GPT-2 model"),
        ("YOLOv8n", "object_detection", "Lightweight object detection"),
        ("BiRefNet", "object_detection", "Advanced object detection")
    ]
    
    results = []
    
    for i, (model_id, task, description) in enumerate(models_to_score, 1):
        print_section(f"Scoring Model {i}: {model_id}")
        print(f"Description: {description}")
        print(f"Task: {task}")
        
        wait_for_user(f"Press Enter to score {model_id}...")
        
        # Score the model
        print("⚡ Measuring energy consumption...")
        time.sleep(1)  # Simulate measurement time
        
        result = scorer.score(
            model=model_id,
            task=task,
            n_samples=50,
            runs=2
        )
        
        # Display results
        measurements = result.measurements
        metadata = result.metadata
        
        print(f"\n✅ Scoring Complete!")
        print(f"   ⚡ Energy: {measurements['energy_per_1k_wh']:.2f} kWh/1k queries")
        print(f"   🌍 CO2: {measurements['co2_per_1k_g']:.2f} kg CO2/1k queries")
        print(f"   🚀 Throughput: {measurements['samples_per_second']:.0f} samples/sec")
        print(f"   📦 Model Size: {metadata.get('model_size_mb', 'N/A')} MB")
        print(f"   🏗️  Architecture: {metadata.get('architecture', 'Unknown')}")
        
        results.append((model_id, result, description))
        
        if i < len(models_to_score):
            wait_for_user("Press Enter to score the next model...")
    
    print(f"\n🎉 All {len(models_to_score)} models scored successfully!")
    wait_for_user("Ready to compare these models?")
    
    return results


def step_2_model_comparison():
    """Step 2: Compare models and show rankings"""
    print_header("Step 2: Model Comparison & Ranking")
    
    print("Now let's compare these models and see which one is most energy-efficient.")
    wait_for_user("Ready to start the comparison?")
    
    # Create scorer and comparator
    scorer = ConfigAwareModelScorer()
    comparator = ModelComparator(scorer=scorer)
    
    # Compare text generation models
    print_section("Comparing Text Generation Models")
    
    text_models = [
        ("distilgpt2", "text-generation"),
        ("gpt2", "text-generation"),
        ("gpt2-medium", "text-generation")
    ]
    
    print("Models to compare:")
    for model_id, task in text_models:
        print(f"  • {model_id} ({task})")
    
    wait_for_user("Press Enter to run the comparison...")
    
    print("🔄 Running comparison analysis...")
    time.sleep(2)  # Simulate comparison time
    
    text_result = comparator.compare_models(
        model_specs=text_models,
        n_samples=50,
        runs=2
    )
    
    print(f"\n✅ Comparison Complete!")
    print(f"🏆 Winner: {text_result.summary['winner']} ({text_result.summary['winner_stars']} stars)")
    print(f"📈 Score Range: {text_result.summary['score_statistics']['min']:.1f} - {text_result.summary['score_statistics']['max']:.1f} stars")
    
    print(f"\n📋 Rankings:")
    for model in text_result.get_rankings():
        star_rating = ModelComparator.format_star_rating(model.score)
        measurements = model.scoring_result.measurements
        
        print(f"  {model.rank}. {model.model_id} - {star_rating}")
        print(f"     Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k | "
              f"CO2: {measurements['co2_per_1k_g']:.1f} kg | "
              f"Speed: {measurements['samples_per_second']:.0f} samples/sec")
    
    wait_for_user("Press Enter to compare computer vision models...")
    
    # Compare computer vision models
    print_section("Comparing Computer Vision Models")
    
    cv_models = [
        ("YOLOv8n", "object_detection"),
        ("BiRefNet", "object_detection")
    ]
    
    print("Models to compare:")
    for model_id, task in cv_models:
        print(f"  • {model_id} ({task})")
    
    wait_for_user("Press Enter to run the CV comparison...")
    
    print("🔄 Running computer vision comparison...")
    time.sleep(2)
    
    cv_result = comparator.compare_models(
        model_specs=cv_models,
        n_samples=50,
        runs=2
    )
    
    print(f"\n✅ CV Comparison Complete!")
    print(f"🏆 Winner: {cv_result.summary['winner']} ({cv_result.summary['winner_stars']} stars)")
    
    print(f"\n📋 Rankings:")
    for model in cv_result.get_rankings():
        star_rating = ModelComparator.format_star_rating(model.score)
        measurements = model.scoring_result.measurements
        
        print(f"  {model.rank}. {model.model_id} - {star_rating}")
        print(f"     Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k | "
              f"CO2: {measurements['co2_per_1k_g']:.1f} kg | "
              f"Speed: {measurements['samples_per_second']:.0f} samples/sec")
    
    return text_result, cv_result


def step_3_custom_weights():
    """Step 3: Demonstrate custom scoring weights"""
    print_header("Step 3: Custom Scoring Weights")
    
    print("Different organizations have different priorities. Let's see how rankings change")
    print("when we prioritize different metrics.")
    wait_for_user("Ready to see custom weight demonstrations?")
    
    scorer = ConfigAwareModelScorer()
    comparator = ModelComparator(scorer=scorer)
    
    models = [
        ("distilgpt2", "text-generation"),
        ("gpt2", "text-generation"),
        ("gpt2-medium", "text-generation")
    ]
    
    # Energy-focused comparison
    print_section("Energy-Focused Comparison")
    print("Scenario: A company prioritizing energy efficiency and sustainability")
    print("Weights: 60% Energy Efficiency, 40% CO2 Efficiency")
    
    wait_for_user("Press Enter to run energy-focused comparison...")
    
    energy_weights = {
        ComparisonMetric.ENERGY_EFFICIENCY: 0.6,
        ComparisonMetric.CO2_EFFICIENCY: 0.4,
        ComparisonMetric.PERFORMANCE: 0.0,
        ComparisonMetric.SPEED: 0.0
    }
    
    print("🔄 Running energy-focused analysis...")
    time.sleep(1)
    
    energy_result = comparator.compare_models(
        model_specs=models,
        custom_weights=energy_weights,
        n_samples=50,
        runs=2
    )
    
    print(f"\n✅ Energy-Focused Results:")
    print(f"🏆 Winner: {energy_result.summary['winner']} ({energy_result.summary['winner_stars']} stars)")
    
    wait_for_user("Press Enter to see performance-focused comparison...")
    
    # Performance-focused comparison
    print_section("Performance-Focused Comparison")
    print("Scenario: A company prioritizing model performance and speed")
    print("Weights: 40% Energy, 30% CO2, 30% Performance")
    
    wait_for_user("Press Enter to run performance-focused comparison...")
    
    performance_weights = {
        ComparisonMetric.ENERGY_EFFICIENCY: 0.4,
        ComparisonMetric.CO2_EFFICIENCY: 0.3,
        ComparisonMetric.PERFORMANCE: 0.3,
        ComparisonMetric.SPEED: 0.0
    }
    
    print("🔄 Running performance-focused analysis...")
    time.sleep(1)
    
    perf_result = comparator.compare_models(
        model_specs=models,
        custom_weights=performance_weights,
        n_samples=50,
        runs=2
    )
    
    print(f"\n✅ Performance-Focused Results:")
    print(f"🏆 Winner: {perf_result.summary['winner']} ({perf_result.summary['winner_stars']} stars)")
    
    print(f"\n💡 Key Insight: Rankings can change based on business priorities!")
    wait_for_user("Press Enter to see the configuration system...")
    
    return energy_result, perf_result


def step_4_configuration_system():
    """Step 4: Show configuration system"""
    print_header("Step 4: Configuration System")
    
    print("The system is highly configurable without requiring code changes.")
    wait_for_user("Ready to explore the configuration system?")
    
    config = get_config()
    
    print_section("Current Configuration")
    print(f"📊 Supported Tasks: {len(config.get_supported_tasks())} types")
    print(f"🖥️  Supported Hardware: {len(config.get_supported_hardware())} types")
    print(f"⚖️  Default Metric Weights: {config.get_metric_weights()}")
    print(f"⭐ Star Rating Range: {config.get('scoring.star_rating.min_stars')} - {config.get('scoring.star_rating.max_stars')}")
    
    # Show model profiles
    print(f"\n🤖 Model Profiles Available:")
    text_models = config.get("model_profiles.text_generation", {})
    cv_models = config.get("model_profiles.computer_vision", {})
    print(f"   Text Generation: {len(text_models)} models")
    print(f"   Computer Vision: {len(cv_models)} models")
    
    wait_for_user("Press Enter to see environment variable override demo...")
    
    # Demonstrate environment variable override
    print_section("Environment Variable Override")
    print("You can override any configuration setting using environment variables.")
    print(f"Current energy efficiency weight: {config.get('scoring.metric_weights.energy_efficiency')}")
    
    wait_for_user("Press Enter to simulate an override...")
    
    # Simulate override
    config._set_nested_value("scoring.metric_weights.energy_efficiency", 0.7)
    config._set_nested_value("scoring.metric_weights.co2_efficiency", 0.3)
    config._set_nested_value("scoring.metric_weights.performance", 0.0)
    config._set_nested_value("scoring.metric_weights.speed", 0.0)
    
    print(f"After override: {config.get_metric_weights()}")
    print("✅ Configuration updated without code changes!")
    
    wait_for_user("Press Enter to see business value summary...")
    
    return config


def step_5_business_value():
    """Step 5: Business value summary"""
    print_header("Step 5: Business Value & Use Cases")
    
    print("Let's summarize the key business benefits of this system.")
    wait_for_user("Ready to see the business value proposition?")
    
    print_section("Key Business Benefits")
    print("🎯 Cost Optimization: Choose energy-efficient models to reduce cloud costs")
    print("🌍 Sustainability: Track and minimize CO2 emissions from ML workloads")
    print("⚖️  Performance Trade-offs: Balance energy consumption with model performance")
    print("🏢 Vendor Comparison: Compare models across different providers")
    print("📊 Compliance: Meet environmental reporting requirements")
    
    wait_for_user("Press Enter to see real-world use cases...")
    
    print_section("Real-World Use Cases")
    print("💼 Model Selection: Choose between GPT-2 variants based on energy efficiency")
    print("🏗️  Infrastructure Planning: Estimate energy costs for production deployments")
    print("🌱 Green AI: Select models that meet sustainability goals")
    print("💰 Cost Analysis: Compare total cost of ownership (energy + compute)")
    print("📈 Performance Benchmarking: Standardized energy efficiency metrics")
    
    wait_for_user("Press Enter to see technical advantages...")
    
    print_section("Technical Advantages")
    print("🔧 No Code Changes: Update model profiles via configuration files")
    print("🌍 Environment Flexibility: Override settings for different deployments")
    print("⭐ Standardized Metrics: Consistent 1-5 star rating system")
    print("🔌 Extensible: Easy to add new models and metrics")
    print("✅ Validated: Automatic configuration validation prevents errors")
    
    wait_for_user("Press Enter for the final summary...")
    
    return True


def final_summary():
    """Final summary"""
    print_header("Demo Summary")
    
    print("✅ Successfully demonstrated:")
    print("  • Individual model energy scoring")
    print("  • Multi-model comparison and ranking")
    print("  • Custom scoring weights for different priorities")
    print("  • Configuration system flexibility")
    print("  • Business value and use cases")
    
    print(f"\n🎉 The Energy Score Tool is ready for production use!")
    print(f"   All models can be scored, compared, and ranked automatically.")
    print(f"   Configuration system enables easy customization without code changes.")
    
    print(f"\n📞 Next Steps:")
    print(f"   • Integrate with your existing ML workflows")
    print(f"   • Customize configuration for your specific needs")
    print(f"   • Start measuring energy consumption of your models")
    print(f"   • Make data-driven decisions about model selection")
    
    print(f"\nThank you for watching the Energy Score Tool demo! 🚀")


def main():
    """Run the interactive demo"""
    print_header("Energy Score Tool - Interactive Demo")
    print("This demo will walk you through each step of the energy scoring process.")
    print("Press Enter after each step to continue.")
    
    try:
        wait_for_user("Ready to start? Press Enter to begin...")
        
        # Step 1: Individual scoring
        scoring_results = step_1_individual_scoring()
        
        # Step 2: Model comparison
        text_result, cv_result = step_2_model_comparison()
        
        # Step 3: Custom weights
        energy_result, perf_result = step_3_custom_weights()
        
        # Step 4: Configuration system
        config = step_4_configuration_system()
        
        # Step 5: Business value
        step_5_business_value()
        
        # Final summary
        final_summary()
        
    except KeyboardInterrupt:
        print(f"\n\nDemo interrupted by user. Thanks for watching!")
    except Exception as e:
        print(f"\n❌ Demo Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
