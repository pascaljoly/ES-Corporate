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


def run_use_case(use_case_name, scenario, models):
    """Run a specific use case with given models"""
    print_header(f"Use Case: {use_case_name}")
    
    print(f"Scenario: {scenario}")
    print(f"You have {len(models)} options to compare based on energy efficiency.")
    wait_for_user(f"Ready to analyze {use_case_name.lower()} models?")
    
    # Create scorer and comparator
    scorer = ConfigAwareModelScorer()
    comparator = ModelComparator(scorer=scorer)
    
    print_section(f"{use_case_name} Models to Compare")
    for model_id, task, description in models:
        print(f"  • {model_id}: {description}")
    
    wait_for_user("Press Enter to run the comparison...")
    
    print(f"🔄 Analyzing {use_case_name.lower()} models...")
    time.sleep(2)  # Simulate analysis time
    
    result = comparator.compare_models(
        model_specs=[(model_id, task) for model_id, task, _ in models],
        n_samples=50,
        runs=2
    )
    
    print(f"\n✅ {use_case_name} Analysis Complete!")
    print(f"🏆 Winner: {result.summary['winner']} ({result.summary['winner_stars']} stars)")
    print(f"📈 Score Range: {result.summary['score_statistics']['min']:.1f} - {result.summary['score_statistics']['max']:.1f} stars")
    
    print(f"\n📋 {use_case_name} Rankings:")
    for model in result.get_rankings():
        star_rating = ModelComparator.format_star_rating(model.score)
        measurements = model.scoring_result.measurements
        
        print(f"  {model.rank}. {model.model_id} - {star_rating}")
        print(f"     Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k | "
              f"CO2: {measurements['co2_per_1k_g']:.1f} kg/1k | "
              f"Speed: {measurements['samples_per_second']:.0f} samples/sec")
    
    # Show CO2/Energy ratio to demonstrate proportionality
    print(f"\n💡 Key Insight - CO2 Efficiency vs Energy Efficiency:")
    print(f"   For the same use case (same datacenter location):")
    ratios = []
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        energy = measurements['energy_per_1k_wh']
        co2 = measurements['co2_per_1k_g']
        ratio = co2 / energy if energy > 0 else 0
        ratios.append(ratio)
        print(f"   • {model.model_id}: {ratio:.3f} kg CO2/kWh")
    
    if len(set([round(r, 2) for r in ratios])) == 1:
        print(f"   ✅ All models have the same CO2/Energy ratio!")
        print(f"   ✅ CO2 efficiency is proportional to energy efficiency")
        print(f"   ✅ Most energy-efficient = Most CO2-efficient")
    else:
        print(f"   📊 CO2/Energy ratios vary slightly (regional variations)")
    
    print(f"   🌍 This ratio represents your datacenter's carbon intensity")
    
    wait_for_user("Press Enter to continue...")
    
    return result




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
    print("Note: For same use case, energy and CO2 efficiency are proportional!")
    
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
    print(f"\n🌍 When CO2 Efficiency Matters Most:")
    print(f"   • Cross-region comparisons (different datacenter locations)")
    print(f"   • Multi-cloud deployments (AWS vs Azure vs GCP)")
    print(f"   • Green energy vs fossil fuel regions")
    print(f"   • Sustainability reporting and ESG compliance")
    print(f"   • Carbon offset planning")
    
    wait_for_user("Press Enter to see HuggingFace AI Energy Score compatibility...")
    
    # HuggingFace mode demonstration
    print_section("HuggingFace AI Energy Score (Default Mode)")
    print("By default, we use HuggingFace AI Energy Score approach:")
    print("• Focus on energy consumption (70% weight)")
    print("• Include CO2 efficiency (30% weight)")
    print("• Exclude performance metrics (0% weight)")
    print("This aligns with the industry standard HF energy score.")
    
    wait_for_user("Press Enter to run HuggingFace-compatible comparison...")
    
    from config_loader import set_huggingface_mode, is_huggingface_mode
    
    # Show current mode
    print(f"✅ Current mode: {'HuggingFace' if is_huggingface_mode() else 'Comprehensive'}")
    print("   • Energy efficiency: 70% weight")
    print("   • CO2 efficiency: 30% weight") 
    print("   • Performance: 0% weight (excluded)")
    print("   • Speed: 0% weight (excluded)")
    
    wait_for_user("Press Enter to run comparison...")
    
    print("🔄 Running HuggingFace-compatible analysis...")
    time.sleep(1)
    
    hf_result = comparator.compare_models(
        model_specs=models,
        n_samples=50,
        runs=2
    )
    
    print(f"\n✅ HuggingFace-Compatible Results:")
    print(f"🏆 Winner: {hf_result.summary['winner']} ({hf_result.summary['winner_stars']} stars)")
    print(f"📊 Focus: Pure energy efficiency (no performance bias)")
    
    # Show how to switch to comprehensive mode
    print(f"\n💡 To use comprehensive scoring (with performance metrics):")
    print(f"   • Set ENERGY_SCORE_SCORING_HUGGINGFACE_MODE_ENABLED=false")
    print(f"   • Or call set_huggingface_mode(False) in code")
    print(f"   • This enables: Energy(40%) + CO2(30%) + Performance(20%) + Speed(10%)")
    
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
    print(f"🤗 HuggingFace Mode: {'Enabled (Default)' if config.is_huggingface_mode() else 'Disabled'}")
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
    """Run the interactive demo with default use cases"""
    run_demo_with_use_cases()


def get_default_use_cases():
    """Get default use cases for the demo"""
    return [
        {
            "name": "Text Generation",
            "scenario": "Your company needs to choose a text generation model for customer support chatbots.",
            "models": [
                ("distilgpt2", "text-generation", "Small, efficient text generator"),
                ("gpt2", "text-generation", "Standard GPT-2 model"),
                ("gpt2-medium", "text-generation", "Larger GPT-2 model")
            ]
        },
        {
            "name": "Computer Vision",
            "scenario": "Your company needs to choose an object detection model for security cameras.",
            "models": [
                ("YOLOv8n", "object_detection", "Lightweight, fast object detection"),
                ("BiRefNet", "object_detection", "Advanced, accurate object detection")
            ]
        }
    ]


def run_demo_with_use_cases(use_cases=None):
    """Run demo with custom use cases"""
    if use_cases is None:
        use_cases = get_default_use_cases()
    
    print_header("Energy Score Tool - Interactive Demo")
    print("This demo shows real-world use cases for model selection.")
    print("Press Enter after each step to continue.")
    
    try:
        wait_for_user("Ready to start? Press Enter to begin...")
        
        results = []
        
        # Run each use case
        for i, use_case in enumerate(use_cases, 1):
            result = run_use_case(
                use_case["name"],
                use_case["scenario"], 
                use_case["models"]
            )
            results.append((use_case["name"], result))
            
            if i < len(use_cases):
                wait_for_user("Press Enter to see the next use case...")
        
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
    # You can customize use cases here or pass them as parameters
    custom_use_cases = None  # Use default use cases
    
    # Example of custom use cases:
    # custom_use_cases = [
    #     {
    #         "name": "Sentiment Analysis",
    #         "scenario": "Choose a model for analyzing customer feedback sentiment.",
    #         "models": [
    #             ("distilbert-base-uncased", "text-classification", "Fast sentiment analysis"),
    #             ("roberta-base", "text-classification", "Accurate sentiment analysis")
    #         ]
    #     }
    # ]
    
    run_demo_with_use_cases(custom_use_cases)
