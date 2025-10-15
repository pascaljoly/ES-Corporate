#!/usr/bin/env python3
"""
Real-World Example: Text Generation Model Comparison

This script shows how to use the Energy Score Tool in a real production scenario.
It compares GPT-2 variants for a customer support chatbot use case.
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "energy-compare"))

def approach_1_measure_first():
    """Approach 1: Measure each model individually, then compare"""
    print("=== APPROACH 1: Measure First, Then Compare ===")
    print()
    
    # Step 1: Prepare your dataset
    print("Step 1: Prepare your dataset")
    print("---------------------------")
    print("# Load your customer support dataset")
    print("import pandas as pd")
    print("dataset = pd.read_csv('customer_questions.csv')")
    print("test_questions = dataset['question'].tolist()[:100]  # First 100 questions")
    print()
    
    # Step 2: Measure each model
    print("Step 2: Measure each model individually")
    print("--------------------------------------")
    print("from ml_energy_score.measure import measure_model_energy")
    print()
    print("# Measure DistilGPT-2")
    print("distilgpt2_result = measure_model_energy(")
    print("    model_path='distilgpt2',")
    print("    task='text-generation',")
    print("    dataset=test_questions,")
    print("    hardware='CPU',  # or 'GPU' if available")
    print("    n_samples=100,")
    print("    runs=3")
    print(")")
    print()
    print("# Measure GPT-2")
    print("gpt2_result = measure_model_energy(")
    print("    model_path='gpt2',")
    print("    task='text-generation',")
    print("    dataset=test_questions,")
    print("    hardware='CPU',")
    print("    n_samples=100,")
    print("    runs=3")
    print(")")
    print()
    print("# Measure GPT-2 Medium")
    print("gpt2_medium_result = measure_model_energy(")
    print("    model_path='gpt2-medium',")
    print("    task='text-generation',")
    print("    dataset=test_questions,")
    print("    hardware='CPU',")
    print("    n_samples=100,")
    print("    runs=3")
    print(")")
    print()
    
    # Step 3: Compare results
    print("Step 3: Compare the results")
    print("---------------------------")
    print("from energy_compare.comparator import ModelComparator")
    print()
    print("comparator = ModelComparator()")
    print("comparison = comparator.compare_models_from_results([")
    print("    distilgpt2_result,")
    print("    gpt2_result,")
    print("    gpt2_medium_result")
    print("])")
    print()
    
    # Step 4: Get results
    print("Step 4: Get the results")
    print("-----------------------")
    print("print(f'Winner: {comparison.summary[\"winner\"]}')")
    print("print(f'Energy savings: {comparison.summary[\"energy_savings\"]}%')")
    print()
    print("for model in comparison.get_rankings():")
    print("    print(f'{model.rank}. {model.model_id}: {model.score:.2f} stars')")
    print("    measurements = model.scoring_result.measurements")
    print("    print(f'   Energy: {measurements[\"energy_per_1k_wh\"]:.2f} kWh/1k queries')")
    print("    print(f'   CO2: {measurements[\"co2_per_1k_g\"]:.2f} kg CO2/1k queries')")
    print()


def approach_2_direct_comparison():
    """Approach 2: Direct comparison (auto-measures internally)"""
    print("=== APPROACH 2: Direct Comparison (Simpler) ===")
    print()
    
    print("Single step: Compare models directly")
    print("-----------------------------------")
    print("from energy_compare.comparator import ModelComparator")
    print()
    print("comparator = ModelComparator()")
    print("comparison = comparator.compare_models(")
    print("    model_specs=[")
    print("        ('distilgpt2', 'text-generation'),")
    print("        ('gpt2', 'text-generation'),")
    print("        ('gpt2-medium', 'text-generation')")
    print("    ],")
    print("    n_samples=100,")
    print("    runs=3")
    print(")")
    print()
    
    print("Get results")
    print("----------")
    print("print(f'Winner: {comparison.summary[\"winner\"]}')")
    print("for model in comparison.get_rankings():")
    print("    print(f'{model.rank}. {model.model_id}: {model.score:.2f} stars')")
    print("    measurements = model.scoring_result.measurements")
    print("    print(f'   Energy: {measurements[\"energy_per_1k_wh\"]:.2f} kWh/1k queries')")
    print("    print(f'   CO2: {measurements[\"co2_per_1k_g\"]:.2f} kg CO2/1k queries')")
    print()


def real_world_workflow():
    """Show the complete real-world workflow"""
    print("=== REAL-WORLD WORKFLOW: Text Generation Models ===")
    print()
    print("SCENARIO: You want to compare GPT-2 variants for your chatbot")
    print("You have a dataset of customer support questions to test with.")
    print()
    
    approach_1_measure_first()
    print()
    approach_2_direct_comparison()
    
    print("=== WHEN TO USE EACH APPROACH ===")
    print()
    print("Use Approach 1 (Measure First) when:")
    print("✅ You have custom datasets")
    print("✅ You want to test on different hardware")
    print("✅ You want to reuse measurements")
    print("✅ You're doing batch processing")
    print()
    print("Use Approach 2 (Direct Comparison) when:")
    print("✅ You want quick comparisons")
    print("✅ You're using standard test data")
    print("✅ You don't need to reuse measurements")
    print("✅ You want simpler workflow")
    print()
    
    print("=== EXPECTED OUTPUT ===")
    print()
    print("Winner: distilgpt2")
    print("1. distilgpt2: 5.00 stars")
    print("   Energy: 2.1 kWh/1k queries")
    print("   CO2: 1.0 kg CO2/1k queries")
    print("2. gpt2: 3.42 stars")
    print("   Energy: 3.5 kWh/1k queries")
    print("   CO2: 1.7 kg CO2/1k queries")
    print("3. gpt2-medium: 1.00 stars")
    print("   Energy: 6.4 kWh/1k queries")
    print("   CO2: 3.2 kg CO2/1k queries")
    print()
    print("Energy savings: 67% (distilgpt2 vs gpt2-medium)")


if __name__ == "__main__":
    real_world_workflow()
