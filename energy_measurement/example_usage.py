#!/usr/bin/env python3
"""
HOW TO USE THE ENERGY MEASUREMENT TOOL
=======================================

This example shows you how to measure the energy consumption of YOUR ML model.

Follow these steps:
1. Create your inference function (the function that runs your model)
2. Prepare your dataset
3. Call measure_energy() with your function and data
4. Review the results

That's it! The tool handles all the energy tracking automatically.
"""

from energy_measurement.measure_energy import measure_energy


# =============================================================================
# STEP 1: CREATE YOUR INFERENCE FUNCTION
# =============================================================================
# This is the function that runs your model on a single sample.
# It should take one sample from your dataset and return a prediction.

def my_model_inference(sample):
    """
    Your inference function - this is where YOUR model runs.

    Args:
        sample: One item from your dataset (can be dict, list, tensor, etc.)

    Returns:
        Your model's prediction (any format)

    HOW TO ADAPT THIS FOR YOUR MODEL:
    1. Load your model (do this outside the function if possible)
    2. Preprocess the sample if needed
    3. Run inference
    4. Return the result
    """
    # Example: Simple text processing
    # REPLACE THIS with your actual model inference code

    text = sample.get('text', '')

    # Your model inference goes here
    # For example:
    # - prediction = your_model(text)
    # - prediction = your_transformer(text)
    # - prediction = your_api_call(text)

    # For this example, we'll simulate a simple prediction
    prediction = len(text) % 3  # Dummy prediction

    return {'prediction': prediction, 'confidence': 0.85}


# =============================================================================
# STEP 2: PREPARE YOUR DATASET
# =============================================================================
# Your dataset should be a list or iterable of samples.
# Each sample will be passed to your inference function.

def create_my_dataset():
    """
    Create or load your dataset.

    Returns:
        List of samples (each sample should match what your inference function expects)

    HOW TO ADAPT THIS FOR YOUR DATA:
    - Load from file: pd.read_csv(), json.load(), etc.
    - Load from HuggingFace: datasets.load_dataset()
    - Use existing list/array
    - Query from database
    """
    # Example: Simple text dataset
    # REPLACE THIS with your actual data loading code

    dataset = [
        {'text': 'This is a sample text for classification', 'label': 0},
        {'text': 'Another example of text data', 'label': 1},
        {'text': 'Energy measurement is important for AI', 'label': 2},
        # ... add more samples
    ]

    # For demonstration, create 100 samples
    dataset = [
        {'text': f'Sample text number {i} for testing', 'label': i % 3}
        for i in range(100)
    ]

    return dataset


# =============================================================================
# STEP 3: MEASURE ENERGY CONSUMPTION
# =============================================================================

def main():
    """
    Main function - this is where you measure energy consumption.
    """

    print("=" * 70)
    print("ENERGY MEASUREMENT EXAMPLE")
    print("=" * 70)
    print()
    print("This example shows you how to measure YOUR model's energy consumption.")
    print()

    # Load your dataset
    print("Step 1: Loading dataset...")
    dataset = create_my_dataset()
    print(f"✓ Loaded {len(dataset)} samples")
    print()

    # Measure energy consumption
    print("Step 2: Measuring energy consumption...")
    print("(This will run your model on the samples and track energy usage)")
    print()

    results = measure_energy(
        # Your inference function
        inference_fn=my_model_inference,

        # Your dataset
        dataset=dataset,

        # Model identification
        model_name="my_text_classifier_v1",  # Change this to your model name
        task_name="text-classification",      # e.g., "image-classification", "object-detection", etc.
        hardware="CPU",                        # Options: "CPU", "T4", "V100", "A100", "H100"

        # How many samples to measure (use smaller number for quick tests)
        num_samples=50,  # Increase to 1000+ for production measurements

        # Random seed (use same seed to compare different models fairly)
        seed=42,

        # Where to save results
        output_dir="results"  # Results saved to: results/{task_name}/{hardware}/
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Model: {results['model_name']}")
    print(f"Task: {results['task_name']}")
    print(f"Hardware: {results['hardware']}")
    print(f"Samples measured: {results['num_samples']}")
    print()
    print(f"✓ Energy consumed: {results['energy_kwh']:.6f} kWh")
    print(f"✓ Duration: {results['duration_seconds']:.2f} seconds")
    print(f"✓ Energy per 1000 queries: {results['kwh_per_1000_queries']:.6f} kWh")
    print()
    print("📁 Full results saved to JSON file")
    print()
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. CUSTOMIZE THIS EXAMPLE:")
    print("   - Replace my_model_inference() with your actual model")
    print("   - Replace create_my_dataset() with your actual data")
    print("   - Update model_name, task_name, and hardware")
    print()
    print("2. MEASURE DIFFERENT MODELS:")
    print("   - Run this script with different models")
    print("   - Use the SAME seed and num_samples for fair comparison")
    print()
    print("3. CALCULATE SCORES:")
    print("   - After measuring multiple models, run calculate_scores.py")
    print("   - This will rank your models by energy efficiency")
    print()
    print("4. PRODUCTION USE:")
    print("   - Increase num_samples to 1000+ for accurate measurements")
    print("   - Run on real hardware (Intel/AMD CPU or NVIDIA GPU)")
    print("   - Note: Apple Silicon (M1/M2) may show 0 kWh due to limitations")
    print()
    print("=" * 70)

    return results


# =============================================================================
# RUN THE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Measurement interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nIf you need help, check the documentation in energy_measurement/README.md")
