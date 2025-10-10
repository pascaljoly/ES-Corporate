"""Example usage of the ML energy measurement script with real models."""

import time
from datasets import load_dataset
from transformers import pipeline
from ml_energy_score.measure import measure_model_energy


def create_simple_inference_function():
    """Create a simple inference function for testing."""
    
    def run_inference_on_dataset(model_name, task, dataset, num_samples=100):
        """
        Example of how to integrate actual model inference.
        This is what would go in the TODO section of measure.py
        """
        print(f"Loading model: {model_name}")
        
        # Load model based on task
        if task == "text-classification":
            classifier = pipeline("text-classification", model=model_name)
            
            # Run inference on samples
            results = []
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                    
                # Get text field (adapt based on dataset structure)
                text = sample.get('text', sample.get('sentence', str(sample)))
                result = classifier(text)
                results.append(result)
                
                # Small delay to simulate processing time
                time.sleep(0.01)
                
        elif task == "image-classification":
            classifier = pipeline("image-classification", model=model_name)
            
            results = []
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                    
                # Get image field (adapt based on dataset structure)  
                image = sample.get('image', sample.get('img'))
                if image:
                    result = classifier(image)
                    results.append(result)
                
                time.sleep(0.01)
        
        else:
            # For other tasks, just simulate work
            results = []
            for i in range(min(num_samples, len(dataset))):
                time.sleep(0.01)  # Simulate processing
                results.append({"prediction": f"result_{i}"})
        
        return results
    
    return run_inference_on_dataset


def test_with_real_model():
    """Test the energy measurement with a real small model."""
    
    print("🚀 Testing with real model...")
    
    try:
        # Use a small, fast model for testing
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        
        # Create a small test dataset
        print("Loading dataset...")
        dataset = load_dataset("imdb", split="test[:50]")  # Just 50 samples
        
        print("Starting energy measurement...")
        result = measure_model_energy(
            model_path=model_name,
            task="text-classification", 
            dataset=dataset,
            hardware="CPU",  # Use CPU for compatibility
            num_samples=10,  # Small number for quick test
            output_dir="test_results"
        )
        
        print("✅ Real model test completed!")
        print(f"Model: {result['model_name']}")
        print(f"Samples processed: {result['num_samples']}")
        print(f"Energy consumed: {result['energy_kwh']} kWh")
        print(f"CO2 emissions: {result['co2_kg']} kg")
        
    except Exception as e:
        print(f"❌ Error in real model test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("ML Energy Score - Example Usage")
    print("=" * 40)
    
    # Show how the inference function would work
    inference_fn = create_simple_inference_function()
    print("✅ Inference function created")
    
    # Test with real model (commented out by default)
    # Uncomment the line below to test with a real model
    # test_with_real_model()
    
    print("\n📝 Next steps to complete the implementation:")
    print("1. Integrate the inference function into measure.py")
    print("2. Extract actual energy/CO2 values from CodeCarbon tracker")
    print("3. Add proper timing for duration calculation")
    print("4. Implement hardware auto-detection")
    print("5. Add support for different model formats (ONNX, TensorRT, etc.)")
