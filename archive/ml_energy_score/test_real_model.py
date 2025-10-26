"""Test the complete energy measurement with a real model."""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml_energy_score.measure import measure_model_energy

def create_mock_text_dataset(size=20):
    """Create a simple mock text dataset for testing."""
    
    class MockDataset:
        def __init__(self, size):
            self.data = [
                {"text": f"This is a sample text for testing number {i}. It should be classified correctly."}
                for i in range(size)
            ]
        
        def __len__(self):
            return len(self.data)
        
        def __iter__(self):
            return iter(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
        
        def select(self, indices):
            selected_data = [self.data[i] for i in indices if i < len(self.data)]
            new_dataset = MockDataset(0)
            new_dataset.data = selected_data
            return new_dataset
    
    return MockDataset(size)


def test_with_small_model():
    """Test with a small, fast model."""
    
    print("🚀 Testing with Real Model")
    print("=" * 40)
    
    # Use a very small model for quick testing
    model_name = "prajjwal1/bert-tiny"  # Very small BERT model
    
    # Create test dataset
    print("Creating mock dataset...")
    dataset = create_mock_text_dataset(20)
    
    try:
        print(f"Testing model: {model_name}")
        print("Starting energy measurement...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = measure_model_energy(
                model_path=model_name,
                task="text-classification",
                dataset=dataset,
                hardware="CPU",  # Use CPU for compatibility
                num_samples=5,   # Small number for quick test
                output_dir=temp_dir
            )
        
        print("\n🎉 SUCCESS! Real model test completed!")
        print("=" * 50)
        print(f"Model: {result['model_name']}")
        print(f"Task: {result['task']}")
        print(f"Hardware: {result['hardware']}")
        print(f"Hardware Detected: {result['hardware_detected']}")
        print(f"Samples Processed: {result['num_samples']}")
        print(f"Duration: {result['duration_seconds']:.3f} seconds")
        print(f"Energy Consumed: {result['energy_kwh']:.6f} kWh")
        print(f"CO2 Emissions: {result['co2_kg']:.6f} kg")
        print(f"Samples/Second: {result['samples_per_second']:.2f}")
        print(f"kWh per 1000 queries: {result['kwh_per_1000_queries']:.6f}")
        
        if result['inference_results_sample']:
            print(f"Sample Results: {result['inference_results_sample']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during real model test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task_validation():
    """Test task validation."""
    
    print("\n🛡️ Testing Task Validation")
    print("=" * 30)
    
    dataset = create_mock_text_dataset(5)
    
    # Test invalid task
    try:
        measure_model_energy(
            model_path="test/model",
            task="invalid-task",
            dataset=dataset,
            hardware="CPU"
        )
        print("❌ Should have raised ValueError for invalid task")
        return False
    except ValueError as e:
        if "not supported" in str(e):
            print("✅ Task validation working correctly")
            return True
        else:
            print(f"❌ Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    print("ML Energy Score - Real Model Testing")
    print("=" * 50)
    
    # Test 1: Task validation
    validation_success = test_task_validation()
    
    # Test 2: Real model (only if validation passed)
    if validation_success:
        model_success = test_with_small_model()
        
        if model_success:
            print("\n🎯 ALL TESTS PASSED!")
            print("Your energy measurement script is fully functional! 🎉")
        else:
            print("\n⚠️ Model test failed, but framework is working")
    else:
        print("\n❌ Validation test failed")
