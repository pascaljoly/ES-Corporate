# tests/test_scorer.py
"""Main test suite for ModelEnergyScorer"""
import pytest
from scorer.core import ModelEnergyScorer, ValidationError, ScoringResult

class TestInputValidation:
    """Test input validation logic"""
    
    def test_valid_inputs_pass(self, sample_model_id, sample_task):
        """Test that valid inputs don't raise errors"""
        scorer = ModelEnergyScorer()
        # Should not raise any exception
        scorer._validate_inputs(sample_model_id, sample_task)
    
    def test_invalid_task_raises_error(self, sample_model_id):
        """Test that invalid task raises ValidationError"""
        scorer = ModelEnergyScorer()
        
        with pytest.raises(ValidationError) as exc_info:
            scorer._validate_inputs(sample_model_id, "invalid_task_xyz")
        
        assert "not supported" in str(exc_info.value)
    
    def test_empty_model_raises_error(self, sample_task):
        """Test that empty model name raises ValidationError"""
        scorer = ModelEnergyScorer()
        
        with pytest.raises(ValidationError) as exc_info:
            scorer._validate_inputs("", sample_task)
        
        assert "Invalid model identifier" in str(exc_info.value)
    
    def test_none_model_raises_error(self, sample_task):
        """Test that None model raises ValidationError"""
        scorer = ModelEnergyScorer()
        
        with pytest.raises(ValidationError):
            scorer._validate_inputs(None, sample_task)
    
    def test_supported_tasks_work(self, sample_model_id):
        """Test all supported tasks are accepted"""
        from scorer.core import SUPPORTED_TASKS
        scorer = ModelEnergyScorer()
        
        for task in SUPPORTED_TASKS:
            # Should not raise
            scorer._validate_inputs(sample_model_id, task)


class TestScoringResult:
    """Test ScoringResult dataclass"""
    
    def test_result_creation(self):
        """Test that ScoringResult can be created"""
        result = ScoringResult(
            model_id="test_model",
            task="text_generation",
            measurements={'energy_per_1k_wh': 4.5},
            hardware={'gpu': 'Test GPU'},
            metadata={'timestamp': '2025-10-02'}
        )
        
        assert result.model_id == "test_model"
        assert result.task == "text_generation"
    
    def test_result_to_dict(self):
        """Test that result can be converted to dictionary"""
        result = ScoringResult(
            model_id="test_model",
            task="text_generation",
            measurements={'energy_per_1k_wh': 4.5},
            hardware={'gpu': 'Test GPU'},
            metadata={'timestamp': '2025-10-02'}
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['model_id'] == "test_model"
        assert result_dict['task'] == "text_generation"
    
    def test_low_variance_passes_validation(self):
        """Test that low variance passes validation"""
        result = ScoringResult(
            model_id="test",
            task="test",
            measurements={
                'statistics': {
                    'coefficient_of_variation': 0.08  # 8% < 15% threshold
                }
            },
            hardware={},
            metadata={}
        )
        
        is_valid, error_msg = result.is_valid()
        
        assert is_valid is True
        assert error_msg is None
    
    def test_high_variance_fails_validation(self):
        """Test that high variance fails validation"""
        result = ScoringResult(
            model_id="test",
            task="test",
            measurements={
                'statistics': {
                    'coefficient_of_variation': 0.25  # 25% > 15% threshold
                }
            },
            hardware={},
            metadata={}
        )
        
        is_valid, error_msg = result.is_valid()
        
        assert is_valid is False
        assert error_msg is not None
        assert "High variance" in error_msg


class TestScoring:
    """Test scoring functionality"""
    
    def test_score_returns_result(self, sample_model_id, sample_task):
        """Test that score() returns a ScoringResult"""
        scorer = ModelEnergyScorer()
        
        result = scorer.score(
            model=sample_model_id,
            task=sample_task,
            n_samples=10,  # Small number for fast test
            runs=2
        )
        
        assert isinstance(result, ScoringResult)
        assert result.model_id == sample_model_id
        assert result.task == sample_task
    
    def test_score_with_invalid_task_raises_error(self, sample_model_id):
        """Test that scoring with invalid task raises error"""
        scorer = ModelEnergyScorer()
        
        with pytest.raises(ValidationError):
            scorer.score(
                model=sample_model_id,
                task="invalid_task",
                runs=1
            )
    
    def test_score_result_has_required_fields(self, sample_model_id, sample_task):
        """Test that result contains all required fields"""
        scorer = ModelEnergyScorer()
        
        result = scorer.score(
            model=sample_model_id,
            task=sample_task,
            runs=2
        )
        
        # Check measurements exist
        assert 'energy_per_1k_wh' in result.measurements
        assert 'co2_per_1k_g' in result.measurements
        assert 'statistics' in result.measurements
        
        # Check hardware info exists
        assert result.hardware is not None
        
        # Check metadata exists
        assert 'timestamp' in result.metadata


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_zero_runs_handled(self, sample_model_id, sample_task):
        """Test that zero runs is handled appropriately"""
        scorer = ModelEnergyScorer()
        
        # Should raise ValidationError for zero runs
        with pytest.raises(ValidationError) as exc_info:
            scorer.score(
                model=sample_model_id,
                task=sample_task,
                runs=0
            )
        
        assert "runs must be positive" in str(exc_info.value)
    
    def test_negative_runs_handled(self, sample_model_id, sample_task):
        """Test that negative runs raises ValidationError"""
        scorer = ModelEnergyScorer()
        
        with pytest.raises(ValidationError) as exc_info:
            scorer.score(
                model=sample_model_id,
                task=sample_task,
                runs=-1
            )
        
        assert "runs must be positive" in str(exc_info.value)
    
    def test_zero_n_samples_handled(self, sample_model_id, sample_task):
        """Test that zero n_samples raises ValidationError"""
        scorer = ModelEnergyScorer()
        
        with pytest.raises(ValidationError) as exc_info:
            scorer.score(
                model=sample_model_id,
                task=sample_task,
                n_samples=0
            )
        
        assert "n_samples must be positive" in str(exc_info.value)
    
    def test_very_small_n_samples(self, sample_model_id, sample_task):
        """Test with very small sample size"""
        scorer = ModelEnergyScorer()
        
        result = scorer.score(
            model=sample_model_id,
            task=sample_task,
            n_samples=1,
            runs=1
        )
        
        assert result is not None
