# tests/test_comparator.py
"""Test suite for ModelComparator"""
import pytest
from unittest.mock import Mock, patch
from comparator.core import (
    ModelComparator, 
    ModelComparison, 
    ComparisonResult, 
    ComparisonMetric,
    ComparisonError
)
from scorer.core import ScoringResult


class TestModelComparison:
    """Test ModelComparison dataclass"""
    
    def test_model_comparison_creation(self):
        """Test creating a ModelComparison"""
        scoring_result = ScoringResult(
            model_id="test_model",
            task="text_generation",
            measurements={'energy_per_1k_wh': 4.5},
            hardware={'gpu': 'Test GPU'},
            metadata={'timestamp': '2025-10-02'}
        )
        
        comparison = ModelComparison(
            model_id="test_model",
            task="text_generation",
            scoring_result=scoring_result,
            rank=1,
            score=0.85
        )
        
        assert comparison.model_id == "test_model"
        assert comparison.task == "text_generation"
        assert comparison.rank == 1
        assert comparison.score == 0.85
    
    def test_model_comparison_to_dict(self):
        """Test converting ModelComparison to dictionary"""
        scoring_result = ScoringResult(
            model_id="test_model",
            task="text_generation",
            measurements={'energy_per_1k_wh': 4.5},
            hardware={'gpu': 'Test GPU'},
            metadata={'timestamp': '2025-10-02'}
        )
        
        comparison = ModelComparison(
            model_id="test_model",
            task="text_generation",
            scoring_result=scoring_result,
            rank=1,
            score=0.85
        )
        
        result_dict = comparison.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['model_id'] == "test_model"
        assert result_dict['rank'] == 1
        assert result_dict['score'] == 0.85
        assert 'measurements' in result_dict
        assert 'hardware' in result_dict


class TestComparisonResult:
    """Test ComparisonResult dataclass"""
    
    def test_comparison_result_creation(self):
        """Test creating a ComparisonResult"""
        scoring_result = ScoringResult(
            model_id="test_model",
            task="text_generation",
            measurements={'energy_per_1k_wh': 4.5},
            hardware={'gpu': 'Test GPU'},
            metadata={'timestamp': '2025-10-02'}
        )
        
        model_comparison = ModelComparison(
            model_id="test_model",
            task="text_generation",
            scoring_result=scoring_result,
            rank=1,
            score=0.85
        )
        
        result = ComparisonResult(
            task="text_generation",
            models=[model_comparison],
            comparison_metrics=[ComparisonMetric.ENERGY_EFFICIENCY]
        )
        
        assert result.task == "text_generation"
        assert len(result.models) == 1
        assert result.models[0].model_id == "test_model"
    
    def test_get_winner(self):
        """Test getting the winner from comparison result"""
        # Create two models with different ranks
        model1 = ModelComparison(
            model_id="model1",
            task="text_generation",
            scoring_result=Mock(),
            rank=1,
            score=0.9
        )
        
        model2 = ModelComparison(
            model_id="model2",
            task="text_generation",
            scoring_result=Mock(),
            rank=2,
            score=0.7
        )
        
        result = ComparisonResult(
            task="text_generation",
            models=[model1, model2],
            comparison_metrics=[ComparisonMetric.ENERGY_EFFICIENCY]
        )
        
        winner = result.get_winner()
        assert winner is not None
        assert winner.model_id == "model1"
        assert winner.rank == 1
    
    def test_get_rankings(self):
        """Test getting models sorted by rank"""
        model1 = ModelComparison(
            model_id="model1",
            task="text_generation",
            scoring_result=Mock(),
            rank=2,
            score=0.7
        )
        
        model2 = ModelComparison(
            model_id="model2",
            task="text_generation",
            scoring_result=Mock(),
            rank=1,
            score=0.9
        )
        
        result = ComparisonResult(
            task="text_generation",
            models=[model1, model2],
            comparison_metrics=[ComparisonMetric.ENERGY_EFFICIENCY]
        )
        
        rankings = result.get_rankings()
        assert len(rankings) == 2
        assert rankings[0].rank == 1
        assert rankings[1].rank == 2


class TestModelComparator:
    """Test ModelComparator class"""
    
    @pytest.fixture
    def mock_scorer(self):
        """Mock ModelEnergyScorer"""
        scorer = Mock()
        return scorer
    
    @pytest.fixture
    def sample_scoring_results(self):
        """Sample scoring results for testing"""
        results = []
        for i, model_id in enumerate(["model1", "model2", "model3"]):
            result = ScoringResult(
                model_id=model_id,
                task="text_generation",
                measurements={
                    'energy_per_1k_wh': 4.0 + i * 0.5,  # 4.0, 4.5, 5.0
                    'co2_per_1k_g': 2.0 + i * 0.25,     # 2.0, 2.25, 2.5
                    'samples_per_second': 100 - i * 10,  # 100, 90, 80
                    'duration_seconds': 10 + i * 2       # 10, 12, 14
                },
                hardware={'gpu': f'GPU{i}'},
                metadata={'timestamp': '2025-10-02'}
            )
            results.append(result)
        return results
    
    def test_comparator_initialization(self, mock_scorer):
        """Test ModelComparator initialization"""
        comparator = ModelComparator(scorer=mock_scorer)
        assert comparator.scorer == mock_scorer
        assert len(comparator.metric_weights) > 0
    
    def test_compare_models_from_results(self, sample_scoring_results):
        """Test comparing models from pre-computed results"""
        comparator = ModelComparator()
        
        result = comparator.compare_models_from_results(sample_scoring_results)
        
        assert isinstance(result, ComparisonResult)
        assert len(result.models) == 3
        assert result.task == "text_generation"
        
        # Check that rankings were assigned
        for model in result.models:
            assert model.rank is not None
            assert model.score is not None
        
        # Check that models are ranked correctly
        rankings = result.get_rankings()
        assert rankings[0].rank == 1
        assert rankings[1].rank == 2
        assert rankings[2].rank == 3
    
    def test_compare_models_with_custom_weights(self, sample_scoring_results):
        """Test comparison with custom metric weights"""
        comparator = ModelComparator()
        
        custom_weights = {
            ComparisonMetric.ENERGY_EFFICIENCY: 0.8,
            ComparisonMetric.CO2_EFFICIENCY: 0.2
        }
        
        result = comparator.compare_models_from_results(
            sample_scoring_results,
            custom_weights=custom_weights
        )
        
        assert isinstance(result, ComparisonResult)
        # Model with lowest energy should win
        winner = result.get_winner()
        assert winner.model_id == "model1"  # Lowest energy
    
    def test_compare_models_validation_errors(self):
        """Test validation errors in model comparison"""
        comparator = ModelComparator()
        
        # Empty model specs
        with pytest.raises(ComparisonError):
            comparator.compare_models([])
        
        # Different tasks
        with pytest.raises(ComparisonError):
            comparator.compare_models([
                ("model1", "text_generation"),
                ("model2", "image_classification")
            ])
        
        # Duplicate model IDs
        with pytest.raises(ComparisonError):
            comparator.compare_models([
                ("model1", "text_generation"),
                ("model1", "text_generation")
            ])
    
    def test_invalid_weights_validation(self, sample_scoring_results):
        """Test validation of custom weights"""
        comparator = ModelComparator()
        
        # Weights that don't sum to 1.0
        invalid_weights = {
            ComparisonMetric.ENERGY_EFFICIENCY: 0.5,
            ComparisonMetric.CO2_EFFICIENCY: 0.3  # Sum = 0.8, not 1.0
        }
        
        with pytest.raises(ComparisonError):
            comparator.compare_models_from_results(
                sample_scoring_results,
                custom_weights=invalid_weights
            )
    
    def test_metric_score_calculations(self, sample_scoring_results):
        """Test individual metric score calculations"""
        comparator = ModelComparator()
        
        # Test energy efficiency (lower is better)
        model1 = ModelComparison(
            model_id="model1",
            task="text_generation",
            scoring_result=sample_scoring_results[0]  # Lowest energy
        )
        
        score = comparator._calculate_metric_score(
            model1, 
            ComparisonMetric.ENERGY_EFFICIENCY, 
            [ModelComparison(
                model_id=f"model{i}",
                task="text_generation",
                scoring_result=result
            ) for i, result in enumerate(sample_scoring_results)]
        )
        
        # Model with lowest energy should get highest score
        assert score == 1.0  # Best score for lowest energy
    
    def test_normalization_functions(self):
        """Test normalization helper functions"""
        comparator = ModelComparator()
        
        # Test lower is better normalization
        values = [1.0, 2.0, 3.0, 4.0]
        normalized = comparator._normalize_lower_better(1.0, values)
        assert normalized == 1.0  # Lowest value gets highest score
        
        normalized = comparator._normalize_lower_better(4.0, values)
        assert normalized == 0.0  # Highest value gets lowest score
        
        # Test higher is better normalization
        normalized = comparator._normalize_higher_better(4.0, values)
        assert normalized == 1.0  # Highest value gets highest score
        
        normalized = comparator._normalize_higher_better(1.0, values)
        assert normalized == 0.0  # Lowest value gets lowest score
    
    def test_summary_generation(self, sample_scoring_results):
        """Test comparison summary generation"""
        comparator = ModelComparator()
        
        result = comparator.compare_models_from_results(sample_scoring_results)
        summary = result.summary
        
        assert "total_models" in summary
        assert "metrics_used" in summary
        assert "score_statistics" in summary
        assert "winner" in summary
        assert "energy_range" in summary
        assert "co2_range" in summary
        
        assert summary["total_models"] == 3
        assert summary["winner"] == "model1"  # Should be the winner
    
    @patch('comparator.core.ModelEnergyScorer')
    def test_compare_models_with_scoring(self, mock_scorer_class, sample_scoring_results):
        """Test comparing models with actual scoring"""
        # Setup mock scorer
        mock_scorer = Mock()
        mock_scorer_class.return_value = mock_scorer
        
        # Configure mock to return our sample results
        mock_scorer.score.side_effect = sample_scoring_results
        
        comparator = ModelComparator()
        
        model_specs = [
            ("model1", "text_generation"),
            ("model2", "text_generation"),
            ("model3", "text_generation")
        ]
        
        result = comparator.compare_models(model_specs, n_samples=10, runs=2)
        
        assert isinstance(result, ComparisonResult)
        assert len(result.models) == 3
        
        # Verify scorer was called for each model
        assert mock_scorer.score.call_count == 3
    
    def test_save_and_load_comparison(self, sample_scoring_results, tmp_path):
        """Test saving and loading comparison results"""
        comparator = ModelComparator()
        
        result = comparator.compare_models_from_results(sample_scoring_results)
        
        # Save to temporary file
        output_file = tmp_path / "comparison.json"
        comparator.save_comparison(result, output_file)
        
        # Load from file
        loaded_result = comparator.load_comparison(output_file)
        
        assert isinstance(loaded_result, ComparisonResult)
        assert len(loaded_result.models) == len(result.models)
        assert loaded_result.task == result.task
        
        # Check that rankings are preserved
        for original, loaded in zip(result.models, loaded_result.models):
            assert original.model_id == loaded.model_id
            assert original.rank == loaded.rank
            assert abs(original.score - loaded.score) < 1e-6


class TestComparisonMetric:
    """Test ComparisonMetric enum"""
    
    def test_metric_values(self):
        """Test that all metrics have expected values"""
        assert ComparisonMetric.ENERGY_EFFICIENCY.value == "energy_efficiency"
        assert ComparisonMetric.CO2_EFFICIENCY.value == "co2_efficiency"
        assert ComparisonMetric.PERFORMANCE.value == "performance"
        assert ComparisonMetric.COST_EFFECTIVENESS.value == "cost_effectiveness"
        assert ComparisonMetric.SPEED.value == "speed"
    
    def test_metric_enumeration(self):
        """Test that we can enumerate all metrics"""
        metrics = list(ComparisonMetric)
        assert len(metrics) == 5
        assert ComparisonMetric.ENERGY_EFFICIENCY in metrics
        assert ComparisonMetric.CO2_EFFICIENCY in metrics
        assert ComparisonMetric.PERFORMANCE in metrics
        assert ComparisonMetric.COST_EFFECTIVENESS in metrics
        assert ComparisonMetric.SPEED in metrics


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_scoring_results(self):
        """Test handling of empty scoring results"""
        comparator = ModelComparator()
        
        with pytest.raises(ComparisonError):
            comparator.compare_models_from_results([])
    
    def test_single_model_comparison(self):
        """Test comparing a single model"""
        scoring_result = ScoringResult(
            model_id="single_model",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 4.0,
                'co2_per_1k_g': 2.0,
                'samples_per_second': 100,
                'duration_seconds': 10
            },
            hardware={'gpu': 'Test GPU'},
            metadata={'timestamp': '2025-10-02'}
        )
        
        comparator = ModelComparator()
        result = comparator.compare_models_from_results([scoring_result])
        
        assert len(result.models) == 1
        assert result.models[0].rank == 1
        assert abs(result.models[0].score - 1.0) < 1e-6  # Single model gets perfect score
    
    def test_models_with_zero_values(self):
        """Test handling of models with zero energy/performance values"""
        scoring_results = [
            ScoringResult(
                model_id="zero_model",
                task="text_generation",
                measurements={
                    'energy_per_1k_wh': 0.0,
                    'co2_per_1k_g': 0.0,
                    'samples_per_second': 0,
                    'duration_seconds': 0
                },
                hardware={'gpu': 'Test GPU'},
                metadata={'timestamp': '2025-10-02'}
            ),
            ScoringResult(
                model_id="normal_model",
                task="text_generation",
                measurements={
                    'energy_per_1k_wh': 4.0,
                    'co2_per_1k_g': 2.0,
                    'samples_per_second': 100,
                    'duration_seconds': 10
                },
                hardware={'gpu': 'Test GPU'},
                metadata={'timestamp': '2025-10-02'}
            )
        ]
        
        comparator = ModelComparator()
        result = comparator.compare_models_from_results(scoring_results)
        
        assert len(result.models) == 2
        # Should not crash with zero values
        for model in result.models:
            assert model.score is not None
