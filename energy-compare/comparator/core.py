# comparator/core.py
"""
Core comparison functionality for energy efficiency analysis.

This module provides tools to compare multiple ML models based on their
energy consumption, CO2 emissions, and performance metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import statistics
from enum import Enum

from scorer.core import ScoringResult, ModelEnergyScorer

logger = logging.getLogger(__name__)


class ComparisonMetric(Enum):
    """Available comparison metrics"""
    ENERGY_EFFICIENCY = "energy_efficiency"  # Lower is better
    CO2_EFFICIENCY = "co2_efficiency"        # Lower is better
    PERFORMANCE = "performance"               # Higher is better
    COST_EFFECTIVENESS = "cost_effectiveness" # Higher is better
    SPEED = "speed"                          # Higher is better


@dataclass
class ModelComparison:
    """Represents a single model in a comparison"""
    model_id: str
    task: str
    scoring_result: ScoringResult
    rank: Optional[int] = None
    score: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "model_id": self.model_id,
            "task": self.task,
            "rank": self.rank,
            "score": self.score,
            "measurements": self.scoring_result.measurements,
            "hardware": self.scoring_result.hardware,
            "metadata": {**self.metadata, **self.scoring_result.metadata}
        }


@dataclass
class ComparisonResult:
    """Result of comparing multiple models"""
    task: str
    models: List[ModelComparison]
    comparison_metrics: List[ComparisonMetric]
    comparison_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    summary: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "task": self.task,
            "models": [model.to_dict() for model in self.models],
            "comparison_metrics": [metric.value for metric in self.comparison_metrics],
            "comparison_timestamp": self.comparison_timestamp,
            "summary": self.summary
        }
    
    def get_winner(self) -> Optional[ModelComparison]:
        """Get the best performing model (rank 1)"""
        for model in self.models:
            if model.rank == 1:
                return model
        return None
    
    def get_rankings(self) -> List[ModelComparison]:
        """Get models sorted by rank"""
        return sorted(self.models, key=lambda x: x.rank or float('inf'))


class ComparisonError(Exception):
    """Error during model comparison"""
    pass


class ModelComparator:
    """
    Compares multiple ML models based on energy efficiency and performance.
    
    Example:
        comparator = ModelComparator()
        result = comparator.compare_models([
            ("gpt2", "text_generation"),
            ("gpt2-medium", "text_generation"),
            ("distilgpt2", "text_generation")
        ])
    """
    
    def __init__(self, scorer: Optional[ModelEnergyScorer] = None):
        """
        Initialize the comparator.
        
        Args:
            scorer: Optional ModelEnergyScorer instance. If None, creates a new one.
        """
        self.scorer = scorer or ModelEnergyScorer()
        self.logger = logging.getLogger(__name__)
        
        # Default comparison weights
        self.metric_weights = {
            ComparisonMetric.ENERGY_EFFICIENCY: 0.4,
            ComparisonMetric.CO2_EFFICIENCY: 0.3,
            ComparisonMetric.PERFORMANCE: 0.2,
            ComparisonMetric.SPEED: 0.1
        }
    
    def compare_models(
        self,
        model_specs: List[Tuple[str, str]],
        metrics: Optional[List[ComparisonMetric]] = None,
        n_samples: int = 100,
        runs: int = 3,
        custom_weights: Optional[Dict[ComparisonMetric, float]] = None
    ) -> ComparisonResult:
        """
        Compare multiple models on the same task.
        
        Args:
            model_specs: List of (model_id, task) tuples
            metrics: List of metrics to compare on. If None, uses all available.
            n_samples: Number of samples for energy measurement
            runs: Number of measurement runs
            custom_weights: Custom weights for metrics (must sum to 1.0)
            
        Returns:
            ComparisonResult with rankings and scores
        """
        self.logger.info(f"Starting comparison of {len(model_specs)} models")
        
        # Validate inputs
        self._validate_model_specs(model_specs)
        
        # Use default metrics if none specified
        if metrics is None:
            metrics = list(ComparisonMetric)
        
        # Validate and set weights
        if custom_weights:
            self._validate_weights(custom_weights)
            self.metric_weights = custom_weights
        
        # Score all models
        model_comparisons = []
        for model_id, task in model_specs:
            self.logger.info(f"Scoring model: {model_id}")
            try:
                scoring_result = self.scorer.score(
                    model=model_id,
                    task=task,
                    n_samples=n_samples,
                    runs=runs
                )
                
                model_comparison = ModelComparison(
                    model_id=model_id,
                    task=task,
                    scoring_result=scoring_result
                )
                model_comparisons.append(model_comparison)
                
            except Exception as e:
                self.logger.error(f"Failed to score {model_id}: {e}")
                raise ComparisonError(f"Failed to score model {model_id}: {e}")
        
        # Calculate comparison scores and rankings
        self._calculate_scores(model_comparisons, metrics)
        self._calculate_rankings(model_comparisons)
        
        # Generate summary
        summary = self._generate_summary(model_comparisons, metrics)
        
        result = ComparisonResult(
            task=model_specs[0][1],  # All models should have same task
            models=model_comparisons,
            comparison_metrics=metrics,
            summary=summary
        )
        
        self.logger.info("Comparison complete")
        return result
    
    def compare_models_from_results(
        self,
        scoring_results: List[ScoringResult],
        metrics: Optional[List[ComparisonMetric]] = None,
        custom_weights: Optional[Dict[ComparisonMetric, float]] = None
    ) -> ComparisonResult:
        """
        Compare models using pre-computed scoring results.
        
        Args:
            scoring_results: List of ScoringResult objects
            metrics: List of metrics to compare on
            custom_weights: Custom weights for metrics
            
        Returns:
            ComparisonResult with rankings and scores
        """
        self.logger.info(f"Comparing {len(scoring_results)} pre-computed results")
        
        if not scoring_results:
            raise ComparisonError("No scoring results provided")
        
        # Validate all results are for the same task
        task = scoring_results[0].task
        for result in scoring_results[1:]:
            if result.task != task:
                raise ComparisonError("All models must be compared on the same task")
        
        # Use default metrics if none specified
        if metrics is None:
            metrics = list(ComparisonMetric)
        
        # Validate and set weights
        if custom_weights:
            self._validate_weights(custom_weights)
            self.metric_weights = custom_weights
        
        # Create model comparisons
        model_comparisons = []
        for scoring_result in scoring_results:
            model_comparison = ModelComparison(
                model_id=scoring_result.model_id,
                task=scoring_result.task,
                scoring_result=scoring_result
            )
            model_comparisons.append(model_comparison)
        
        # Calculate scores and rankings
        self._calculate_scores(model_comparisons, metrics)
        self._calculate_rankings(model_comparisons)
        
        # Generate summary
        summary = self._generate_summary(model_comparisons, metrics)
        
        result = ComparisonResult(
            task=task,
            models=model_comparisons,
            comparison_metrics=metrics,
            summary=summary
        )
        
        self.logger.info("Comparison from results complete")
        return result
    
    def _validate_model_specs(self, model_specs: List[Tuple[str, str]]):
        """Validate model specifications"""
        if not model_specs:
            raise ComparisonError("No model specifications provided")
        
        # Check all models use the same task
        if len(set(spec[1] for spec in model_specs)) > 1:
            raise ComparisonError("All models must be compared on the same task")
        
        # Check for duplicate models
        model_ids = [spec[0] for spec in model_specs]
        if len(model_ids) != len(set(model_ids)):
            raise ComparisonError("Duplicate model IDs found")
    
    def _validate_weights(self, weights: Dict[ComparisonMetric, float]):
        """Validate that weights sum to 1.0"""
        total_weight = sum(weights.values())
        if not abs(total_weight - 1.0) < 1e-6:
            raise ComparisonError(f"Weights must sum to 1.0, got {total_weight}")
    
    def _calculate_scores(
        self, 
        model_comparisons: List[ModelComparison], 
        metrics: List[ComparisonMetric]
    ):
        """Calculate composite scores for each model (1-5 star scale)"""
        for model in model_comparisons:
            score = 0.0
            total_weight = 0.0
            
            for metric in metrics:
                metric_score = self._calculate_metric_score(model, metric, model_comparisons)
                weight = self.metric_weights.get(metric, 0.0)
                score += metric_score * weight
                total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                score = score / total_weight
            
            # Ensure score is in 1-5 star range
            model.score = max(1.0, min(5.0, score))
    
    def _calculate_metric_score(
        self, 
        model: ModelComparison, 
        metric: ComparisonMetric, 
        all_models: List[ModelComparison]
    ) -> float:
        """Calculate normalized score for a specific metric"""
        measurements = model.scoring_result.measurements
        
        if metric == ComparisonMetric.ENERGY_EFFICIENCY:
            # Lower energy is better - invert and normalize
            energy = measurements.get('energy_per_1k_wh', 0)
            all_energies = [m.scoring_result.measurements.get('energy_per_1k_wh', 0) for m in all_models]
            return self._normalize_lower_better(energy, all_energies)
        
        elif metric == ComparisonMetric.CO2_EFFICIENCY:
            # Lower CO2 is better - invert and normalize
            co2 = measurements.get('co2_per_1k_g', 0)
            all_co2 = [m.scoring_result.measurements.get('co2_per_1k_g', 0) for m in all_models]
            return self._normalize_lower_better(co2, all_co2)
        
        elif metric == ComparisonMetric.PERFORMANCE:
            # Higher performance is better - use throughput as proxy
            throughput = measurements.get('samples_per_second', 1)
            all_throughputs = [m.scoring_result.measurements.get('samples_per_second', 1) for m in all_models]
            return self._normalize_higher_better(throughput, all_throughputs)
        
        elif metric == ComparisonMetric.SPEED:
            # Higher speed is better - use inverse of duration
            duration = measurements.get('duration_seconds', 1)
            speed = 1.0 / duration if duration > 0 else 0
            all_durations = [m.scoring_result.measurements.get('duration_seconds', 1) for m in all_models]
            all_speeds = [1.0 / d if d > 0 else 0 for d in all_durations]
            return self._normalize_higher_better(speed, all_speeds)
        
        elif metric == ComparisonMetric.COST_EFFECTIVENESS:
            # Higher cost-effectiveness is better - energy per performance
            energy = measurements.get('energy_per_1k_wh', 1)
            throughput = measurements.get('samples_per_second', 1)
            cost_effectiveness = throughput / energy if energy > 0 else 0
            all_energies = [m.scoring_result.measurements.get('energy_per_1k_wh', 1) for m in all_models]
            all_throughputs = [m.scoring_result.measurements.get('samples_per_second', 1) for m in all_models]
            all_cost_effectiveness = [t/e if e > 0 else 0 for t, e in zip(all_throughputs, all_energies)]
            return self._normalize_higher_better(cost_effectiveness, all_cost_effectiveness)
        
        return 0.0
    
    def _normalize_lower_better(self, value: float, all_values: List[float]) -> float:
        """Normalize values where lower is better (1-5 star scale, 5 is best)"""
        if not all_values:
            return 3.0  # Default neutral score (3 stars)
        
        # Handle case where all values are 0 or identical
        if all(v == 0 for v in all_values) or len(set(all_values)) == 1:
            return 3.0  # All models get neutral score (3 stars)
        
        max_val = max(all_values)
        min_val = min(all_values)
        
        # Invert so lower values get higher scores, then scale to 1-5
        normalized = (max_val - value) / (max_val - min_val)
        star_score = 1.0 + (normalized * 4.0)  # Scale from 0-1 to 1-5
        return max(1.0, min(5.0, star_score))  # Ensure 1-5 star range
    
    def _normalize_higher_better(self, value: float, all_values: List[float]) -> float:
        """Normalize values where higher is better (1-5 star scale, 5 is best)"""
        if not all_values:
            return 3.0  # Default neutral score (3 stars)
        
        # Handle case where all values are 0 or identical
        if all(v == 0 for v in all_values) or len(set(all_values)) == 1:
            return 3.0  # All models get neutral score (3 stars)
        
        max_val = max(all_values)
        min_val = min(all_values)
        
        normalized = (value - min_val) / (max_val - min_val)
        star_score = 1.0 + (normalized * 4.0)  # Scale from 0-1 to 1-5
        return max(1.0, min(5.0, star_score))  # Ensure 1-5 star range
    
    def _calculate_rankings(self, model_comparisons: List[ModelComparison]):
        """Calculate rankings based on composite scores"""
        # Sort by score (descending)
        sorted_models = sorted(model_comparisons, key=lambda x: x.score or 0, reverse=True)
        
        # Assign ranks
        for rank, model in enumerate(sorted_models, 1):
            model.rank = rank
    
    def _generate_summary(self, model_comparisons: List[ModelComparison], metrics: List[ComparisonMetric]) -> Dict:
        """Generate comparison summary statistics"""
        if not model_comparisons:
            return {}
        
        scores = [m.score for m in model_comparisons if m.score is not None]
        
        summary = {
            "total_models": len(model_comparisons),
            "metrics_used": [m.value for m in metrics],
            "scoring_system": "1-5 stars (5 = best)",
            "score_statistics": {
                "mean": round(statistics.mean(scores), 2) if scores else 0,
                "median": round(statistics.median(scores), 2) if scores else 0,
                "std": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0,
                "min": round(min(scores), 2) if scores else 0,
                "max": round(max(scores), 2) if scores else 0
            },
            "winner": model_comparisons[0].model_id if model_comparisons else None,
            "winner_stars": round(model_comparisons[0].score, 1) if model_comparisons else 0,
            "energy_range": {
                "min_kwh": min(m.scoring_result.measurements.get('energy_per_1k_wh', 0) for m in model_comparisons),
                "max_kwh": max(m.scoring_result.measurements.get('energy_per_1k_wh', 0) for m in model_comparisons)
            },
            "co2_range": {
                "min_kg": min(m.scoring_result.measurements.get('co2_per_1k_g', 0) for m in model_comparisons),
                "max_kg": max(m.scoring_result.measurements.get('co2_per_1k_g', 0) for m in model_comparisons)
            }
        }
        
        return summary
    
    @staticmethod
    def format_star_rating(score: float) -> str:
        """Format a score as a star rating (e.g., '4.2 ⭐')"""
        if score is None:
            return "N/A"
        
        # Round to 1 decimal place
        rounded_score = round(score, 1)
        
        # Create star representation
        full_stars = int(rounded_score)
        has_half_star = (rounded_score - full_stars) >= 0.5
        
        stars = "⭐" * full_stars
        if has_half_star:
            stars += "⭐"  # Add one more star for half
        
        return f"{rounded_score} {stars}"
    
    def save_comparison(self, result: ComparisonResult, output_path: Union[str, Path]):
        """Save comparison result to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        self.logger.info(f"Comparison saved to {output_path}")
    
    def load_comparison(self, input_path: Union[str, Path]) -> ComparisonResult:
        """Load comparison result from JSON file"""
        input_path = Path(input_path)
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct ComparisonResult from dict
        models = []
        for model_data in data['models']:
            scoring_result = ScoringResult(
                model_id=model_data['model_id'],
                task=model_data['task'],
                measurements=model_data['measurements'],
                hardware=model_data['hardware'],
                metadata=model_data['metadata']
            )
            
            model_comparison = ModelComparison(
                model_id=model_data['model_id'],
                task=model_data['task'],
                scoring_result=scoring_result,
                rank=model_data.get('rank'),
                score=model_data.get('score'),
                metadata=model_data.get('metadata', {})
            )
            models.append(model_comparison)
        
        metrics = [ComparisonMetric(m) for m in data['comparison_metrics']]
        
        result = ComparisonResult(
            task=data['task'],
            models=models,
            comparison_metrics=metrics,
            comparison_timestamp=data['comparison_timestamp'],
            summary=data['summary']
        )
        
        return result
