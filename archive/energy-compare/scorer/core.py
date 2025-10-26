# scorer/core.py
import logging
from typing import Dict, Optional
from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path to import config_loader
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config_loader import get_config

logger = logging.getLogger(__name__)

# Get supported tasks from configuration
config = get_config()
SUPPORTED_TASKS = config.get_supported_tasks()

@dataclass
class ScoringResult:
    """Result of energy scoring"""
    model_id: str
    task: str
    measurements: Dict
    hardware: Dict
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "task": self.task,
            "measurements": self.measurements,
            "hardware": self.hardware,
            "metadata": self.metadata
        }
    
    def is_valid(self) -> tuple[bool, Optional[str]]:
        """Check if measurements are reliable"""
        max_cv = config.get("scoring.validation.max_coefficient_of_variation", 0.15)
        cv = self.measurements.get('statistics', {}).get('coefficient_of_variation', 0)
        if cv > max_cv:
            return False, f"High variance in measurements (CV={cv:.2%})"
        return True, None


class ValidationError(Exception):
    """Input validation failed"""
    pass


class MeasurementError(Exception):
    """Energy measurement failed"""
    pass


class ModelEnergyScorer:
    """
    Measures energy consumption of ML models during inference.
    
    Example:
        scorer = ModelEnergyScorer()
        result = scorer.score(
            model="gpt2",
            task="text_generation"
        )
    """
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def score(
        self, 
        model: str,
        task: str,
        n_samples: int = 100,
        runs: int = 3
    ) -> ScoringResult:
        """
        Score a model's energy consumption.
        
        Args:
            model: Model name or identifier
            task: Task type (text_generation, image_classification, etc.)
            n_samples: Number of samples to run inference on
            runs: Number of measurement runs for reliability
            
        Returns:
            ScoringResult with measurements
        """
        self.logger.info(f"Starting scoring for {model} on {task}")
        
        # Validate inputs
        self._validate_inputs(model, task, n_samples, runs)
        
        # For now, return a mock result
        # We'll implement real measurement later
        result = ScoringResult(
            model_id=model,
            task=task,
            measurements={
                'energy_per_1k_wh': 4.5,
                'co2_per_1k_g': 21.0,
                'statistics': {
                    'coefficient_of_variation': 0.08,
                    'runs': runs
                }
            },
            hardware={'gpu': 'Mock GPU'},
            metadata={'timestamp': datetime.now().isoformat()}
        )
        
        self.logger.info("Scoring complete")
        return result
    
    def _validate_inputs(self, model: str, task: str, n_samples: int = None, runs: int = None):
        """Validate inputs before scoring"""
        # Validate inputs
        
        if task not in SUPPORTED_TASKS:
            raise ValidationError(
                f"Task '{task}' not supported. "
                f"Supported tasks: {', '.join(SUPPORTED_TASKS)}"
            )
        
        if not model or not isinstance(model, str):
            raise ValidationError(f"Invalid model identifier: {model}")
        
        if n_samples is not None and n_samples <= 0:
            raise ValidationError(f"n_samples must be positive, got: {n_samples}")
        
        if runs is not None and runs <= 0:
            raise ValidationError(f"runs must be positive, got: {runs}")
        
        # Input validation passed
