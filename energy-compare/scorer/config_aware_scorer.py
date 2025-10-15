"""
Configuration-aware model scorer that uses model profiles from config.yaml.

This module provides a scorer that reads model performance profiles from the
configuration file instead of using hardcoded values.
"""

import logging
import random
from typing import Dict, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path to import config_loader
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config_loader import get_config
from .core import ModelEnergyScorer, ScoringResult

logger = logging.getLogger(__name__)


class ConfigAwareModelScorer(ModelEnergyScorer):
    """
    A model scorer that uses configuration-based model profiles.
    
    This scorer reads model performance characteristics from the config.yaml file
    instead of using hardcoded values, making it easy to update model profiles
    without changing code.
    """
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def score(
        self, 
        model: str,
        task: str,
        n_samples: int = 100,
        runs: int = 3
    ) -> ScoringResult:
        """
        Score a model's energy consumption using configuration-based profiles.
        
        Args:
            model: Model name or identifier
            task: Task type (text_generation, image_classification, etc.)
            n_samples: Number of samples to run inference on
            runs: Number of measurement runs for reliability
            
        Returns:
            ScoringResult with measurements
        """
        self.logger.debug(f"Starting scoring for {model} on {task} (config-aware)")
        
        # Validate inputs
        self._validate_inputs(model, task, n_samples, runs)
        
        # Get model profile from configuration
        profile = self._get_model_profile(model, task)
        
        # Add realistic variation if enabled
        variation_factor = self._get_variation_factor(runs)
        
        # Calculate measurements with variation
        measurements = self._calculate_measurements(profile, variation_factor, runs)
        
        # Get hardware info
        hardware = self._get_hardware_info()
        
        # Create metadata
        metadata = self._create_metadata(profile, n_samples, runs, task)
        
        result = ScoringResult(
            model_id=model,
            task=task,
            measurements=measurements,
            hardware=hardware,
            metadata=metadata
        )
        
        self.logger.debug(f"Scoring complete for {model}")
        return result
    
    def _get_model_profile(self, model: str, task: str) -> Dict:
        """Get model profile from configuration"""
        # Try to determine task category from task name
        task_category = self._get_task_category(task)
        
        # Get profile from configuration
        profile = self.config.get_model_profile(model, task_category)
        
        self.logger.debug(f"Using profile for {model}")
        return profile
    
    def _get_task_category(self, task: str) -> Optional[str]:
        """Determine task category from task name"""
        task_mapping = {
            'text_generation': 'text_generation',
            'text-generation': 'text_generation',
            'text_classification': 'text_classification',
            'text-classification': 'text_classification',
            'sentiment-analysis': 'text_classification',
            'image_classification': 'computer_vision',
            'image-classification': 'computer_vision',
            'object_detection': 'computer_vision',
            'object-detection': 'computer_vision'
        }
        return task_mapping.get(task)
    
    def _get_variation_factor(self, runs: int) -> float:
        """Get variation factor for realistic testing"""
        testing_config = self.config.get_testing_config()
        mock_config = testing_config.get('mock_data', {})
        
        if not mock_config.get('enable_variation', True):
            return 1.0
        
        # Get variation range from config
        variation_range = mock_config.get('variation_factor_range', [0.95, 1.05])
        min_var, max_var = variation_range
        
        # Add variation based on number of runs
        base_variation = random.uniform(min_var, max_var)
        run_variation = (runs - 1) / 10  # Small additional variation based on runs
        return base_variation + run_variation
    
    def _calculate_measurements(self, profile: Dict, variation_factor: float, runs: int) -> Dict:
        """Calculate measurements with variation"""
        # Get default CV from config
        default_cv = self.config.get("scoring.validation.default_coefficient_of_variation", 0.03)
        variation_range = self.config.get("scoring.validation.variation_range", 0.05)
        
        # Calculate base measurements with variation
        energy = profile.get("energy_per_1k_wh", 5.0) * variation_factor
        co2 = profile.get("co2_per_1k_g", 2.5) * variation_factor
        throughput = profile.get("samples_per_second", 50) * variation_factor
        duration = profile.get("duration_seconds", 20.0) / variation_factor
        
        # Calculate statistics
        cv = default_cv + random.uniform(0, variation_range)
        
        return {
            'energy_per_1k_wh': energy,
            'co2_per_1k_g': co2,
            'samples_per_second': throughput,
            'duration_seconds': duration,
            'statistics': {
                'coefficient_of_variation': cv,
                'runs': runs
            }
        }
    
    def _get_hardware_info(self) -> Dict:
        """Get hardware information"""
        # For now, return mock hardware info
        # In a real implementation, this would detect actual hardware
        return {
            'gpu': 'RTX 4090',
            'cpu': 'AMD Ryzen 9 7950X'
        }
    
    def _create_metadata(self, profile: Dict, n_samples: int, runs: int, task: str) -> Dict:
        """Create metadata for the scoring result"""
        return {
            'timestamp': datetime.now().isoformat() + 'Z',
            'model_size_mb': profile.get("size_mb", 100),
            'architecture': profile.get("architecture", "Unknown"),
            'n_samples': n_samples,
            'runs': runs,
            'task_type': task,
            'scorer_type': 'config_aware'
        }
    
    def get_available_models(self, task_category: str = None) -> Dict[str, Dict]:
        """
        Get all available model profiles from configuration.
        
        Args:
            task_category: Optional task category to filter by
            
        Returns:
            Dictionary of model profiles
        """
        model_profiles = self.config.get("model_profiles", {})
        
        if task_category and task_category in model_profiles:
            return model_profiles[task_category]
        
        # Return all models from all categories
        all_models = {}
        for category, models in model_profiles.items():
            if isinstance(models, dict) and category != 'defaults':
                all_models.update(models)
        
        return all_models
    
    def add_model_profile(self, model_id: str, profile: Dict, task_category: str = None):
        """
        Add a new model profile to the configuration.
        
        Note: This modifies the in-memory configuration but doesn't persist to file.
        For persistent changes, update the config.yaml file directly.
        
        Args:
            model_id: Model identifier
            profile: Model performance profile
            task_category: Task category to add to
        """
        if task_category:
            # Add to specific task category
            task_profiles = self.config.get(f"model_profiles.{task_category}", {})
            task_profiles[model_id] = profile
            self.config._set_nested_value(f"model_profiles.{task_category}.{model_id}", profile)
        else:
            # Add to a general category
            self.config._set_nested_value(f"model_profiles.general.{model_id}", profile)
        
        self.logger.debug(f"Added model profile for {model_id}")
    
    def validate_model_profile(self, profile: Dict) -> tuple[bool, Optional[str]]:
        """
        Validate a model profile has required fields.
        
        Args:
            profile: Model profile dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = [
            'energy_per_1k_wh',
            'co2_per_1k_g', 
            'samples_per_second',
            'duration_seconds',
            'size_mb',
            'architecture'
        ]
        
        missing_fields = [field for field in required_fields if field not in profile]
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        # Validate numeric fields are positive
        numeric_fields = ['energy_per_1k_wh', 'co2_per_1k_g', 'samples_per_second', 'duration_seconds', 'size_mb']
        for field in numeric_fields:
            if not isinstance(profile[field], (int, float)) or profile[field] <= 0:
                return False, f"Field '{field}' must be a positive number"
        
        return True, None
