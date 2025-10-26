"""
Configuration loader for the Energy Score Tool.

This module provides utilities to load and manage configuration from YAML files,
with support for environment variable overrides and validation.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration loading or validation error"""
    pass


class ConfigLoader:
    """
    Loads and manages configuration from YAML files.
    
    Features:
    - Load configuration from YAML files
    - Environment variable overrides
    - Configuration validation
    - Default value fallbacks
    - Nested configuration access
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to configuration file. If None, looks for config.yaml in current directory.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                raise ConfigError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
            
            logger.debug(f"Loaded configuration from {self.config_path}")
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'scoring.star_rating.max_stars')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_profile(self, model_id: str, task_type: str = None) -> Dict[str, Any]:
        """
        Get model profile configuration.
        
        Args:
            model_id: Model identifier
            task_type: Task type (text_generation, computer_vision, etc.)
            
        Returns:
            Model profile dictionary
        """
        # Try to find model in specific task category first
        if task_type:
            task_profiles = self.get(f"model_profiles.{task_type}", {})
            if model_id in task_profiles:
                return task_profiles[model_id]
        
        # Search all task categories
        model_profiles = self.get("model_profiles", {})
        for task_cat, profiles in model_profiles.items():
            if isinstance(profiles, dict) and model_id in profiles:
                return profiles[model_id]
        
        # Return default profile
        return self.get("model_profiles.defaults", {})
    
    def get_metric_weights(self) -> Dict[str, float]:
        """Get metric weights for scoring"""
        # Check if HuggingFace mode is enabled
        if self.get("scoring.huggingface_mode.enabled", False):
            return self.get("scoring.huggingface_mode.metric_weights", {})
        else:
            return self.get("scoring.metric_weights", {})
    
    def is_huggingface_mode(self) -> bool:
        """Check if HuggingFace mode is enabled"""
        return self.get("scoring.huggingface_mode.enabled", False)
    
    def set_huggingface_mode(self, enabled: bool = True):
        """Enable or disable HuggingFace mode"""
        self._set_nested_value("scoring.huggingface_mode.enabled", enabled)
    
    def get_supported_hardware(self) -> Dict[str, str]:
        """Get supported hardware types"""
        return self.get("hardware.supported_types", {})
    
    def get_supported_tasks(self) -> list:
        """Get supported task types"""
        return self.get("tasks.supported", [])
    
    def get_codecarbon_config(self) -> Dict[str, Any]:
        """Get CodeCarbon configuration"""
        return self.get("codecarbon", {})
    
    def get_measurement_config(self) -> Dict[str, Any]:
        """Get measurement configuration"""
        return self.get("measurement", {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.get("output", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get("logging", {})
    
    def get_testing_config(self) -> Dict[str, Any]:
        """Get testing configuration"""
        return self.get("testing", {})
    
    def validate_config(self) -> bool:
        """
        Validate configuration for common issues.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigError: If configuration is invalid
        """
        # Validate metric weights sum to 1.0
        weights = self.get_metric_weights()
        if weights:
            total_weight = sum(weights.values())
            if not abs(total_weight - 1.0) < 1e-6:
                raise ConfigError(f"Metric weights must sum to 1.0, got {total_weight}")
        
        # Validate star rating range
        min_stars = self.get("scoring.star_rating.min_stars", 1.0)
        max_stars = self.get("scoring.star_rating.max_stars", 5.0)
        if min_stars >= max_stars:
            raise ConfigError(f"min_stars ({min_stars}) must be less than max_stars ({max_stars})")
        
        # Validate supported hardware and tasks are not empty
        if not self.get_supported_hardware():
            raise ConfigError("No supported hardware types defined")
        
        if not self.get_supported_tasks():
            raise ConfigError("No supported task types defined")
        
        logger.debug("Configuration validation passed")
        return True
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
        self.validate_config()
        logger.debug("Configuration reloaded")
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self._config.copy()
    
    def update_from_env(self):
        """
        Update configuration from environment variables.
        
        Environment variables should be prefixed with 'ENERGY_SCORE_' and use
        dot notation for nested keys (e.g., ENERGY_SCORE_SCORING_METRIC_WEIGHTS_ENERGY_EFFICIENCY=0.5)
        """
        prefix = "ENERGY_SCORE_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert environment variable name to config key
                # Handle special cases where underscores should be preserved
                remaining = key[len(prefix):].lower()
                
                # Special handling for metric_weights and other compound keys
                if 'metric_weights' in remaining:
                    # Convert ENERGY_SCORE_SCORING_METRIC_WEIGHTS_ENERGY_EFFICIENCY
                    # to scoring.metric_weights.energy_efficiency
                    parts = remaining.split('_')
                    if len(parts) >= 4 and parts[0] == 'scoring' and parts[1] == 'metric' and parts[2] == 'weights':
                        metric_name = '_'.join(parts[3:])
                        config_key = f"scoring.metric_weights.{metric_name}"
                    else:
                        config_key = remaining.replace('_', '.')
                elif 'huggingface_mode' in remaining:
                    # Handle ENERGY_SCORE_SCORING_HUGGINGFACE_MODE_ENABLED
                    # Convert to scoring.huggingface_mode.enabled
                    if remaining == 'scoring_huggingface_mode_enabled':
                        config_key = 'scoring.huggingface_mode.enabled'
                    else:
                        config_key = remaining.replace('_', '.')
                else:
                    config_key = remaining.replace('_', '.')
                
                # Try to convert value to appropriate type
                try:
                    # Try integer
                    if value.isdigit():
                        value = int(value)
                    # Try float
                    elif '.' in value and value.replace('.', '').isdigit():
                        value = float(value)
                    # Try boolean
                    elif value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                except ValueError:
                    pass  # Keep as string
                
                # Set nested value
                self._set_nested_value(config_key, value)
    
    def _set_nested_value(self, key: str, value: Any):
        """Set nested configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value


# Global configuration instance
_config_loader: Optional[ConfigLoader] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> ConfigLoader:
    """
    Get global configuration loader instance.
    
    Args:
        config_path: Path to configuration file (only used on first call)
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
        _config_loader.update_from_env()
        _config_loader.validate_config()
    
    return _config_loader


def reload_config():
    """Reload global configuration"""
    global _config_loader
    if _config_loader is not None:
        _config_loader.reload()


# Convenience functions for common configuration access
def get_model_profile(model_id: str, task_type: str = None) -> Dict[str, Any]:
    """Get model profile for a specific model"""
    return get_config().get_model_profile(model_id, task_type)


def get_metric_weights() -> Dict[str, float]:
    """Get metric weights for scoring"""
    return get_config().get_metric_weights()


def is_huggingface_mode() -> bool:
    """Check if HuggingFace mode is enabled"""
    return get_config().is_huggingface_mode()


def set_huggingface_mode(enabled: bool = True):
    """Enable or disable HuggingFace mode"""
    get_config().set_huggingface_mode(enabled)


def get_supported_hardware() -> Dict[str, str]:
    """Get supported hardware types"""
    return get_config().get_supported_hardware()


def get_supported_tasks() -> list:
    """Get supported task types"""
    return get_config().get_supported_tasks()


def get_codecarbon_config() -> Dict[str, Any]:
    """Get CodeCarbon configuration"""
    return get_config().get_codecarbon_config()


def get_measurement_config() -> Dict[str, Any]:
    """Get measurement configuration"""
    return get_config().get_measurement_config()


def get_output_config() -> Dict[str, Any]:
    """Get output configuration"""
    return get_config().get_output_config()


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration"""
    return get_config().get_logging_config()


def get_testing_config() -> Dict[str, Any]:
    """Get testing configuration"""
    return get_config().get_testing_config()
