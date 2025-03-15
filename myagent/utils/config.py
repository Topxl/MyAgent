"""
Configuration utilities for MyAgent.
"""
import os
import json
from typing import Dict, Any, Optional


class Config:
    """
    Configuration class for MyAgent.
    Handles loading and saving configuration settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration handler.
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        if config_path is None:
            # Use default path: myagent_root/config/config.json
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_dir = os.path.join(base_dir, "config")
            os.makedirs(config_dir, exist_ok=True)
            self.config_path = os.path.join(config_dir, "config.json")
        else:
            self.config_path = config_path
        
        # Default configuration
        self.default_config = {
            "vad": {
                "threshold": 0.5,
                "sampling_rate": 16000
            },
            "noise_reduction": {
                "use_demucs": False,
                "stationary_threshold": 0.5,
                "n_fft": 2048
            },
            "speaker_id": {
                "similarity_threshold": 0.75,
                "model_dir": None
            },
            "transcription": {
                "model_size": "base",
                "device": "cpu",
                "compute_type": "float32",
                "language": None
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False
            },
            "paths": {
                "data_dir": None,
                "models_dir": None,
                "temp_dir": None
            }
        }
        
        # Current configuration
        self.config = self.load()
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary.
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Merge with default config to ensure all keys exist
                return self._merge_configs(self.default_config, config)
            except Exception as e:
                print(f"Error loading configuration: {e}")
                return self.default_config.copy()
        else:
            return self.default_config.copy()
    
    def save(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key in dot notation (e.g., "vad.threshold").
            default: Default value to return if key not found.
            
        Returns:
            Configuration value or default.
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key in dot notation (e.g., "vad.threshold").
            value: Value to set.
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the correct level
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def _merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            default: Default configuration.
            override: Override configuration.
            
        Returns:
            Merged configuration.
        """
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def reset(self) -> None:
        """Reset configuration to default values."""
        self.config = self.default_config.copy()


# Global config instance
_config_instance = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Optional path to configuration file.
        
    Returns:
        Configuration instance.
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance


def read_config(key: str, default: Any = None) -> Any:
    """
    Read a configuration value.
    
    Args:
        key: Configuration key in dot notation.
        default: Default value if key not found.
        
    Returns:
        Configuration value.
    """
    return get_config().get(key, default)


def save_config() -> bool:
    """
    Save the current configuration.
    
    Returns:
        True if successful, False otherwise.
    """
    return get_config().save()


def set_config(key: str, value: Any) -> None:
    """
    Set a configuration value.
    
    Args:
        key: Configuration key in dot notation.
        value: Value to set.
    """
    get_config().set(key, value)
    
    # Save the configuration
    save_config()
