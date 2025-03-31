"""Configuration settings for SimNexus."""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


def get_config_dir() -> Path:
    """
    Get the directory for storing configuration files.
    
    Returns:
        Path: The configuration directory path
    """
    if os.name == 'nt':  # Windows
        app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
        config_dir = Path(app_data) / "SimNexus"
    else:  # macOS, Linux, etc.
        config_dir = Path.home() / ".config" / "simnexus"
    
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration settings.
    
    Returns:
        Dict[str, Any]: Default configuration settings
    """
    return {
        "general": {
            "default_output_dir": str(Path.home() / "simnexus_results"),
            "random_seed": None,
            "visualization_backend": "matplotlib",
        },
        "stock_market": {
            "default_days": 365,
            "default_volatility": 0.02,
            "default_drift": 0.001,
            "default_start_price": 100.0,
        },
        "resource_fluctuations": {
            "default_days": 365,
            "default_volatility": 0.03,
            "default_drift": 0.001,
            "default_start_price": 100.0,
        },
        "product_popularity": {
            "default_days": 365,
            "default_initial_popularity": 0.1,
            "default_virality_factor": 0.1,
            "default_marketing_effectiveness": 0.05,
        }
    }


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_file: Optional path to the configuration file
        
    Returns:
        Dict[str, Any]: The loaded configuration
    """
    if config_file is None:
        config_path = get_config_dir() / "config.json"
    else:
        config_path = Path(config_file)
    
    # Create default config if it doesn't exist
    if not config_path.exists():
        default_config = get_default_config()
        save_config(default_config, str(config_path))
        return default_config
    
    # Load existing config
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        print(f"Error loading config from {config_path}, using defaults")
        return get_default_config()


def save_config(config: Dict[str, Any], config_file: Optional[str] = None) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: The configuration to save
        config_file: Optional path to the configuration file
    """
    if config_file is None:
        config_path = get_config_dir() / "config.json"
    else:
        config_path = Path(config_file)
    
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save config
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except IOError:
        print(f"Error saving config to {config_path}")