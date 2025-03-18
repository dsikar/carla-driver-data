import json
import os

def load_config(config_path='config.json'):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def save_config(config, config_path='config.json'):
    """
    Save configuration to a JSON file.
    
    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to save the configuration file
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_path}")