# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:25:25 2026

@author: ngozi
"""

"""
Utility functions for logging, configuration loading, and common helpers.
"""

import logging
import yaml
from pathlib import Path
from datetime import datetime


def setup_logging(config):
    """
    Set up logging based on configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with logging settings
    """
    log_dir = Path(config['output']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_config(config_path="config/config.yaml"):
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config):
    """
    Validate that required configuration keys exist.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Raises
    ------
    ValueError
        If required keys are missing
    """
    required_keys = ['data', 'forecast', 'models', 'compute', 'output']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Check data paths exist
    for path_key in ['sales_path', 'calendar_path', 'prices_path']:
        path = Path(config['data'][path_key])
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")