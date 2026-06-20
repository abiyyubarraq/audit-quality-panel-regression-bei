"""
Configuration Management Module

Loads and manages configuration from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """
    Load and manage configuration settings

    Usage:
        config = ConfigManager()
        params = config.load_config('config/default_config.yaml')
        alpha = params['statistical_tests']['significance_level']
    """

    def __init__(self, config_dir: str = 'config'):
        """
        Initialize config manager

        Args:
            config_dir: Directory containing config files (default: 'config')
        """
        self.config_dir = Path(config_dir)

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Args:
            config_file: Path to config file (relative to config_dir or absolute)

        Returns:
            Dictionary with configuration settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = Path(config_file)

        # If relative path, prepend config_dir
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {str(e)}")

    def get_test_params(self, config_file: str = 'default_config.yaml') -> Dict[str, Any]:
        """
        Get statistical test parameters from config

        Args:
            config_file: Config file name (default: 'default_config.yaml')

        Returns:
            Dictionary with test parameters
        """
        config = self.load_config(config_file)
        return config.get('statistical_tests', {})

    def get_panel_params(self, config_file: str = 'default_config.yaml') -> Dict[str, Any]:
        """
        Get panel regression parameters from config

        Args:
            config_file: Config file name (default: 'default_config.yaml')

        Returns:
            Dictionary with panel regression parameters
        """
        config = self.load_config(config_file)
        return config.get('panel_regression', {})

    def get_test_config(self, config_file: str = 'default_config.yaml') -> Dict[str, Any]:
        """
        Get full test configuration

        Args:
            config_file: Config file name (default: 'default_config.yaml')

        Returns:
            Dictionary with complete configuration
        """
        return self.load_config(config_file)

    def get_variable_info(self, variable_file: str = 'variables.yaml') -> Dict[str, Any]:
        """
        Get variable definitions from config

        Args:
            variable_file: Variable config file name (default: 'variables.yaml')

        Returns:
            Dictionary with variable information
        """
        return self.load_config(variable_file)


__all__ = ['ConfigManager']
