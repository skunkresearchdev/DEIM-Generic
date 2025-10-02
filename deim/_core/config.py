"""
Configuration management for DEIM
Handles loading configs for 'under', 'sides', or custom YAML files
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import copy


class ConfigManager:
    """
    Manages configuration for DEIM models

    Handles:
    - Loading predefined configs ('under', 'sides')
    - Loading custom YAML configs
    - Applying parameter overrides
    - Dataset path resolution
    """

    def __init__(self, config: str):
        """
        Initialize config manager

        Args:
            config: 'under', 'sides', or path to custom YAML
        """
        self.config_type = config
        self.base_config = self._load_base_config(config)
        self.config = copy.deepcopy(self.base_config)

    def _load_base_config(self, config: str) -> Dict[str, Any]:
        """Load the base configuration"""

        # Get config directory
        config_dir = Path(__file__).parent.parent / "_configs"

        if config == 'under':
            config_path = config_dir / "under.yml"
            dataset_name = "yolo_dataset_under"

        elif config == 'sides':
            config_path = config_dir / "sides.yml"
            dataset_name = "yolo_dataset_sides"

        else:
            # Custom config path
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            dataset_name = None

        # Load YAML config
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # Handle includes if present
        if "__include__" in cfg:
            includes = cfg.pop("__include__")
            base_cfg = {}

            for include_path in includes:
                # Resolve include path relative to config file
                if not Path(include_path).is_absolute():
                    include_path = config_path.parent / include_path

                # Check if the path exists
                if include_path.exists():
                    with open(include_path, 'r') as f:
                        include_cfg = yaml.safe_load(f)
                        if include_cfg:
                            base_cfg = self._merge_configs(base_cfg, include_cfg)
                else:
                    print(f"Warning: Include file not found: {include_path}")

            # Merge with main config
            cfg = self._merge_configs(base_cfg, cfg)

        # Set dataset path if using predefined configs
        if dataset_name:
            dataset_base = Path("/home/hidara/Documents/datasets")
            cfg['dataset_path'] = str(dataset_base / dataset_name)

            # Update dataset config paths
            if 'Dataset' in cfg:
                cfg['Dataset']['img_folder'] = str(dataset_base / dataset_name)
                cfg['Dataset']['ann_file'] = str(dataset_base / dataset_name)

            # Update train/val dataset paths
            if 'Train' in cfg:
                if 'Dataset' in cfg['Train']:
                    cfg['Train']['Dataset']['img_folder'] = str(dataset_base / dataset_name / "train")
                    cfg['Train']['Dataset']['ann_file'] = str(dataset_base / dataset_name / "train")

            if 'Eval' in cfg:
                if 'Dataset' in cfg['Eval']:
                    cfg['Eval']['Dataset']['img_folder'] = str(dataset_base / dataset_name / "val")
                    cfg['Eval']['Dataset']['ann_file'] = str(dataset_base / dataset_name / "val")

        return cfg

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two config dictionaries"""
        result = copy.deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def apply_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply parameter overrides to config

        Args:
            overrides: Dictionary of parameters to override

        Returns:
            Updated config dictionary
        """
        self.config = copy.deepcopy(self.base_config)

        # Map common parameter names to config structure
        # Note: The original config uses 'epoches' (with an 'e')
        param_mapping = {
            'epochs': 'epoches',  # Map to 'epoches' as used in original configs
            'batch_size': 'train_dataloader.total_batch_size',
            'learning_rate': 'optimizer.lr',
            'dataset_path': 'dataset_path',
            'output_dir': 'output_dir',
        }

        for key, value in overrides.items():
            if key in param_mapping:
                # Use mapped path
                path = param_mapping[key]
                self._set_nested(self.config, path, value)
            else:
                # Direct assignment
                self.config[key] = value

        return self.config

    def _set_nested(self, cfg: Dict[str, Any], path: str, value: Any):
        """Set a nested config value using dot notation"""
        keys = path.split('.')
        current = cfg

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config

    def save_config(self, path: str):
        """Save current configuration to file"""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)