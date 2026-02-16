"""
Configuration management for training and inference.
Supports YAML and JSON configuration files.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Configuration class for deepfake detection pipeline.

    All parameters are documented and have sensible defaults.
    """
    # Model Configuration
    model_name: str = 'efficientnet-b1'
    num_classes: int = 2
    dropout_rate: float = 0.5
    pretrained: bool = True

    # Training Configuration
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 8e-4
    weight_decay: float = 1e-3
    warmup_epochs: int = 5

    # Data Configuration
    image_size: int = 240
    num_workers: int = 4
    pin_memory: bool = True
    use_heavy_augmentation: bool = False

    # Paths
    train_real_dirs: list = field(default_factory=list)
    train_fake_dirs: list = field(default_factory=list)
    val_real_dirs: list = field(default_factory=list)
    val_fake_dirs: list = field(default_factory=list)
    test_real_dirs: list = field(default_factory=list)
    test_fake_dirs: list = field(default_factory=list)

    # Checkpoint Configuration
    checkpoint_dir: str = 'checkpoints'
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 5

    # Logging Configuration
    log_dir: str = 'logs'
    results_dir: str = 'results'
    experiment_name: str = 'deepfake_detection'

    # Device Configuration
    device: str = 'cuda'
    mixed_precision: bool = True

    # Evaluation Configuration
    test_batch_size: int = 100
    eer_grid_density: int = 10000

    # Resume Training
    resume_from_checkpoint: Optional[str] = None
    start_epoch: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"learning_rate must be in (0, 1), got {self.learning_rate}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save configuration to file.

        Args:
            filepath: Path to save configuration (supports .json and .yaml)

        Example:
            >>> config = Config()
            >>> config.save('config.yaml')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        if filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create Config from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            Config instance

        Example:
            >>> config_dict = {'model_name': 'efficientnet-b0', 'batch_size': 64}
            >>> config = Config.from_dict(config_dict)
        """
        return cls(**config_dict)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Config':
        """
        Load configuration from file.

        Args:
            filepath: Path to configuration file

        Returns:
            Config instance

        Example:
            >>> config = Config.load('config.yaml')
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        if filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        logger.info(f"Configuration loaded from {filepath}")
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        """String representation of configuration."""
        lines = ["Configuration:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def load_config(filepath: Union[str, Path]) -> Config:
    """
    Load configuration from file.

    Args:
        filepath: Path to configuration file

    Returns:
        Config instance

    Example:
        >>> config = load_config('config.yaml')
    """
    return Config.load(filepath)


def save_config(config: Config, filepath: Union[str, Path]) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration instance
        filepath: Path to save configuration

    Example:
        >>> config = Config()
        >>> save_config(config, 'config.yaml')
    """
    config.save(filepath)


def create_default_config(filepath: Union[str, Path]) -> Config:
    """
    Create and save a default configuration file.

    Args:
        filepath: Path to save default configuration

    Returns:
        Default Config instance

    Example:
        >>> config = create_default_config('default_config.yaml')
    """
    config = Config()
    config.save(filepath)
    logger.info(f"Default configuration created at {filepath}")
    return config
