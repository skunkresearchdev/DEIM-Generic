"""
DEIM API - Simple interface for training and inference
Similar to ultralytics YOLO but for DEIM models
"""

import os
import sys
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import datetime
import torch

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from _core.config import ConfigManager
from _core.trainer import Trainer
from _core.predictor import Predictor


class DEIM:
    """
    DEIM Model API - Simple interface for training and inference

    Args:
        config: Configuration name ('under', 'sides') or path to custom YAML
        device: CUDA device (default: 'cuda:0', always GPU)

    Examples:
        >>> # Training from scratch
        >>> model = DEIM(config='under')
        >>> model.train(epochs=100, batch_size=32)

        >>> # Training with pretrained weights
        >>> model = DEIM(config='sides')
        >>> model.train(pretrained='base_model.pth', epochs=50)

        >>> # Inference
        >>> model = DEIM(config='under')
        >>> model.load('deim_outputs/under/20241002_143022/best_stg1.pth')
        >>> results = model.predict('image.jpg', visualize=True)
    """

    def __init__(self, config: str = 'under', device: str = 'cuda:0'):
        """Initialize DEIM model with config"""

        # Always use GPU
        if not torch.cuda.is_available():
            raise RuntimeError("DEIM requires GPU. No GPU detected!")

        self.device = torch.device(device)
        self.config_name = config

        # Initialize config manager
        self.config_manager = ConfigManager(config)
        self.cfg = self.config_manager.get_config()

        # Model and predictor (lazy initialization)
        self.model = None
        self.predictor = None
        self.trainer = None

        print(f"✓ DEIM initialized with config: {config}")
        print(f"  Device: {self.device}")

    def train(self,
              pretrained: Optional[str] = None,
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None,
              learning_rate: Optional[float] = None,
              dataset_path: Optional[str] = None,
              output_dir: Optional[str] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train the DEIM model

        Args:
            pretrained: Path to pretrained weights or None for training from scratch
            epochs: Number of training epochs (overrides config)
            batch_size: Batch size (overrides config)
            learning_rate: Learning rate (overrides config)
            dataset_path: Custom dataset path (overrides config)
            output_dir: Custom output directory (overrides config)
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training results and output paths

        Examples:
            >>> model = DEIM(config='under')
            >>> # Train from scratch
            >>> model.train(epochs=100, batch_size=32)

            >>> # Train with pretrained weights
            >>> model.train(pretrained='base_model.pth', epochs=50)

            >>> # Custom dataset
            >>> model.train(dataset_path='/path/to/dataset', epochs=100)
        """

        print("\n" + "="*60)
        print("DEIM TRAINING")
        print("="*60)

        # Update config with overrides
        overrides = {}
        if epochs is not None:
            overrides['epochs'] = epochs
        if batch_size is not None:
            overrides['batch_size'] = batch_size
        if learning_rate is not None:
            overrides['learning_rate'] = learning_rate
        if dataset_path is not None:
            overrides['dataset_path'] = dataset_path
        if output_dir is not None:
            overrides['output_dir'] = output_dir

        # Add any additional kwargs
        overrides.update(kwargs)

        # Apply overrides
        if overrides:
            self.cfg = self.config_manager.apply_overrides(overrides)

        # Initialize trainer
        self.trainer = Trainer(
            config=self.cfg,
            device=self.device,
            pretrained=pretrained
        )

        # Start training
        print(f"\n📊 Training Configuration:")
        print(f"  Config: {self.config_name}")
        print(f"  Pretrained: {pretrained if pretrained else 'None (from scratch)'}")
        print(f"  Epochs: {self.cfg.get('epochs', 100)}")
        print(f"  Batch Size: {self.cfg.get('batch_size', 32)}")
        print(f"  Learning Rate: {self.cfg.get('learning_rate', 0.001)}")

        # Determine output base directory
        # NOTE: Do NOT add timestamp here - the training engine (yaml_config.py)
        # automatically appends a timestamp to output_dir. If we add one here too,
        # we get nested timestamps like: deim_outputs/under/20251002_215916/20251002_215921/
        if 'output_dir' in self.cfg:
            # Use output_dir from config (e.g., 'deim_outputs/under')
            output_dir = self.cfg['output_dir']
        else:
            # Use default structure based on config name
            if self.config_name in ['under', 'sides']:
                output_dir = f"deim_outputs/{self.config_name}"
            else:
                output_dir = "deim_outputs/custom"

        # The training engine will create: {output_dir}/{timestamp}/
        # e.g., deim_outputs/under/20251003_123045/
        print(f"  Output Base: {output_dir}")

        print("\n⚡ Starting training...")
        print("  Note: This will take time. Monitor logs for progress.")

        # Run training
        results = self.trainer.train(output_dir=output_dir)

        print(f"\n✅ Training complete!")
        print(f"  Models saved to: {output_dir}/<timestamp>/")
        print(f"  Note: Training engine creates timestamped subdirectory automatically")

        return results

    def load(self, checkpoint_path: str):
        """
        Load a trained model from checkpoint

        Args:
            checkpoint_path: Path to .pth checkpoint file

        Examples:
            >>> model = DEIM(config='under')
            >>> model.load('deim_outputs/under/20241002_143022/best_stg1.pth')
        """

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"📥 Loading model from: {checkpoint_path}")

        # Initialize predictor with loaded weights
        self.predictor = Predictor(
            config=self.cfg,
            checkpoint_path=checkpoint_path,
            device=self.device
        )

        print(f"✓ Model loaded successfully")

    def predict(self,
                source: Union[str, List[str]],
                conf_threshold: float = 0.4,
                visualize: bool = False,
                save_path: Optional[str] = None,
                save_dir: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """
        Run inference on images or videos

        Args:
            source: Image path, video path, list of paths, or directory
            conf_threshold: Confidence threshold for detections
            visualize: Whether to visualize detections
            save_path: Path to save single output (for single image/video)
            save_dir: Directory to save batch outputs

        Returns:
            Detection results as dictionary or list of dictionaries

        Examples:
            >>> # Single image
            >>> results = model.predict('image.jpg', visualize=True)

            >>> # Multiple images
            >>> results = model.predict(['img1.jpg', 'img2.jpg'])

            >>> # Directory
            >>> results = model.predict('path/to/images/', save_dir='outputs/')

            >>> # Video
            >>> results = model.predict('video.mp4', visualize=True, save_path='output.mp4')
        """

        if self.predictor is None:
            raise RuntimeError("No model loaded! Call .load() first or train a model")

        print(f"\n🔍 Running inference...")

        # Determine source type
        if isinstance(source, str):
            source_path = Path(source)

            if source_path.is_dir():
                # Directory of images
                print(f"  Source: Directory ({source})")
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                sources = [
                    str(f) for f in source_path.glob('*')
                    if f.suffix.lower() in image_extensions
                ]
                print(f"  Found {len(sources)} images")

            elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                # Video file
                print(f"  Source: Video ({source})")
                sources = source

            else:
                # Single image
                print(f"  Source: Image ({source})")
                sources = source

        elif isinstance(source, list):
            # List of paths
            print(f"  Source: List of {len(source)} items")
            sources = source
        else:
            raise ValueError(f"Invalid source type: {type(source)}")

        # Run prediction
        results = self.predictor.predict(
            sources=sources,
            conf_threshold=conf_threshold,
            visualize=visualize,
            save_path=save_path,
            save_dir=save_dir
        )

        print(f"✓ Inference complete")

        return results

    def export(self, export_path: str, format: str = 'pytorch'):
        """
        Export model to different formats

        Args:
            export_path: Path to save exported model
            format: Export format ('pytorch', 'onnx', 'torchscript')

        Examples:
            >>> model.export('model.pth', format='pytorch')
            >>> model.export('model.onnx', format='onnx')
        """

        if self.predictor is None:
            raise RuntimeError("No model loaded! Call .load() first")

        print(f"📦 Exporting model to {format} format...")

        if format == 'pytorch':
            # Save PyTorch model
            torch.save(self.predictor.model.state_dict(), export_path)
            print(f"✓ Model exported to: {export_path}")

        elif format == 'onnx':
            # Export to ONNX (TODO: implement ONNX export)
            print("⚠️  ONNX export not yet implemented")

        else:
            raise ValueError(f"Unsupported export format: {format}")