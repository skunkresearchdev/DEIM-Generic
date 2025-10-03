"""
Training orchestration for DEIM
Wraps the existing training logic from _old/train.py
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import yaml


class Trainer:
    """
    Training orchestrator for DEIM models

    Handles training workflow similar to new_run.py but through Python API
    """

    def __init__(self, config: Dict[str, Any], device: torch.device, pretrained: Optional[str] = None):
        """
        Initialize trainer

        Args:
            config: Configuration dictionary
            device: PyTorch device
            pretrained: Path to pretrained weights or None
        """
        self.config = config
        self.device = device
        self.pretrained = pretrained

        # Get paths
        self.deim_root = Path(__file__).parent.parent
        self.train_script = self.deim_root / "_engine" / "train.py"

        # Ensure train script exists
        if not self.train_script.exists():
            raise FileNotFoundError(f"Training script not found: {self.train_script}")

    def train(self, output_dir: str) -> Dict[str, Any]:
        """
        Run training process

        Args:
            output_dir: Directory to save outputs

        Returns:
            Dictionary with training results
        """

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save config for this training run
        config_save_path = output_path / "config.yml"
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # Build training command (similar to new_run.py)
        # Use absolute paths for config and output
        cmd = [
            sys.executable,  # Use current Python interpreter
            "-m", "torch.distributed.run",
            "--master_port=7777",
            "--nproc_per_node=1",
            "train.py",  # Use relative path since we're running from _engine
            "-c", str(config_save_path.absolute()),  # Use absolute path
            "--use-amp",
            "--seed=0",
        ]

        # Add pretrained weights if provided
        if self.pretrained:
            # Convert to absolute path if relative
            pretrained_path = Path(self.pretrained)
            if not pretrained_path.is_absolute():
                pretrained_path = pretrained_path.absolute()
            cmd.extend(["-t", str(pretrained_path)])

        # Set output directory in command (absolute path)
        cmd.extend(["--output-dir", str(output_path.absolute())])

        # Set environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.device.index) if self.device.index is not None else "0"

        print("\n🚀 Launching training process...")
        print(f"Command: {' '.join(cmd)}")

        try:
            # Run training from _engine directory
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(self.deim_root / "_engine")  # Run from _engine directory
            )

            # Stream output in real-time
            for line in process.stdout:
                print(line.rstrip())

            # Wait for completion
            return_code = process.wait()

            if return_code != 0:
                raise RuntimeError(f"Training failed with return code {return_code}")

            print("\n✓ Training completed successfully")

            # Collect results
            results = {
                'output_dir': str(output_path),
                'config_path': str(config_save_path),
                'best_model': str(output_path / "best_stg1.pth"),
            }

            # Check for stage 2 model
            stage2_path = output_path / "best_stg2.pth"
            if stage2_path.exists():
                results['best_model_stage2'] = str(stage2_path)

            return results

        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            raise


# Alternative implementation using direct module import (if subprocess doesn't work)
class DirectTrainer:
    """
    Direct training using module imports instead of subprocess
    This is a backup implementation
    """

    def __init__(self, config: Dict[str, Any], device: torch.device, pretrained: Optional[str] = None):
        """Initialize direct trainer"""
        self.config = config
        self.device = device
        self.pretrained = pretrained

    def train(self, output_dir: str) -> Dict[str, Any]:
        """
        Run training directly by importing modules

        Note: This is more complex as it requires setting up the training
        environment properly. The subprocess approach above is preferred.
        """

        # Add engine to path
        engine_path = Path(__file__).parent.parent / "_engine"
        if str(engine_path) not in sys.path:
            sys.path.insert(0, str(engine_path))

        # Import training modules
        from core import YAMLConfig
        from solver import TASKS

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Update config
        self.config['output_dir'] = str(output_path)

        # Save config
        config_path = output_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # Create args-like object
        class Args:
            def __init__(self, config_path, pretrained=None):
                self.config = config_path
                self.tuning = pretrained
                self.resume = None
                self.test_only = False
                self.use_amp = True
                self.seed = 0
                self.print_rank = 0
                self.print_method = 'builtin'
                self.output_dir = str(output_path)

        args = Args(str(config_path), self.pretrained)

        # Load config
        cfg = YAMLConfig(args.config)

        # Get task (detector)
        task = TASKS.get(cfg.get('task', 'detection'))

        # Create solver
        solver = task(cfg, args)

        # Run training
        solver.train()

        # Return results
        results = {
            'output_dir': str(output_path),
            'config_path': str(config_path),
            'best_model': str(output_path / "best_stg1.pth"),
        }

        return results