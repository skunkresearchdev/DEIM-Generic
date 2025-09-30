"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse
from pathlib import Path

from engine.core import YAMLConfig, yaml_utils
from engine.misc import dist_utils
from engine.solver import TASKS

debug = False

if debug:
    import torch

    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def main(
    args,
) -> None:
    """main"""
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), (
        "Only support from_scrach or resume or tuning at one time"
    )

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({
        k: v
        for k, v in args.__dict__.items()
        if k
        not in [
            "update",
        ]
        and v is not None
    })

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    print("cfg: ", cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


def path_exists(path: str):
    """
    Checks if the given path exists.

    Args:
        path (str): The path to check.

    Returns:
        str: The original path if it exists, otherwise raises a FileNotFoundError.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument(
        "-c",
        "--config",
        type=path_exists,
        required=True,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-r", "--resume", type=path_exists, help="Path to resume from checkpoint"
    )
    parser.add_argument(
        "-t", "--tuning", type=path_exists, help="Path to tuning checkpoint"
    )
    parser.add_argument("-d", "--device", type=str, help="Device to use")
    parser.add_argument("--seed", type=int, help="Seed for experiment reproducibility")
    parser.add_argument(
        "--use-amp", action="store_true", help="Enable auto mixed precision training"
    )
    parser.add_argument("--output-dir", type=str, help="Path to the output directory")
    parser.add_argument(
        "--summary-dir", type=str, help="Path to the TensorBoard summary directory"
    )
    parser.add_argument(
        "--test-only", action="store_true", default=False, help="Only run testing"
    )

    # priority 1
    parser.add_argument("-u", "--update", nargs="+", help="Update YAML config")

    # env
    parser.add_argument(
        "--print-method", type=str, default="builtin", help="Print method"
    )
    parser.add_argument(
        "--print-rank", type=int, default=0, help="Rank ID for printing"
    )

    parser.add_argument("--local-rank", type=int, help="Local rank ID")
    args = parser.parse_args()
    
    main(args)
