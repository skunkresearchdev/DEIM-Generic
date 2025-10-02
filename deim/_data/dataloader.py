"""
DataLoader wrapper for DEIM
Handles batch creation and collation for object detection
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from typing import Dict, Any, Optional, List


def collate_fn(batch):
    """
    Custom collate function for object detection

    Handles variable number of objects per image
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    # Stack images into batch
    images = torch.stack(images, 0)

    return images, targets


class DataLoader:
    """
    DataLoader wrapper for DEIM training and inference

    Provides sensible defaults for object detection
    """

    def __init__(self,
                 dataset,
                 batch_size: int = 32,
                 shuffle: bool = False,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 **kwargs):
        """
        Initialize DataLoader

        Args:
            dataset: Dataset object
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for GPU transfer
            drop_last: Whether to drop last incomplete batch
            **kwargs: Additional arguments for PyTorch DataLoader
        """

        self.dataset = dataset
        self.batch_size = batch_size

        # Create PyTorch DataLoader
        self.loader = TorchDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **kwargs
        )

    def __iter__(self):
        """Iterate through batches"""
        return iter(self.loader)

    def __len__(self):
        """Get number of batches"""
        return len(self.loader)