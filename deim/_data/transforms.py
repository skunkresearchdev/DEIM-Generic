"""
Transforms for DEIM training and inference
Data augmentation and preprocessing
"""

import torch
import torchvision.transforms as T
from typing import Tuple, Dict, Any, Optional
import random
import numpy as np
from PIL import Image


class Transforms:
    """
    Transform pipeline for DEIM

    Handles image preprocessing and augmentation
    """

    def __init__(self,
                 img_size: int = 640,
                 augment: bool = False,
                 normalize: bool = True):
        """
        Initialize transforms

        Args:
            img_size: Target image size
            augment: Whether to apply augmentation (for training)
            normalize: Whether to normalize images
        """
        self.img_size = img_size
        self.augment = augment
        self.normalize = normalize

        # Basic transforms
        self.resize = T.Resize((img_size, img_size))
        self.to_tensor = T.ToTensor()

        # Normalization (ImageNet stats)
        if normalize:
            self.norm = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

    def __call__(self,
                 image: Image.Image,
                 target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict]:
        """
        Apply transforms to image and target

        Args:
            image: PIL Image
            target: Dictionary with 'boxes' and 'labels'

        Returns:
            Transformed image and target
        """

        # Get original size
        w, h = image.size

        # Apply augmentation if enabled
        if self.augment:
            image, target = self._augment(image, target)

        # Resize image
        image = self.resize(image)

        # Scale boxes to match resized image
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            # Scale boxes
            scale_x = self.img_size / w
            scale_y = self.img_size / h

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            target['boxes'] = boxes

        # Convert to tensor
        image = self.to_tensor(image)

        # Normalize if enabled
        if self.normalize:
            image = self.norm(image)

        return image, target

    def _augment(self,
                 image: Image.Image,
                 target: Dict) -> Tuple[Image.Image, Dict]:
        """Apply data augmentation"""

        # Random horizontal flip
        if random.random() < 0.5:
            image, target = self._horizontal_flip(image, target)

        # Random color jitter
        if random.random() < 0.5:
            image = self._color_jitter(image)

        return image, target

    def _horizontal_flip(self,
                        image: Image.Image,
                        target: Dict) -> Tuple[Image.Image, Dict]:
        """Apply horizontal flip"""

        w, h = image.size
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone() if torch.is_tensor(target['boxes']) else target['boxes'].copy()

            # Flip x coordinates
            if torch.is_tensor(boxes):
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            else:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

            target['boxes'] = boxes

        return image, target

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        """Apply color jittering"""

        jitter = T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )

        return jitter(image)


def get_transform(img_size: int = 640,
                  train: bool = False) -> Transforms:
    """
    Get appropriate transform for training or inference

    Args:
        img_size: Target image size
        train: Whether this is for training (enables augmentation)

    Returns:
        Transform object
    """

    return Transforms(
        img_size=img_size,
        augment=train,
        normalize=True
    )