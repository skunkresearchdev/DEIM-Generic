"""
DEIM - DETR with Improved Matching

A simple and powerful object detection module
Similar to ultralytics but for DEIM models

Example:
    >>> from deim import DEIM
    >>>
    >>> # Train from scratch
    >>> model = DEIM(config='under')
    >>> model.train(epochs=100, batch_size=32)
    >>>
    >>> # Inference
    >>> model = DEIM(config='under')
    >>> model.load('path/to/checkpoint.pth')
    >>> results = model.predict('image.jpg')
"""

__version__ = '1.0.0'

from .api import DEIM

__all__ = ['DEIM']