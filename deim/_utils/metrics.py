"""
Metrics calculation for DEIM
Evaluation utilities for object detection
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any


def calculate_metrics(predictions: List[Dict],
                     targets: List[Dict],
                     iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate object detection metrics

    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with metrics (mAP, precision, recall, etc.)
    """

    # Simple placeholder implementation
    # In production, would use COCO evaluation metrics

    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives

    for pred, target in zip(predictions, targets):
        pred_boxes = pred.get('boxes', torch.empty(0, 4))
        pred_scores = pred.get('scores', torch.empty(0))
        pred_labels = pred.get('labels', torch.empty(0))

        target_boxes = target.get('boxes', torch.empty(0, 4))
        target_labels = target.get('labels', torch.empty(0))

        # Calculate IoU between predictions and targets
        if len(pred_boxes) > 0 and len(target_boxes) > 0:
            ious = calculate_iou(pred_boxes, target_boxes)

            # Match predictions to targets
            matched_targets = set()

            for i in range(len(pred_boxes)):
                max_iou = ious[i].max() if len(ious[i]) > 0 else 0
                max_idx = ious[i].argmax() if len(ious[i]) > 0 else -1

                if max_iou > iou_threshold and max_idx not in matched_targets:
                    # Check if labels match
                    if pred_labels[i] == target_labels[max_idx]:
                        total_tp += 1
                        matched_targets.add(max_idx.item() if torch.is_tensor(max_idx) else max_idx)
                    else:
                        total_fp += 1
                else:
                    total_fp += 1

            # Count false negatives (unmatched targets)
            total_fn += len(target_boxes) - len(matched_targets)

        elif len(pred_boxes) > 0:
            # All predictions are false positives
            total_fp += len(pred_boxes)
        elif len(target_boxes) > 0:
            # All targets are false negatives
            total_fn += len(target_boxes)

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }


def calculate_iou(boxes1: torch.Tensor,
                  boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes

    Args:
        boxes1: Tensor of shape (N, 4)
        boxes2: Tensor of shape (M, 4)

    Returns:
        IoU matrix of shape (N, M)
    """

    # Ensure tensors
    if not torch.is_tensor(boxes1):
        boxes1 = torch.tensor(boxes1)
    if not torch.is_tensor(boxes2):
        boxes2 = torch.tensor(boxes2)

    # Calculate intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Calculate union
    union = area1[:, None] + area2 - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-6)

    return iou