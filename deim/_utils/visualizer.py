"""
Visualization utilities for DEIM
Uses supervision package for annotating detection results
"""

import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path


class Visualizer:
    """
    Visualization handler for DEIM detections

    Uses supervision package for high-quality annotations
    """

    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        """
        Initialize visualizer

        Args:
            class_names: Dictionary mapping class IDs to names
        """
        self.class_names = class_names or {}

        # Try to import supervision
        try:
            import supervision as sv
            self.sv = sv
            self.available = True

            # Initialize annotators
            self.box_annotator = sv.BoxAnnotator()
            self.label_annotator = sv.LabelAnnotator()

            # Optional: mask annotator for segmentation
            try:
                self.mask_annotator = sv.MaskAnnotator()
            except:
                self.mask_annotator = None

        except ImportError:
            self.available = False
            print("⚠️ Supervision not installed. Install with: pip install supervision")

    def visualize(self,
                  image: np.ndarray,
                  detections: Dict[str, Any],
                  conf_threshold: float = 0.4) -> np.ndarray:
        """
        Visualize detections on image

        Args:
            image: Input image as numpy array
            detections: Detection results with 'boxes', 'scores', 'labels'
            conf_threshold: Confidence threshold for display

        Returns:
            Annotated image
        """

        if not self.available:
            print("Supervision not available. Returning original image.")
            return image

        try:
            # Filter by confidence
            if 'scores' in detections:
                mask = detections['scores'] > conf_threshold
                boxes = detections['boxes'][mask]
                scores = detections['scores'][mask]
                labels = detections['labels'][mask]
            else:
                boxes = detections['boxes']
                scores = np.ones(len(boxes))
                labels = detections.get('labels', np.zeros(len(boxes)))

            # Create supervision Detections object
            sv_detections = self.sv.Detections(
                xyxy=boxes,
                confidence=scores,
                class_id=labels.astype(int)
            )

            # Create labels
            labels_list = []
            for class_id, score in zip(labels, scores):
                class_name = self.class_names.get(int(class_id), f"Class {class_id}")
                label = f"{class_name} {score:.2f}"
                labels_list.append(label)

            # Annotate image
            annotated = image.copy()
            annotated = self.box_annotator.annotate(
                scene=annotated,
                detections=sv_detections
            )
            annotated = self.label_annotator.annotate(
                scene=annotated,
                detections=sv_detections,
                labels=labels_list
            )

            return annotated

        except Exception as e:
            print(f"Visualization error: {str(e)}")
            return image

    def save_visualization(self,
                          image: np.ndarray,
                          save_path: str):
        """Save visualization to file"""
        from PIL import Image

        img = Image.fromarray(image)
        img.save(save_path)
        print(f"  Saved visualization: {save_path}")