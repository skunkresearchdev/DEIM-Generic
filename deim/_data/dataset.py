"""
Dataset handling for DEIM
Supports COCO and YOLO annotation formats with auto-detection
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
import numpy as np


class Dataset(TorchDataset):
    """
    DEIM Dataset handler

    Auto-detects COCO vs YOLO format and loads accordingly
    """

    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 transform: Optional[Any] = None):
        """
        Initialize dataset

        Args:
            data_path: Path to dataset root
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transforms to apply
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform

        # Auto-detect format
        self.format = self._detect_format()

        # Load annotations
        if self.format == 'coco':
            self._load_coco()
        elif self.format == 'yolo':
            self._load_yolo()
        else:
            raise ValueError(f"Unknown dataset format at {data_path}")

    def _detect_format(self) -> str:
        """Auto-detect dataset format"""

        # Check for COCO format
        coco_ann = self.data_path / 'annotations' / f'{self.split}.json'
        if coco_ann.exists():
            return 'coco'

        # Check for YOLO format
        yolo_dir = self.data_path / self.split
        if yolo_dir.exists():
            # Look for .txt label files
            label_files = list(yolo_dir.glob('*.txt'))
            if label_files:
                return 'yolo'

        return 'unknown'

    def _load_coco(self):
        """Load COCO format annotations"""
        ann_file = self.data_path / 'annotations' / f'{self.split}.json'

        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        self.images = coco_data['images']
        self.annotations = coco_data['annotations']
        self.categories = {cat['id']: cat['name']
                          for cat in coco_data['categories']}

        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def _load_yolo(self):
        """Load YOLO format annotations"""
        split_dir = self.data_path / self.split

        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(split_dir.glob(ext))

        self.images = []
        self.annotations = []

        for img_path in image_files:
            # Get corresponding label file
            label_path = img_path.with_suffix('.txt')

            if label_path.exists():
                # Read YOLO format labels
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                boxes = []
                labels = []

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO format: x_center, y_center, width, height (normalized)
                        bbox = [float(x) for x in parts[1:5]]

                        labels.append(class_id)
                        boxes.append(bbox)

                self.images.append({
                    'file_name': img_path.name,
                    'path': str(img_path)
                })

                self.annotations.append({
                    'boxes': boxes,
                    'labels': labels
                })

    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get item by index

        Returns:
            Tuple of (image, target) where target contains boxes and labels
        """

        if self.format == 'coco':
            return self._get_coco_item(idx)
        else:
            return self._get_yolo_item(idx)

    def _get_coco_item(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get COCO format item"""
        img_info = self.images[idx]
        img_path = self.data_path / 'images' / img_info['file_name']

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Get annotations
        img_id = img_info['id']
        anns = self.img_to_anns.get(img_id, [])

        boxes = []
        labels = []

        for ann in anns:
            # Convert from COCO bbox format (x, y, w, h) to (x1, y1, x2, y2)
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }

        # Apply transforms
        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def _get_yolo_item(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get YOLO format item"""
        img_info = self.images[idx]

        # Load image
        image = Image.open(img_info['path']).convert('RGB')
        w, h = image.size

        # Get annotations
        ann = self.annotations[idx]

        boxes = []
        for box in ann['boxes']:
            # Convert from YOLO format (x_center, y_center, width, height) normalized
            # to absolute (x1, y1, x2, y2)
            x_center, y_center, width, height = box

            x1 = (x_center - width/2) * w
            y1 = (y_center - height/2) * h
            x2 = (x_center + width/2) * w
            y2 = (y_center + height/2) * h

            boxes.append([x1, y1, x2, y2])

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(ann['labels'], dtype=torch.int64)
        }

        # Apply transforms
        if self.transform:
            image, target = self.transform(image, target)

        return image, target