# Dataset Format Conversion Guide

## Overview

This project uses **COCO format** as the primary annotation format, even though we train YOLO models. This decision is based on:

1. **Multi-Model Testing**: We evaluate multiple detection architectures (YOLO, RT-DETR, D-FINE, etc.), many of which natively expect COCO format
2. **Standardization**: COCO is the industry standard for object detection datasets
3. **Rich Metadata**: COCO format preserves dataset splits, category information, and annotation metadata in a structured way
4. **Framework Compatibility**: Easier integration with various frameworks (Detectron2, MMDetection, Ultralytics)
5. **Evaluation Metrics**: COCO provides standardized evaluation metrics (mAP, AR) used across the research community

## Format Comparison

| Aspect | COCO Format | YOLO Format |
|--------|-------------|-------------|
| **File Structure** | Single JSON per split | One .txt per image |
| **Coordinates** | Absolute pixels | Normalized (0-1) |
| **Metadata** | Rich (categories, licenses, info) | Minimal |
| **Training Speed** | Slower parsing | Faster parsing |
| **Debugging** | Harder (large JSON) | Easier (per-image text) |
| **Multi-Framework** | ✅ Wide support | ❌ YOLO-specific |
| **Best For** | Benchmarking, research | YOLO training only |

## Bidirectional Conversion

### COCO → YOLO (Using Ultralytics)

Ultralytics provides built-in conversion from COCO to YOLO format:

```python
from ultralytics.data.converter import convert_coco

# Basic conversion
convert_coco(
    labels_dir="path/to/coco/annotations/",  # Directory with instances_train2017.json, etc.
    save_dir="path/to/output/yolo_labels/",  # Output directory for YOLO .txt files
    use_segments=False,  # Set True for segmentation tasks
    use_keypoints=False,  # Set True for pose estimation
    cls91to80=False,  # Set True to map COCO 91 classes to 80
)

# With segmentation masks
convert_coco(
    labels_dir="datasets/coco/annotations/",
    save_dir="datasets/coco_yolo/labels/",
    use_segments=True,
)
```

**Output Structure:**
```
yolo_labels/
├── train/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── val/
    ├── image1.txt
    └── ...
```

**YOLO Format (per line):**
```
class_id center_x center_y width height
0 0.716797 0.395833 0.216406 0.147222
```

### YOLO → COCO (Using External Tool)

For YOLO to COCO conversion, use the [Yolo-to-COCO-format-converter](https://github.com/Taeyoung96/Yolo-to-COCO-format-converter):

```bash
# Clone the repository
git clone https://github.com/Taeyoung96/Yolo-to-COCO-format-converter.git
cd Yolo-to-COCO-format-converter

# Install dependencies
pip install -r requirements.txt
```

**Usage:**

```python
import os
from Yolo_to_COCO import yolo_to_coco

# Define paths
yolo_labels_dir = "path/to/yolo/labels/"
images_dir = "path/to/images/"
output_json = "path/to/output/coco_annotations.json"

# Define class names (must match your YOLO classes)
class_names = ["class1", "class2", "class3"]

# Convert
yolo_to_coco(
    yolo_labels_dir=yolo_labels_dir,
    images_dir=images_dir,
    output_json=output_json,
    class_names=class_names,
)
```

**Alternative: Manual Conversion Script**

```python
import json
import os
from pathlib import Path
from PIL import Image

def yolo_to_coco(yolo_dir, images_dir, class_names, output_json):
    """
    Convert YOLO format annotations to COCO format.

    Args:
        yolo_dir: Directory containing YOLO .txt files
        images_dir: Directory containing corresponding images
        class_names: List of class names in order
        output_json: Output path for COCO JSON file
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create categories
    for idx, name in enumerate(class_names):
        coco_format["categories"].append({
            "id": idx,
            "name": name,
            "supercategory": "none"
        })

    annotation_id = 0

    # Process each image
    for img_id, txt_file in enumerate(sorted(Path(yolo_dir).glob("*.txt"))):
        # Get corresponding image
        img_name = txt_file.stem
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = Path(images_dir) / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if not img_path:
            continue

        # Get image dimensions
        img = Image.open(img_path)
        width, height = img.size

        # Add image info
        coco_format["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })

        # Read YOLO annotations
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                center_x = float(parts[1]) * width
                center_y = float(parts[2]) * height
                bbox_width = float(parts[3]) * width
                bbox_height = float(parts[4]) * height

                # Convert to COCO format (top-left x, y, width, height)
                x = center_x - bbox_width / 2
                y = center_y - bbox_height / 2

                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": [x, y, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                annotation_id += 1

    # Save COCO JSON
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"Conversion complete: {len(coco_format['images'])} images, "
          f"{len(coco_format['annotations'])} annotations")

# Example usage
yolo_to_coco(
    yolo_dir="datasets/yolo/labels/train/",
    images_dir="datasets/yolo/images/train/",
    class_names=["damage", "corrosion", "crack"],
    output_json="datasets/coco/annotations/instances_train.json"
)
```

## Workflow in This Project

1. **Annotation**: Create annotations in COCO format using tools like CVAT, LabelImg, or Roboflow
2. **Storage**: Keep master dataset in COCO format (`deim/_configs/_base/dataset_*.yml`)
3. **Training**: When training YOLO models, Ultralytics automatically handles COCO format
4. **Conversion** (if needed): Use `convert_coco()` if you need explicit YOLO .txt files for custom pipelines

## References

- [Ultralytics Converter Documentation](https://docs.ultralytics.com/reference/data/converter/)
- [Yolo-to-COCO-format-converter](https://github.com/Taeyoung96/Yolo-to-COCO-format-converter)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [YOLO Format Specification](https://docs.ultralytics.com/datasets/detect/)
