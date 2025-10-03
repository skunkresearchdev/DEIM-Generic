# DEIM Quick Start - Custom Dataset Training

Train DEIM on your own dataset in 3 steps. See [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) for detailed explanations.

---

## Step 1: Prepare Your Dataset (COCO Format)

```
my_dataset/
├── train/images/          # Training images
├── val/images/            # Validation images
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

**COCO JSON format**: Use existing tools to convert YOLO/other formats to COCO.

---

## Step 2: Create 3 Config Files

Copy and modify the `under` configs as templates:

### 2.1 Dataset Config
**File**: `deim/_configs/_base/dataset_my_dataset.yml`

```yaml
task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ["bbox"]

# CHANGE THESE:
num_classes: 3  # Your number of classes
class_names:    # Your class names for visualization
  0: cat
  1: dog
  2: bird

remap_mscoco_category: False  # False for custom datasets

# CHANGE THESE PATHS:
train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: /path/to/my_dataset/train/images
    ann_file: /path/to/my_dataset/annotations/instances_train.json
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: /path/to/my_dataset/val/images
    ann_file: /path/to/my_dataset/annotations/instances_val.json
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 4
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
```

### 2.2 Dataloader Config (Data Augmentation)
**File**: `deim/_configs/_base/dataloader_my_dataset.yml`

**Start with this simple config**, then tune based on results:

```yaml
# Data augmentation and preprocessing

train_dataloader:
  total_batch_size: 16  # Adjust based on GPU memory (8/16/32)
  num_workers: 4
  dataset:
    transforms:
      type: Compose
      ops:
        # Basic augmentations
        - {type: RandomPhotometricDistort, p: 0.5}
        - {type: RandomZoomOut, fill: 0}
        - {type: RandomIoUCrop, p: 0.8}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: RandomHorizontalFlip}

        # Optional: Add domain-specific augmentations
        # - {type: GaussianBlur, kernel_size: [3, 5], sigma: [0.1, 2.0], p: 0.3}
        # - {type: RandomRotation, degrees: 10, p: 0.5}
        # - {type: RandomPerspective, distortion_scale: 0.2, p: 0.3}
        # - {type: RandomAdjustSharpness, sharpness_factor: 2, p: 0.3}

        # Required preprocessing
        - {type: Resize, size: [640, 640]}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}

      policy:
        name: stop_epoch
        epoch: 180  # Stop heavy augmentations at 90% of total epochs
        ops: ['Mosaic', 'RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']

  collate_fn:
    type: BatchImageCollateFunction
    base_size: 640
    stop_epoch: 180

  shuffle: True

val_dataloader:
  total_batch_size: 64  # Can be larger (no gradients)
  num_workers: 4
  dataset:
    transforms:
      ops:
        - {type: Resize, size: [640, 640]}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
  shuffle: False
```

**💡 Augmentation Tips**:
- **Start simple** (basic augmentations above), add more if needed
- **Remove augmentations** that hurt your domain (e.g., rotation for circular objects)
- **Add comments** explaining your reasoning (like the thermal example in docs)

### 2.3 Main Config
**File**: `deim/_configs/my_dataset.yml`

```yaml
# DEIM Configuration for 'my_dataset' detection

__include__:
  [
    "./_base/dataset_my_dataset.yml",
    "./_base/runtime.yml",
    "./_base/dfine_hgnetv2.yml",
    "./_base/dataloader_my_dataset.yml",
    "./_base/optimizer.yml",
  ]

output_dir: ./deim_outputs/my_dataset

# Model architecture (usually don't need to change)
DEIM:
  backbone: HGNetv2

HGNetv2:
  name: "B0"  # B0 (fast) or B2/B4 (accurate)
  return_idx: [2, 3]
  freeze_at: -1
  freeze_norm: False
  use_lab: True

HybridEncoder:
  in_channels: [512, 1024]
  feat_strides: [16, 32]
  hidden_dim: 128
  use_encoder_idx: [1]
  dim_feedforward: 512
  expansion: 0.34
  depth_mult: 0.5

DFINETransformer:
  feat_channels: [128, 128]
  feat_strides: [16, 32]
  hidden_dim: 128
  dim_feedforward: 512
  num_levels: 2
  num_layers: 3
  eval_idx: -1
  num_points: [6, 6]

# Training parameters
epoches: 200  # Adjust based on dataset size

optimizer:
  type: AdamW
  lr: 0.0008  # May need tuning
  betas: [0.9, 0.999]
  weight_decay: 0.0001
  params:
    - params: "^(?=.*backbone)(?!.*norm|bn).*$"
      lr: 0.0004
    - params: "^(?=.*backbone)(?=.*norm|bn).*$"
      lr: 0.0004
      weight_decay: 0.
    - params: "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$"
      weight_decay: 0.

train_dataloader:
  total_batch_size: 16  # Override if needed

val_dataloader:
  total_batch_size: 64
```

---

## Step 3: Train!

```python
from deim import DEIM

# Initialize with your config
model = DEIM(config='my_dataset')

# Train from scratch
model.train(epochs=200)

# Or fine-tune from pretrained weights
# model.train(
#     pretrained='deim_outputs/under/best_stg2.pth',
#     epochs=100
# )
```

Training outputs saved to: `deim_outputs/my_dataset/`

---

## Step 4: Run Inference

```python
# Load trained model
model = DEIM(config='my_dataset')
model.load('deim_outputs/my_dataset/best_stg2.pth')

# Run inference (always returns list)
results = model.predict(['image1.jpg', 'image2.jpg'], visualize=True)

# Display results
from PIL import Image
for r in results:
    display(Image.fromarray(r['visualization']))
```

---

## Common Changes

### Adjust Batch Size (Out of Memory?)
```yaml
# In dataloader_my_dataset.yml
train_dataloader:
  total_batch_size: 8  # Reduce from 16
```

### Speed vs Accuracy Trade-off
```yaml
# In my_dataset.yml
HGNetv2:
  name: "B0"  # Fastest (default)
  # name: "B2"  # Balanced
  # name: "B4"  # Most accurate
```

### Train Longer
```yaml
# In my_dataset.yml
epoches: 300  # Increase from 200
```

### Disable Specific Augmentations
```yaml
# In dataloader_my_dataset.yml, comment out unwanted augmentations:
# - {type: RandomRotation, degrees: 10, p: 0.5}  # Commented out
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `total_batch_size` to 8 or 4 |
| Training too slow | Reduce `num_workers` or disable heavy augmentations |
| Poor accuracy | Train longer (300 epochs), use larger backbone (B2/B4) |
| Class names not showing | Add `class_names` to `dataset_my_dataset.yml` |

---

## Next Steps

- ✅ Monitor training in `deim_outputs/my_dataset/`
- ✅ Best model saved as `best_stg2.pth`
- ✅ See [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) for detailed parameter tuning
- ✅ Check `dataloader_under.yml` for real-world augmentation examples with reasoning

---

**Need more details?** See [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) for:
- Detailed parameter explanations
- When to use/avoid specific augmentations
- Use-case specific configurations
- Performance tuning guide
