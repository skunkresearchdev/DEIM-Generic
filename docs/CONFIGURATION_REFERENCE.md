# DEIM Configuration Reference

**Complete technical reference** for all DEIM configuration parameters and training options.

> 💡 **New to DEIM?** Start with [QUICKSTART.md](QUICKSTART.md) for a simple 3-step guide to get started quickly.

This document provides in-depth explanations of all configuration files, parameters, and customization options for training DEIM on custom datasets.

---

## Step 1: Prepare Your Dataset

DEIM uses **COCO format** for object detection. Your dataset folder should look like:

```
my_dataset/
├── train/
│   └── images/
│       ├── img_001.jpg
│       ├── img_002.jpg
│       └── ...
├── val/
│   └── images/
│       ├── img_100.jpg
│       └── ...
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

### COCO Annotation Format

The `instances_train.json` and `instances_val.json` files should follow this structure:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "img_001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [x, y, width, height],
      "area": 12800,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "my_class",
      "supercategory": "object"
    }
  ]
}
```

**Note**: COCO bbox format is `[x, y, width, height]` where `(x, y)` is the **top-left corner**.

### Converting from YOLO Format

If you have YOLO format annotations, you can convert them using tools like:
- [YOLO to COCO converter](https://github.com/Taeyoung96/Yolo-to-COCO-format-converter)
- Or write a simple conversion script

---

## Step 2: Create Configuration Files

DEIM uses a **modular config system** with base configs and imports. You'll need to create **3 files**:

1. **`_base/dataset_*.yml`** - Dataset paths and class definitions
2. **`_base/dataloader_*.yml`** - Data augmentation and preprocessing (the most important for tuning!)
3. **`*.yml`** - Main config that imports and combines everything

This separation allows you to:
- ✅ Reuse augmentation strategies across different models
- ✅ Keep related settings together
- ✅ Override specific values without duplicating everything

### 2.1 Create Base Dataset Config

**File**: `deim/_configs/_base/dataset_my_dataset.yml`

```yaml
# Dataset configuration for 'my_dataset' detection

task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ["bbox"]

# Number of classes in your dataset
num_classes: 3  # e.g., 3 classes: cat, dog, bird

# Class name mappings for visualization
class_names:
  0: cat
  1: dog
  2: bird

# Set to True if using COCO-pretrained models
remap_mscoco_category: False

# Training data
train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: /path/to/my_dataset/train/images
    ann_file: /path/to/my_dataset/annotations/instances_train.json
    return_masks: False
    transforms:
      type: Compose
      ops: ~  # Will be defined in dataloader config
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

# Validation data
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

### 2.2 Create Dataloader Config

**File**: `deim/_configs/_base/dataloader_my_dataset.yml`

```yaml
# Data augmentation and preprocessing for training

train_dataloader:
  total_batch_size: 32  # Adjust based on your GPU memory
  num_workers: 4
  dataset:
    transforms:
      type: Compose
      ops:
        # Data augmentation (applied during training)
        - type: RandomPhotometricDistort
          p: 0.5
        - type: RandomZoomOut
          fill: 0
        - type: RandomIoUCrop
          p: 0.8
        - type: SanitizeBoundingBoxes
          min_size: 1
        - type: RandomHorizontalFlip

        # Optional: Advanced augmentations
        - type: GaussianBlur
          kernel_size: [3, 5]
          sigma: [0.1, 2.0]
          p: 0.3
        - type: RandomRotation
          degrees: 10
          p: 0.5
        - type: RandomPerspective
          distortion_scale: 0.2
          p: 0.3
        - type: RandomAdjustSharpness
          sharpness_factor: 2
          p: 0.3

        # Final resize and normalization
        - type: Resize
          size: [640, 640]
        - type: SanitizeBoundingBoxes
          min_size: 1
        - type: ConvertPILImage
          dtype: float32
          scale: true
        - type: ConvertBoxes
          fmt: cxcywh
          normalize: true

      # Stop heavy augmentations after certain epoch
      policy:
        name: stop_epoch
        epoch: 200  # Stop augmentations at epoch 200
        ops:
          - Mosaic
          - RandomPhotometricDistort
          - RandomZoomOut
          - RandomIoUCrop
          - GaussianBlur
          - RandomRotation
          - RandomPerspective
          - RandomAdjustSharpness

  collate_fn:
    type: BatchImageCollateFunction
    base_size: 640
    stop_epoch: 200

# Validation dataloader (minimal preprocessing)
val_dataloader:
  total_batch_size: 128  # Can be larger since no gradients
  num_workers: 4
```

### 2.3 Create Main Config

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

# Model architecture
DEIM:
  backbone: HGNetv2  # Options: HGNetv2, ResNet50, CSPDarkNet

# Backbone configuration
HGNetv2:
  name: "B0"  # Options: B0 (fastest), B1, B2, B3, B4, B5, B6 (most accurate)
  return_idx: [2, 3]
  freeze_at: -1  # -1 = no freezing, 0 = freeze all, N = freeze first N stages
  freeze_norm: False
  use_lab: True  # Use LAB color space normalization

# Encoder configuration
HybridEncoder:
  in_channels: [512, 1024]  # Must match backbone output channels
  feat_strides: [16, 32]
  hidden_dim: 128  # Lower = faster, higher = more accurate
  use_encoder_idx: [1]
  dim_feedforward: 512
  expansion: 0.34
  depth_mult: 0.5

# Decoder configuration
DFINETransformer:
  feat_channels: [128, 128]
  feat_strides: [16, 32]
  hidden_dim: 128
  dim_feedforward: 512
  num_levels: 2
  num_layers: 3  # More layers = better accuracy, slower training
  eval_idx: -1
  num_points: [6, 6]

# Optimizer settings
optimizer:
  lr: 0.0008  # Learning rate - adjust based on batch size
  # Rule of thumb: lr = 0.0001 * (batch_size / 16)
```

---

## Step 3: Configuration Parameters Explained

### Dataset Parameters

| Parameter | Description | Example | Technical Details |
|-----------|-------------|---------|-------------------|
| `num_classes` | Number of object classes in your dataset | `3` for cat, dog, bird | Defines output dimension for classification head |
| `class_names` | Mapping of class IDs to readable names | `{0: cat, 1: dog, 2: bird}` | Used for visualization in predict(); must match category_id in COCO JSON |
| `img_folder` | Path to image directory | `/path/to/images/` | Absolute or relative path; must contain all images referenced in ann_file |
| `ann_file` | Path to COCO JSON annotation file | `/path/to/instances.json` | COCO format: images, annotations, categories arrays |
| `remap_mscoco_category` | Use COCO category mapping (only for COCO-pretrained) | `False` for custom datasets | `True` if using COCO's 80 classes, `False` for custom classes |
| `return_masks` | Return instance segmentation masks | `False` | Set to `True` only for instance segmentation tasks |
| `num_workers` | Data loading threads | `4-8` | Higher values speed up data loading; limit based on CPU cores |
| `shuffle` | Shuffle data during training | `True` | Always `True` for training, `False` for validation |
| `drop_last` | Drop incomplete batches | `True` | Prevents size mismatch issues during training |

### Model Architecture Parameters

#### Backbone Configuration (HGNetv2)

| Parameter | Description | Values | Impact | GPU Memory |
|-----------|-------------|---------|--------|------------|
| `name` | Backbone size variant | `B0`, `B1`, `B2`, `B3`, `B4`, `B5`, `B6` | B0 fastest (~60 FPS), B6 most accurate | B0: ~6GB, B4: ~18GB (batch 32) |
| `return_idx` | Feature pyramid levels to return | `[2, 3]` (default) | Controls multi-scale detection; typically last 2 stages |
| `freeze_at` | Freeze backbone stages | `-1` (no freeze), `0` (all), `N` (first N) | Freezing speeds training but reduces adaptability |
| `freeze_norm` | Freeze batch normalization layers | `False` | Set `True` for fine-tuning on small datasets |
| `use_lab` | Use LAB color space normalization | `True` | LAB improves robustness to lighting variations |
| `in_channels` | Backbone output channels | `[512, 1024]` for B0 | Must match backbone architecture |

#### Encoder Configuration (HybridEncoder)

| Parameter | Description | Default | Impact |
|-----------|-------------|---------|--------|
| `hidden_dim` | Feature embedding dimension | `128` | Higher = more capacity, slower inference |
| `feat_strides` | Downsampling strides | `[16, 32]` | Matches backbone output strides |
| `use_encoder_idx` | Which encoder stages to use | `[1]` | Index of feature pyramid levels |
| `dim_feedforward` | FFN dimension | `512` | Higher = more expressiveness, slower |
| `expansion` | Channel expansion ratio | `0.34` | Controls CSPNet expansion |
| `depth_mult` | Depth multiplier | `0.5` | Scales number of layers |

#### Decoder Configuration (DFINETransformer)

| Parameter | Description | Default | Impact |
|-----------|-------------|---------|--------|
| `num_layers` | Decoder transformer layers | `3` | More layers = better accuracy, slower training |
| `num_queries` | Maximum detections per image | `300` | Higher = more detections, slower inference |
| `hidden_dim` | Decoder feature dimension | `128` | Must match encoder hidden_dim |
| `dim_feedforward` | FFN dimension | `512` | Higher = more capacity |
| `num_levels` | Feature pyramid levels | `2` | Matches encoder output levels |
| `num_points` | Deformable attention points | `[6, 6]` | More points = finer localization |
| `eval_idx` | Which decoder layer to evaluate | `-1` (last) | Use -1 for best predictions |

### Training Parameters

| Parameter | Description | Typical Values | How to Tune |
|-----------|-------------|----------------|-------------|
| `epoches` | Total training epochs | `100-300` | Small datasets: 200-300, Large: 100-150 |
| `total_batch_size` | Batch size (across all GPUs) | `16-64` | Limited by GPU memory; higher = more stable |
| `lr` | Learning rate | `0.0001-0.001` | Rule of thumb: `0.0001 * (batch_size / 16)` |
| `betas` | Adam optimizer betas | `[0.9, 0.999]` | Rarely need to change |
| `weight_decay` | L2 regularization | `0.0001` | Higher prevents overfitting on small datasets |
| `clip_max_norm` | Gradient clipping | `0.1` | Prevents exploding gradients |

### Data Augmentation Parameters (Advanced)

#### Core Augmentations

| Transform | Purpose | Parameters | When to Use | When NOT to Use |
|-----------|---------|------------|-------------|-----------------|
| `RandomPhotometricDistort` | Color/brightness jittering | `p=0.5` | Always for robustness to lighting | Grayscale/thermal images |
| `RandomZoomOut` | Creates scale variation | `fill=0` (background color) | Multi-scale objects | Fixed-size objects only |
| `RandomIoUCrop` | Crops maintaining objects | `p=0.8` (crop probability) | Dense scenes, data augmentation | **Nested objects** (breaks hierarchy) |
| `SanitizeBoundingBoxes` | Removes invalid boxes | `min_size=1` | Always (data integrity) | Never skip |
| `RandomHorizontalFlip` | Mirror image horizontally | `p=0.5` | Symmetric scenes | **Asymmetric** (text, logos, arrows) |

#### Advanced Augmentations

| Transform | Purpose | Parameters | Domain-Specific Use Cases | Avoid When |
|-----------|---------|------------|---------------------------|------------|
| `GaussianBlur` | Simulates motion/focus blur | `kernel_size=[3,5]`, `sigma=[0.1,2.0]`, `p=0.3` | Low-quality cameras, motion blur, atmospheric effects (dust, fog) | High-resolution, sharp details critical |
| `RandomRotation` | Rotation invariance | `degrees=10`, `p=0.5` | Aerial imagery, objects with arbitrary orientation | **Circular objects** (radially symmetric), gravity-dependent |
| `RandomPerspective` | Camera angle variation | `distortion_scale=0.2`, `p=0.3` | 3D scenes, varying camera angles | **Circular features** (distorts geometry), 2D top-down views |
| `RandomAdjustSharpness` | Sharpness variation | `sharpness_factor=2`, `p=0.3` | Mixed quality data, edge detection tasks | Uniformly sharp datasets |
| `Mosaic` | 4-image mosaic | `mosaic_prob=0.5` | Small datasets (4x data), multi-scale | Large datasets (adds overhead) |

#### Required Preprocessing (Never Remove)

| Transform | Purpose | Parameters | Notes |
|-----------|---------|------------|-------|
| `Resize` | Normalize image size | `size=[640, 640]` | Must match model input size |
| `ConvertPILImage` | Convert to tensor | `dtype='float32'`, `scale=True` | Scales [0,255] → [0,1] |
| `ConvertBoxes` | Normalize box format | `fmt='cxcywh'`, `normalize=True` | DETR expects center coords |

#### Augmentation Policy (Stop Heavy Augmentation)

```yaml
policy:
  name: stop_epoch
  epoch: 180  # Stop at 90% of total epochs (200 * 0.9)
  ops: ['Mosaic', 'RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']
```

**Why stop augmentation?** Heavy augmentation early helps generalization, but final epochs benefit from clean data for fine-tuning.

**💡 Pro Tip**: Document your augmentation reasoning with comments! See real production example below.

### Real-World Example: Thermal Imaging for Truck Brakes

Here's a production config with domain-specific reasoning (from `dataloader_under.yml`):

```yaml
train_dataloader:
  dataset:
    transforms:
      ops:
        # Thermal signature variations (hot brakes, hub heat)
        - {type: RandomPhotometricDistort, p: 0.5}

        # Simulates varying truck distances from camera
        - {type: RandomZoomOut, fill: 0}

        # REDUCED from 0.8 - less aggressive to preserve nested hierarchy
        # (wheels → hubs → brakes are nested objects)
        - {type: RandomIoUCrop, p: 0.3}

        - {type: SanitizeBoundingBoxes, min_size: 1}

        # Left/right wheel symmetry
        - {type: RandomHorizontalFlip}

        # Heat shimmer from hot brakes, dust
        - {type: GaussianBlur, kernel_size: [3, 5], sigma: [0.1, 2.0], p: 0.3}

        # Critical for hub bolt details
        - {type: RandomAdjustSharpness, sharpness_factor: 2, p: 0.3}

        # REMOVED RandomRotation - circular wheels/hubs don't benefit (radially symmetric)
        # REMOVED RandomPerspective - distorts circular features, hurts bolt pattern detection

        - {type: Resize, size: [640, 640]}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}
```

**Key Insights**:
- ✅ Comments explain the **domain context** (thermal, mining, trucks)
- ✅ Explains **why** augmentations were reduced/removed (nested objects, circular geometry)
- ✅ Links augmentations to **real-world phenomena** (heat shimmer, dust, varying distances)
- ✅ Future maintainers understand the reasoning, not just the config

---

## Step 3.5: Advanced Configuration Topics

### Optimizer Configuration (Advanced)

The optimizer config supports **parameter-specific learning rates** using regex patterns:

```yaml
optimizer:
  type: AdamW
  lr: 0.0008  # Default learning rate
  betas: [0.9, 0.999]
  weight_decay: 0.0001

  # Parameter-specific overrides
  params:
    # Backbone gets 50% of base learning rate
    - params: "^(?=.*backbone)(?!.*norm|bn).*$"
      lr: 0.0004  # 0.5x base lr

    # Backbone normalization layers: lower lr, no weight decay
    - params: "^(?=.*backbone)(?=.*norm|bn).*$"
      lr: 0.0004
      weight_decay: 0.  # No regularization on norm layers

    # Encoder/decoder normalization and bias: no weight decay
    - params: "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$"
      weight_decay: 0.
```

**Why different learning rates?**
- **Backbone**: Often pretrained, needs gentler updates
- **Normalization layers**: Sensitive to weight decay, can destabilize training
- **Bias terms**: Regularizing bias hurts performance, set weight_decay=0

### Learning Rate Scheduling

```yaml
lr_scheduler:
  type: MultiStepLR
  milestones: [180, 240]  # Reduce lr at these epochs
  gamma: 0.1  # Multiply lr by 0.1 at each milestone
```

**Common strategies**:
- **MultiStepLR**: Step decay at fixed epochs
- **CosineAnnealingLR**: Smooth decay following cosine curve
- **ReduceLROnPlateau**: Reduce when validation metric plateaus

### EMA (Exponential Moving Average)

```yaml
ema:
  enabled: True
  decay: 0.9999
  warmup_epochs: 5
```

**Benefits**: Smoother convergence, better generalization, more stable validation metrics

### Mixed Precision Training (AMP)

Automatically enabled for GPU training. Speeds up training by ~2x and reduces memory by ~40%.

### Validation Configuration

```yaml
val_dataloader:
  total_batch_size: 64  # Can be larger (no gradients)
  dataset:
    transforms:
      ops:
        - {type: Resize, size: [640, 640]}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
```

**Key differences from training**:
- No augmentations (only resize + convert)
- Larger batch size (no backward pass)
- No shuffling, no dropping last batch

---

## Step 4: Training Your Model

```python
from deim import DEIM

# Initialize with your config
model = DEIM(config='my_dataset')

# Option 1: Train from scratch
model.train(epochs=100)

# Option 2: Fine-tune from pretrained weights
model.train(
    pretrained='deim_outputs/under/20251002_215916/best_stg2.pth',
    epochs=50
)

# Option 3: Override config parameters
model.train(
    epochs=200,
    batch_size=32,
    learning_rate=0.0005
)
```

---

## Step 5: Inference

```python
# Load trained model
model.load('deim_outputs/my_dataset/best_stg2.pth')

# Run inference
results = model.predict('image.jpg', visualize=True)

# Display results
from PIL import Image
for r in results:
    display(Image.fromarray(r['visualization']))
```

---

## Advanced Troubleshooting

### 1. Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions** (in order of effectiveness):
```yaml
# Option 1: Reduce batch size (most effective)
total_batch_size: 8  # Reduce from 16 or 32

# Option 2: Use gradient accumulation (simulates larger batch)
gradient_accumulation_steps: 4  # Effective batch = 8 * 4 = 32

# Option 3: Smaller model
HGNetv2:
  name: "B0"  # Smallest backbone
hidden_dim: 96  # Reduce from 128

# Option 4: Lower image resolution
- type: Resize
  size: [512, 512]  # Reduce from 640x640
```

**GPU Memory Guide** (batch size 16):
- B0 + 128 dim: ~4GB
- B0 + 256 dim: ~6GB
- B2 + 256 dim: ~10GB
- B4 + 256 dim: ~16GB

### 2. Slow Training

**Symptoms**: <0.5 iterations/second, long epoch times

**Diagnosis**:
```python
# Profile data loading vs GPU computation
# If GPU utilization <80%, bottleneck is data loading
```

**Solutions**:
```yaml
# Data loading bottleneck
num_workers: 8  # Increase from 4 (up to CPU cores)
pin_memory: True  # Faster host-to-GPU transfer

# Too many augmentations
# Remove expensive transforms:
# - type: Mosaic  # Most expensive
# - type: RandomRotation  # Moderately expensive
# - type: RandomPerspective  # Moderately expensive

# GPU bottleneck (model too large)
total_batch_size: 32  # Increase if memory allows
# Or use smaller model (B0 instead of B2)
```

### 3. Poor Accuracy / Not Learning

**Diagnostic Checklist**:

| Issue | Symptom | Solution |
|-------|---------|----------|
| Bad annotations | mAP <10% after 50 epochs | Verify COCO JSON format, check bboxes |
| Too few epochs | mAP improving but slow | Train 200-300 epochs for small datasets |
| Learning rate too high | Loss diverges/NaN | Reduce lr to 0.0001-0.0005 |
| Learning rate too low | Loss decreases very slowly | Increase lr to 0.001-0.002 |
| Model too small | mAP plateaus at 30-40% | Use B2/B4 backbone, increase hidden_dim |
| Overfitting | Train mAP high, val mAP low | Add augmentations, increase weight_decay |
| Underfitting | Both train and val mAP low | Larger model, train longer, reduce regularization |
| Class imbalance | Good on common classes, bad on rare | Use class weights, oversample rare classes |

**Advanced debugging**:
```yaml
# Enable gradient clipping if loss explodes
clip_max_norm: 0.1

# Freeze backbone initially, then unfreeze
HGNetv2:
  freeze_at: 4  # Freeze all stages initially
# After 50 epochs, set freeze_at: -1 and resume training
```

### 4. Bounding Box Issues

**Symptoms**: Boxes in wrong location, wrong size, or missing

**Coordinate format check**:
- COCO format: `[x, y, width, height]` where (x,y) is **top-left**
- DEIM internal: center format `(cx, cy, w, h)` normalized to [0,1]

**Common issues**:
```python
# Verify annotations are correct
from pycocotools.coco import COCO
coco = COCO('annotations/instances_train.json')
img_id = 1
anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
# Check: bbox values reasonable? Within image bounds?
```

### 5. Classes Not Showing Names

**Solution**: Add `class_names` to **source config**, not output config
```yaml
# File: deim/_configs/_base/dataset_my_dataset.yml (CORRECT)
class_names:
  0: my_class_1
  1: my_class_2

# NOT in deim_outputs/my_dataset/config.yml (output dir, won't work)
```

### 6. Model Not Loading Pretrained Weights

**Symptoms**: Warning about missing/unexpected keys

**Solutions**:
```python
# Check if num_classes matches
# If pretrained on COCO (80 classes), you need to ignore classification head
model.train(
    pretrained='coco_model.pth',
    strict_loading=False  # Ignore head mismatch
)
```

### 7. NaN Loss

**Causes and fixes**:
```yaml
# Cause 1: Learning rate too high
optimizer:
  lr: 0.0001  # Reduce from 0.001

# Cause 2: No gradient clipping
clip_max_norm: 0.1  # Add this

# Cause 3: Mixed precision issues (rare)
use_amp: False  # Disable AMP temporarily
```

---

## Model Size vs Performance Guide

| Backbone | Hidden Dim | Speed (FPS) | mAP | Use Case |
|----------|-----------|-------------|-----|----------|
| B0 | 96 | ~60 | ~40% | Real-time applications |
| B0 | 128 | ~50 | ~45% | Balanced (default) |
| B2 | 256 | ~30 | ~50% | High accuracy needed |
| B4 | 256 | ~20 | ~55% | Maximum accuracy |

**GPU Memory Usage** (batch size 32):
- B0 + hidden_dim 128: ~6GB
- B2 + hidden_dim 256: ~12GB
- B4 + hidden_dim 256: ~18GB

---

## Example Configs by Use Case

### Small Objects Detection (e.g., thermal sensors, PCB defects)
```yaml
HGNetv2:
  name: "B2"  # Need more capacity
hidden_dim: 256
num_layers: 6  # More layers for fine details

# More aggressive multi-scale augmentation
train_dataloader:
  dataset:
    transforms:
      ops:
        - type: RandomZoomOut
          fill: 0
        - type: RandomIoUCrop
          p: 0.9  # Increase crop probability
```

### Few Classes, High Accuracy (e.g., 1-3 classes)
```yaml
num_classes: 1
HGNetv2:
  name: "B4"  # Use largest backbone
hidden_dim: 256
num_layers: 6
epoches: 300  # Train longer
```

### Fast Inference Required (e.g., real-time detection)
```yaml
HGNetv2:
  name: "B0"
hidden_dim: 96  # Minimal dimensions
num_layers: 3
num_queries: 100  # Fewer queries
```

---

## Next Steps

1. **Monitor Training**: Check `deim_outputs/my_dataset/` for logs and checkpoints
2. **Evaluate**: Best model is automatically saved as `best_stg2.pth`
3. **Tune Hyperparameters**: Adjust learning rate, augmentations based on results
4. **Export for Production**: Use `model.export()` for deployment

---

## References

- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [Data Augmentation Guide](https://pytorch.org/vision/stable/transforms.html)
- [HGNetV2 Paper](https://arxiv.org/abs/2204.00993)
