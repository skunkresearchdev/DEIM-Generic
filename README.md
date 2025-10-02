# DEIM - DETR with Improved Matching

A simple and powerful object detection module, similar to Ultralytics but for DEIM models.

## Features

- **Simple API**: Train and run inference in less than 10 lines of code
- **Auto-detection**: Automatically detects COCO vs YOLO dataset formats
- **Pre-configured**: Ready-to-use configurations for 'under' and 'sides' detection
- **Transfer Learning**: Easy to use pre-trained weights
- **Visualization**: Built-in support for supervision package annotations

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install DEIM module
pip install -e .
```

## Quick Start

### Training

```python
from deim import DEIM

# Train from scratch
model = DEIM(config='under')
model.train(epochs=100, batch_size=32)

# Train with pretrained weights
model = DEIM(config='sides')
model.train(
    pretrained='base_model.pth',
    epochs=50,
    learning_rate=0.001
)

# Train with custom dataset
model = DEIM(config='under')
model.train(
    dataset_path='/path/to/your/dataset',
    epochs=100
)
```

### Inference

```python
from deim import DEIM

# Load model and run inference
model = DEIM(config='under')
model.load('deim_outputs/under/20241002_143022/best_stg1.pth')

# Single image
results = model.predict('image.jpg', visualize=True)

# Multiple images
results = model.predict(['img1.jpg', 'img2.jpg'])

# Video
results = model.predict('video.mp4', save_path='output.mp4')

# Directory of images
results = model.predict('path/to/images/', save_dir='outputs/')
```

## Configuration

### Pre-configured Models

- **`under`**: Configuration for under-vehicle detection
- **`sides`**: Configuration for side-view detection

### Custom Configuration

```python
# Use custom YAML config
model = DEIM(config='path/to/custom.yml')
model.train(epochs=100)
```

### Parameter Overrides

```python
model = DEIM(config='under')
model.train(
    epochs=200,           # Override epochs
    batch_size=16,        # Override batch size
    learning_rate=0.0001, # Override learning rate
    dataset_path='/custom/path'  # Custom dataset
)
```

## Dataset Structure

### COCO Format
```
dataset/
├── annotations/
│   ├── train.json
│   └── val.json
└── images/
    ├── train/
    └── val/
```

### YOLO Format
```
dataset/
├── train/
│   ├── image1.jpg
│   ├── image1.txt
│   ├── image2.jpg
│   └── image2.txt
└── val/
    ├── image1.jpg
    ├── image1.txt
    └── ...
```

## Adding Annotated Files for Training

The module uses the `supervision` package for visualization. To add new annotated training data:

1. **Prepare your images** in either COCO or YOLO format
2. **Place them** in the appropriate dataset directory
3. **Train** with the dataset path:

```python
model = DEIM(config='under')
model.train(dataset_path='/path/to/your/annotated/data')
```

## Output Structure

Training outputs are saved with timestamps:
```
deim_outputs/
├── under/
│   └── 20241002_143022/
│       ├── best_stg1.pth    # Stage 1 model
│       ├── best_stg2.pth    # Stage 2 model (if applicable)
│       └── config.yml       # Training configuration
└── sides/
    └── ...
```

## Environment

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **GPU**: CUDA-enabled GPU required
- **Interpreter**: `/home/hidara/miniconda3/envs/deim/bin/python`

## Development

```bash
# Linting
ruff check .

# Type checking
pyright .
```

## Architecture

- **HGNetv2**: Backbone network
- **DEIM**: DETR with Improved Matching
- **DFINE**: Decoder transformer

## License

See LICENSE file for details.

## Acknowledgments

Based on DEIM (DETR with Improved Matching) architecture for fast convergence in object detection tasks.