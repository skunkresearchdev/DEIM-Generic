#!/home/hidara/miniconda3/envs/deim/bin/python
"""
Example usage of DEIM module

This demonstrates how to use DEIM for training and inference
in less than 10 lines of code.

DO NOT RUN THIS SCRIPT WHILE GPU IS BUSY WITH TRAINING!
"""

from deim import DEIM


def example_training():
    """Example: Training from scratch"""

    # Initialize model with 'under' configuration
    model = DEIM(config='under')

    # Train the model (this would run actual training on GPU)
    # Uncomment to run:
    # model.train(epochs=320, batch_size=8)

    print("Training example prepared (not executed to avoid GPU conflict)")


def example_training_pretrained():
    """Example: Training with pretrained weights"""

    # Initialize model
    model = DEIM(config='sides')

    # Train with pretrained weights (transfer learning)
    # Use the best model from a previous training run
    model.train(
        pretrained='deim_outputs/best_models/sides/best_stg1.pth',
        epochs=500,
        batch_size=32
    )

    print("Transfer learning example prepared")


def example_inference():
    """Example: Running inference on images"""

    # Initialize model
    model = DEIM(config='under')

    # Load trained weights from a completed training run
    # model.load('deim_outputs/under/20251002_215916/best_stg2.pth')

    # Run inference on single image
    # results = model.predict('image.jpg', visualize=True)

    # Run inference on multiple images
    # results = model.predict(['img1.jpg', 'img2.jpg'])

    # Run inference on video
    # results = model.predict('video.mp4', save_path='output.mp4')

    print("Inference example prepared")


def example_custom_dataset():
    """Example: Training on a custom dataset

    For detailed instructions, see: docs/CUSTOM_DATASET_GUIDE.md

    Quick steps:
    1. Prepare dataset in COCO format
    2. Create config files:
       - deim/_configs/_base/dataset_my_dataset.yml
       - deim/_configs/_base/dataloader_my_dataset.yml
       - deim/_configs/my_dataset.yml
    3. Train!
    """

    # Initialize model with your custom config name
    model = DEIM(config='my_dataset')  # Uses deim/_configs/my_dataset.yml

    # Train from scratch
    model.train(epochs=100)

    # Or fine-tune from pretrained weights
    # model.train(
    #     pretrained='deim_outputs/under/best_stg2.pth',
    #     epochs=50
    # )

    print("Custom dataset training started!")
    print("See docs/CUSTOM_DATASET_GUIDE.md for complete configuration guide")


if __name__ == '__main__':
    print("DEIM Usage Examples")
    print("=" * 50)
    print("\nNOTE: Examples are commented out to avoid GPU conflicts")
    print("Uncomment the code you want to run when GPU is available\n")

    # example_training()
    example_training_pretrained()
    # example_inference()
    # example_custom_dataset()

    print("\n✅ All examples prepared successfully!")
    print("\nTo use DEIM in your code:")
    print("1. from deim import DEIM")
    print("2. model = DEIM(config='under')")
    print("3. model.train(epochs=100)  # For training")
    print("   OR")
    print("3. model.load('checkpoint.pth')  # For inference")
    print("4. results = model.predict('image.jpg')")
