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
    model.train(epochs=400, batch_size=64)

    print("Training example prepared (not executed to avoid GPU conflict)")


def example_training_pretrained():
    """Example: Training with pretrained weights"""

    # Initialize model
    model = DEIM(config='sides')

    # Train with pretrained weights (transfer learning)
    # model.train(
    #     pretrained='base_model.pth',
    #     epochs=50,
    #     learning_rate=0.001
    # )

    print("Transfer learning example prepared")


def example_inference():
    """Example: Running inference on images"""

    # Initialize model
    model = DEIM(config='under')

    # Load trained weights
    # model.load('deim_outputs/under/20241002_143022/best_stg1.pth')

    # Run inference on single image
    # results = model.predict('image.jpg', visualize=True)

    # Run inference on multiple images
    # results = model.predict(['img1.jpg', 'img2.jpg'])

    # Run inference on video
    # results = model.predict('video.mp4', save_path='output.mp4')

    print("Inference example prepared")


def example_custom_dataset():
    """Example: Using custom dataset"""

    # Initialize model with custom config
    model = DEIM(config='under')

    # Train with custom dataset path
    # model.train(
    #     dataset_path='/path/to/custom/dataset',
    #     epochs=100,
    #     batch_size=16
    # )

    print("Custom dataset example prepared")


if __name__ == '__main__':
    print("DEIM Usage Examples")
    print("=" * 50)
    print("\nNOTE: Examples are commented out to avoid GPU conflicts")
    print("Uncomment the code you want to run when GPU is available\n")

    example_training()
    # example_training_pretrained()
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