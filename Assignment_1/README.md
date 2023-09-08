# Semantic Segmentation - TEK5040/9040 Autumn-2023

## Introduction

This project focuses on semantic segmentation of optical images, a crucial task in various autonomous applications, including autonomous cars. Semantic segmentation involves classifying each pixel in an input image into distinct classes. For example, in road segmentation, pixels are classified as either "ROAD" or "NO-ROAD."

If you're unfamiliar with semantic segmentation, you can learn more about it [here](https://www.jeremyjordan.me/semantic-segmentation/).

By completing this assignment, you will gain insights into performing semantic segmentation using deep neural networks and become familiar with essential deep learning operations using TensorFlow.

## Dataset

We use the Kitti road dataset, consisting of 289 labeled images, with labels for each pixel. Training images and labels are located in the following directories:
- Training Images: `data/road/training/image2/`
- Ground Truth Labels: `data/road/training/gt_image2/`

## Preparation

Before getting started, ensure that you have:
- Extracted the assignment package (`tek5040_mandatory_assignment1.zip`) into a directory of your choice.
- Installed TensorFlow 2.x on your computer. You can find installation instructions [here](https://www.tensorflow.org/install).

## Task and Questions

### 3.1 Implement train step and val step functions

- Implement `train_step` and `val_step` functions in `train.py`. These functions perform various tasks related to model training and validation.

### 3.2 Run the train script

- Execute the `train.py` script, which provides a basic setup for training a neural network for road segmentation.

### 3.3 Observe the results in Tensorboard

- Visualize training and validation results using Tensorboard. You can run Tensorboard with the following command:

tensorboard --logdir=<train_dir>

### 3.4 Epochs and train steps

- Determine the number of training steps per epoch and the total number of training steps.

### 3.5 Metrics

- Explore different metrics beyond accuracy, such as precision and recall, and consider when accuracy may not be the most suitable metric for certain tasks.

### 3.6 Implement U-net

- Implement the U-net architecture, making the specified adjustments in `segmentation_models.py`. Answer questions related to transposed convolution, trainable parameters, model behavior, activation functions, skip connections, and modifications for classification or image transformation tasks.

Train your model and evaluate if it improves over the original model.

## How to Run

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies (TensorFlow 2.x).
4. Execute the training script:

python3 train.py <train_dir>
Replace `<train_dir>` with a directory name of your choice to store the results.
