# DNNFromScratch

**DNNFromScratch** is a fully custom-built deep neural network framework developed in raw Python for analyzing handwritten digits. This project is built entirely from scratch without the use of external libraries like TensorFlow, PyTorch, or NumPy, allowing users to gain a deeper understanding of neural network architecture and training mechanics.

## 🔍 Overview

DNNFromScratch emulates the functionality of TensorFlow’s **Sequential** model, enabling users to stack layers and fully control neural network components, including layers, activation functions, optimizers, and learning rates. This project is perfect for those interested in exploring deep learning fundamentals, as it covers forward and backward propagation, custom optimizers, and weight updates without high-level abstractions.

## 🚀 Key Features

- **Layer Stacking:** Mimics TensorFlow’s Sequential model for straightforward layer stacking and model definition.
- **Customizable Dense Layers:** Enables flexible network architectures by allowing users to specify the number of neurons per layer.
- **Activation Functions:** Supports several popular activation functions, including:
  - ReLU
  - Leaky ReLU
  - Sigmoid
  - Softmax
- **Optimizers:** Implements fundamental optimization techniques, such as:
  - Gradient Descent
  - Stochastic Gradient Descent (SGD)
  - Mini-batch Gradient Descent
- **Configurable Loss Functions:** Supports common loss functions for classification tasks, including:
  - Mean Squared Error (MSE)
  - Categorical Cross-Entropy
  - Sparse Categorical Cross-Entropy
- **Manual Forward and Backward Propagation:** Provides complete control over forward and backward passes, allowing users to see how activations and gradients are calculated step-by-step.
- **Weight Update Mechanics:** Applies computed weight deltas and bias adjustments after each instance or mini-batch, illustrating the mechanics of weight updates.
- **Predictive Capabilities:** A `predict` method for generating predictions on new samples, making it easy to evaluate model performance.

## 📂 File Structure

- `Sequential.py`: Contains the core Sequential model class for stacking layers and managing forward/backward propagation.
- `DenseLayer.py`: Defines dense layers, each with customizable neurons and activation functions.
- `MNIST.py`, `digits.py`: Sample datasets for testing the model.
- `main.py`: Example usage of the framework.
