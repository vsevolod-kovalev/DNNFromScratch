# Neural Network from Scratch in Python

## Introduction

As a computer science student passionate about machine learning and mobile development, I set out to recreate TensorFlow's Sequential neural network model using only Python 3 and its standard libraries. This project deepened my understanding of machine learning by allowing me to implement the fundamental components of deep learning from the ground up.

## Problem It Solves

This project addresses the problem of **handwritten digit recognition**. By training a neural network on the MNIST dataset, the model learns to classify handwritten digits from 0 to 9.
<p align="center">
  <img src="https://github.com/user-attachments/assets/87410667-7c43-4b66-875e-a4a0728f2b1e" width="270" alt="image">
  <img src="https://github.com/user-attachments/assets/c2587af2-3c10-4891-a294-cb9517974bdf" width="270" alt="image">
  <img src="https://github.com/user-attachments/assets/def98a2f-82ee-41bc-8a5a-b5ae7a26474f" width="270" alt="image">
</p>
<p align="center">
  <em>Figure: Accurate predictions of three sample digits following training on the MNIST dataset. The neural network architecture with three hidden layers with ReLU activation functions and a Softmax output layer to classify digits with high confidence.</em>
</p>

## Features

- **Activation Functions**:
  - ReLU (Rectified Linear Unit)
  - Leaky ReLU
  - Sigmoid
  - Softmax (with Categorical Cross-Entropy)
- **Loss Functions**:
  - Mean Squared Error (MSE)
  - Categorical Cross-Entropy (CCE)
  - Binary Cross-Entropy (BCE)
- **Optimizer**:
  - Stochastic Gradient Descent (SGD)
- **Dataset**:
  - MNIST handwritten digits dataset.

## Project Structure

```
├── datasets/
│   └── mnist/
│       ├── mnist_train.csv
│       └── mnist_test.csv
├── demo.py
├── DenseLayer.py
├── main.py
├── MNIST.py
├── Sequential.py
└── README.md
```
- `datasets/mnist/`: Contains the MNIST dataset in CSV format.
- `demo.py`: Demonstrates the functionality using TensorFlow and NumPy for comparison.
- `DenseLayer.py`: Implements the dense (fully connected) layer.
- `main.py`: Main script to train and test the neural network.
- `MNIST.py`: Handles loading and preprocessing of the MNIST dataset.
- `Sequential.py`: Mimics TensorFlow's Sequential model to build the network.
- `README.md`: Project documentation.

## Dependencies
Although the project avoids high-level libraries, some standard Python modules are essential:

### 1. `random`
Initializing all weights to the same values causes the "symmetry problem," where neurons within a layer learn identical features or gradients, preventing efficient learning. For instance, setting all weights to 0.2 causes similar output values and high loss after the first epoch, as shown in the attached example (screenshot). To break symmetry, I used Python’s `random` module, which employs the Mersenne Twister to generate pseudo-random numbers. Initializing weights in the range `[-0.1, 0.1]` breaks symmetry from the start, facilitating faster convergence. Additionally, `random` is used to shuffle the data after each epoch, ensuring the model doesn’t learn any sequential ordering from the dataset. Writing a custom random generator is complex and outside the scope of this project.

<p align="center">
  <img width="589" alt="Epoch Progress" src="https://github.com/user-attachments/assets/8c6d0992-77eb-483d-a8b3-30159076f26e">
</p>

<p align="center">
  <em>Figure: The graph illustrates the importance of random weight initialization. Without it, neural networks can suffer from high and stagnant loss levels due to the <strong>symmetry problem</strong>, which prevents effective learning.</em>
</p>


### 2. `math`
Some activation and loss functions require exponentials and natural logarithms. For example, implementing the softmax activation function with Categorical Cross Entropy (CCE) ensures output probabilities sum to `1.0`, indicating the likelihood of each class. During backpropagation, calculating the combined derivative of softmax and CCE simplifies to `y_hat − y`, eliminating the need to compute the derivative of softmax independently. Both the `log` and `exp` functions from the `math` library are essential here. While implementing `exp` is manageable, `log` requires either Taylor series expansion or complex algorithms, which are beyond the focus of this project. Therefore, I opted for the built-in `math` implementations.

### 3. `time`
The `time` module is used solely for tracking the time taken to complete each learning step. This measurement is not essential for the model's functionality but is helpful for assessing training efficiency.

## Usage
1. Clone the repository:
`git clone https://github.com/vsevolod-kovalev/DNNFromScratch`
2. Navigate to the project directory:
`cd DNNFromScratch`
3. Run the main script:
`python main.py`
This will train the neural network on the MNIST dataset and display sample predictions.

## Implementation Details

* Neural Network Architecture
The network is built using the Sequential model, similar to TensorFlow's API. Layers are added in sequence, and each layer is a DenseLayer object specifying the number of neurons and activation function.

## Activation Functions

* **ReLU**: This function returns the input itself if it is greater than zero, otherwise, it returns zero. It is used to introduce non-linearity to the model.
* **Leaky ReLU**: Similar to ReLU, but instead of returning zero for negative inputs, it returns a small negative value proportional to the input. This slight value ensures that Leaky ReLU never has a gradient of zero.
* **Sigmoid**: This function maps any input value into the range between 0 and 1. The outputs are shaped like an "S".
* **Softmax**: Used primarily in the output layer of a classifier, this function converts logits into probabilities by taking the exponentials of each output and then normalizing these values by dividing by the sum of all the exponentials. This ensures that the output values are in the range (0, 1) and sum up to 1, making them interpretable as probabilities.
## Loss Functions

* **Mean Squared Error (MSE)**: Commonly used for regression tasks.
* **Categorical Cross-Entropy (CCE)**: Suitable for multi-class classification with one-hot encoded targets.
* **Binary Cross-Entropy (BCE)**: Used for binary classification tasks.

## Optimizer

* **Stochastic Gradient Descent (SGD)**: Updates weights incrementally for each training sample, which can lead to faster convergence on large datasets.
