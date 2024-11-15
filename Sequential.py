# for data shuffling
import random
import time
import math

from typing import List, Tuple
from DenseLayer import DenseLayer

LRELU_ALPHA = 0.01
Y_EPSILON = 0.01
SOFTMAX_EPSILON = 0.01
SIGMOID_Z_MAX = 30

def sanitized_log(y_hat: float) -> float:
    return math.log(max(y_hat, Y_EPSILON))

# Activation functions
def reLU(z: float, derivative: bool = False) -> float:
    if derivative:
        return 1.0 if z > 0 else 0.0
    return max(0, z)

def LreLU(z: float, derivative: bool = False) -> float:
    if derivative:
        return 1.0 if z > 0 else LRELU_ALPHA
    return z if z > 0 else LRELU_ALPHA * z

def sigmoid(z: float, derivative: bool = False) -> float:
    sig = 0.0
    if abs(z) > SIGMOID_Z_MAX:
        sig = 1.0 if z > 0 else 0.0
    else:
        sig = 1.0 / (1.0 + math.exp(-z))
    if derivative:
        return sig * (1.0 - sig)
    return sig

def softmax(Z: List[float]) -> List[float]:
    # Shift every z by z_max to prevent overflow
    z_max = max(Z)
    z_exp = [math.exp(z - z_max) for z in Z]
    z_sum = sum(z_exp)
    A = [z / z_sum for z in z_exp]
    return A

# Loss functions
def MSE(Y_hat, Y) -> Tuple[float, List[float]]:
    loss = sum([(y_hat - y) ** 2 for y_hat, y in zip(Y_hat, Y)]) / len(Y)
    loss_gradient = [2 * (y_hat - y) for y_hat, y in zip(Y_hat, Y)]
    return loss, loss_gradient

def BCE(Y_hat: List[float], Y: List[float]) -> Tuple[float, List[float]]:
    loss = -sum(
        y * sanitized_log(y_hat) + (1 - y) * sanitized_log(1 - y_hat)
        for y_hat, y in zip(Y_hat, Y)
    ) / len(Y)
    loss_gradient = [
        -(y / max(y_hat, Y_EPSILON)) + (1 - y) / max(1 - y_hat, Y_EPSILON)
        for y_hat, y in zip(Y_hat, Y)
    ]
    return loss, loss_gradient

def CCE(Y_hat: List[float], Y: List[float], with_softmax: bool = False) -> Tuple[float, List[float]]:
    loss = -sum(y * sanitized_log(y_hat) for y_hat, y in zip(Y_hat, Y)) / len(Y)
    if with_softmax:
        loss_gradient = [y_hat - y for y_hat, y in zip(Y_hat, Y)]
    else:
        loss_gradient = [-y / max(y_hat, Y_EPSILON) for y_hat, y in zip(Y_hat, Y)]
    return loss, loss_gradient

class Sequential:
    # Helper functions to create zeroed structures
    @staticmethod
    def zero_layer_neuron(layers: List[List[float]]):
        return [[0.0 for _ in range(layer.n_neurons)] for layer in layers]

    @staticmethod
    def zero_layer_neuron_weight(layers: List[List[float]]):
        return [
            [[0.0 for _ in range(len(neuron))]
             for neuron in layer.layer]
            for layer in layers
        ]

    def __init__(self, layers: List[DenseLayer], input_size: int):
        self.layers = layers
        self.learning_rate = 0
        self.loss_function = None
        self.optimizer = None

        if not layers:
            raise Exception("Layers cannot be empty!")

        layers[0].compile(input_size)
        for i in range(1, len(layers)):
            layers[i].compile(layers[i - 1].n_neurons)

    def compile(self, loss_function: str, optimizer: str, learning_rate: float):
        self.learning_rate = learning_rate
        self.loss_function = loss_function.lower()
        self.optimizer = optimizer.lower()

        # Ensure softmax is used with appropriate loss functions
        if (
            self.layers
            and self.layers[-1].activation == 'softmax'
            and self.loss_function not in [
                'cce', 'categorical_cross_entropy',
                'scce', 'sparse_categorical_cross_entropy']
        ):
            raise Exception("Softmax must be used with CCE or SCCE!")

    # TODO: merge with forward()
    def predict_digit(self, x: List[float]):
        pre_activations = self.zero_layer_neuron(self.layers)
        outputs = self.zero_layer_neuron(self.layers)

        for layer_i, layer in enumerate(self.layers):
            layer_input = x if layer_i == 0 else outputs[layer_i - 1]
            activation = layer.activation

            if activation == 'softmax':
                # Compute pre-activations for all neurons
                for neuron_i, neuron in enumerate(layer.layer):
                    z = neuron[-1] + sum(
                        layer_input[input_i] * neuron[input_i]
                        for input_i in range(len(layer_input))
                    )
                    pre_activations[layer_i][neuron_i] = z
                # Apply softmax to the entire layer's pre-activations
                outputs[layer_i] = softmax(pre_activations[layer_i])
            else:
                for neuron_i, neuron in enumerate(layer.layer):
                    z = neuron[-1] + sum(
                        layer_input[input_i] * neuron[input_i]
                        for input_i in range(len(layer_input))
                    )
                    pre_activations[layer_i][neuron_i] = z

                    match activation:
                        case '_no_activation':
                            a = z
                        case 'sigmoid' | 'sig':
                            a = sigmoid(z)
                        case 'relu':
                            a = reLU(z)
                        case 'lrelu' | 'l_relu' | 'leaky_relu' | 'leakyrelu':
                            a = LreLU(z)
                        case _:
                            raise Exception("Unknown activation function")
                    outputs[layer_i][neuron_i] = a

        y_hat = outputs[-1]
        max_probability = max(y_hat)
        max_probability_digit = y_hat.index(max_probability)
        return [max_probability, max_probability_digit]

    def step(self, X, Y):
        # Initialize local pre_activations and outputs for this sample
        pre_activations = self.zero_layer_neuron(self.layers)
        outputs = self.zero_layer_neuron(self.layers)

        def forward(X):
            for layer_i, layer in enumerate(self.layers):
                layer_input = X if layer_i == 0 else outputs[layer_i - 1]
                activation = layer.activation

                for neuron_i, neuron in enumerate(layer.layer):
                    z = neuron[-1] + sum(
                        layer_input[input_i] * neuron[input_i]
                        for input_i in range(len(layer_input))
                    )
                    pre_activations[layer_i][neuron_i] = z

                    match activation:
                        case '_no_activation':
                            a = z
                        case 'sigmoid' | 'sig':
                            a = sigmoid(z)
                        case 'relu':
                            a = reLU(z)
                        case 'lrelu' | 'l_relu' | 'leaky_relu' | 'leakyrelu':
                            a = LreLU(z)
                        case 'softmax':
                            if layer_i != len(self.layers) - 1:
                                raise Exception("Softmax cannot be used as an activation function for a hidden layer!")
                            # Apply softmax after all pre-activations are calculated
                            if neuron_i == len(layer.layer) - 1:
                                outputs[layer_i] = softmax(pre_activations[layer_i])
                            continue
                        case _:
                            raise Exception("Unknown activation function")
                    outputs[layer_i][neuron_i] = a
            return outputs[-1]

        def backward(pre_activations, outputs, X, Y, loss_gradient):
            if not (outputs and Y and pre_activations):
                raise Exception("Outputs, Pre-Activations, or Y are empty!")
            signals = {}
            deltas = self.zero_layer_neuron_weight(self.layers)
            n_layers = len(outputs)

            # Calculate loss derivatives for each neuron in the output layer
            for y_i, y in enumerate(Y):
                loss_derivative = loss_gradient[y_i]
                destination = (n_layers - 1, y_i)
                signals[destination] = [loss_derivative]

            for layer_i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_i].layer
                activation = self.layers[layer_i].activation

                for neuron_i in range(len(layer)):
                    derivative_so_far = sum(signals.get((layer_i, neuron_i), [0]))
                    if derivative_so_far == 0.0:
                        continue

                    weights = layer[neuron_i]
                    z = pre_activations[layer_i][neuron_i]

                    match activation:
                        case '_no_activation':
                            activation_gradient = 1.0
                        case 'softmax':
                            activation_gradient = 1.0
                        case 'sigmoid' | 'sig':
                            activation_gradient = sigmoid(z, derivative=True)
                        case 'relu':
                            activation_gradient = reLU(z, derivative=True)
                        case 'lrelu' | 'l_relu' | 'leaky_relu' | 'leakyrelu':
                            activation_gradient = LreLU(z, derivative=True)
                        case _:
                            raise Exception("Unknown activation function")

                    derivative_so_far *= activation_gradient

                    for weight_i, weight in enumerate(weights):
                        if weight_i == len(weights) - 1:
                            deltas[layer_i][neuron_i][weight_i] = derivative_so_far
                            continue

                        if layer_i - 1 >= 0:
                            matching_input = outputs[layer_i - 1][weight_i]
                        else:
                            matching_input = X[weight_i]

                        deltas[layer_i][neuron_i][weight_i] = derivative_so_far * matching_input

                        if layer_i - 1 >= 0:
                            destination = (layer_i - 1, weight_i)
                            if destination not in signals:
                                signals[destination] = []
                            signals[destination].append(derivative_so_far * weight)
            return deltas

        y_hat = forward(X)

        match self.loss_function:
            case 'mse' | 'mean_squared_error':
                loss, loss_gradient = MSE(y_hat, Y)
            case 'bce' | 'binary_cross_entropy':
                loss, loss_gradient = BCE(y_hat, Y)
            case 'cce' | 'categorical_cross_entropy':
                if self.layers and self.layers[-1].activation == 'softmax':
                    loss, loss_gradient = CCE(y_hat, Y, with_softmax=True)
                else:
                    loss, loss_gradient = CCE(y_hat, Y)
            case _:
                raise Exception('Unknown loss function')

        deltas = backward(pre_activations, outputs, X, Y, loss_gradient)
        return deltas, loss, y_hat

    def fit(self, X, Y, epochs: int = 1) -> None:
        def apply_deltas(deltas) -> None:
            for layer_i, layer in enumerate(self.layers):
                for neuron_i, neuron in enumerate(layer.layer):
                    for w_i in range(len(neuron)):
                        neuron[w_i] -= self.learning_rate * deltas[layer_i][neuron_i][w_i]

        def print_learning_step_info(start_time: float, loss: float, class_i: int, epoch_i: int, epochs: int, num_samples: int, y, y_hat) -> None:
            time_elapsed = time.time() - start_time
            progress = (class_i + 1) / num_samples
            bar_length = 20
            filled_length = int(bar_length * progress)
            progress_bar = f"[{'#' * filled_length}{'-' * (bar_length - filled_length)}]"
            print("\nEpoch Progress")
            print("========================================")
            print(f"Epoch {epoch_i} of {epochs}")
            print(f"Progress: {progress_bar} {progress * 100:.2f}%")
            print(f"Sample {class_i + 1} of {num_samples}")
            print(f"Time Elapsed: {time_elapsed:.3f} seconds")
            print("========================================")
            print("  Target:", " | ".join([f"{val:.3f}" for val in y]))
            print("  Output:", " | ".join([f"{val:.3f}" for val in y_hat]))
            print(f"  Loss: {loss:.5f}")
            print("========================================\n")

        num_samples = len(Y)
        current_time = time.time()
        print_info_every = 500

        for epoch_i in range(1, epochs+1):
            # Shuffle data at the beginning of each epoch
            combined = list(zip(X, Y))
            random.shuffle(combined)
            X_shuffled, Y_shuffled = zip(*combined)

            match self.optimizer:
                case 'sgd' | 'stochastic_gradient_descent':
                    for class_i in range(num_samples):
                        deltas, loss, y_hat = self.step(X_shuffled[class_i], Y_shuffled[class_i])
                        if class_i % print_info_every == 0:
                            print_learning_step_info(current_time, loss, class_i, epoch_i, epochs, num_samples, Y_shuffled[class_i], y_hat)
                            current_time = time.time()
                        apply_deltas(deltas)
                case _:
                    raise Exception("Unknown optimizer")