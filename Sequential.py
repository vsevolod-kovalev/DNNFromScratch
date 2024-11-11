# for debugging; remove before release:
import time

from typing import List
from DenseLayer import DenseLayer

ROUND_PRECISION = 5
LRELU_ALPHA = 0.01
MATH_E_20 = 2.71828182845904523536

# TODO: move activation functions into Sequential 
def reLU(z: float, derivative: bool = False) -> float:
    if derivative:
        return 1.0 if z > 0 else 0.0
    return max(0, z)
def LreLU(z: float, derivative: bool = False) -> float:
    if derivative:
        return 1.0 if z > 0 else LRELU_ALPHA
    return z if z > 0 else LRELU_ALPHA * z
def sigmoid(z: float, derivative: bool = False) -> float:
    sig = 1.0 / (1.0 + MATH_E_20 ** (-z))
    if derivative:
        return sig * (1.0 - sig)
    return sig

class Sequential:
    # New vector with shape:        Layer   x   Neuron
    @staticmethod
    def zero_layer_neuron(layers: List[List[float]]):
        return [[0.0 for _ in range(layers[layer_i].n_neurons)]
                     for layer_i in range(len(layers))]
    # New vector with shape:        Layer   x   Neuron  x   Weight
    @staticmethod
    def zero_layer_neuron_weight(layers: List[List[float]]):
        return [[[0.0 for _ in range(len(weights))]
                      for weights in layers[layer_i].layer]
                      for layer_i in range(len(layers))]
    @staticmethod
    def round_vector(vector: List[float], precision = ROUND_PRECISION):
        for i in range(len(vector)):
            vector[i] = round(vector[i], precision)
        return vector
    def __init__(self, layers: List[DenseLayer], input_size: int):
        self.layers = layers
        self.learning_rate = 0
        self.loss = None
        self.optimizer = None
        self.outputs = self.zero_layer_neuron(layers)
        self.pre_activations = self.zero_layer_neuron(layers)
        if not layers:
            raise Exception("Layers cannot be empty!")
        layers[0].compile(input_size)
        for i in range(1, len(layers)):
            layers[i].compile(layers[i - 1].n_neurons)
    def compile(self, loss: str, optimizer: str, learning_rate: float):
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
    def step(self, X, Y):
        # computing neurons' pre-activations and outputs in each layer
        def forward(X):
            for layer_i, layer in enumerate(self.layers):
                layer_input = X if layer_i == 0 else self.outputs[layer_i - 1]
                activation = layer.activation
                for neuron_i, neuron in enumerate(layer.layer):
                    # vector multiplication without parallelization
                    # the last weight is always bias
                    self.pre_activations[layer_i][neuron_i] = neuron[-1] + sum(
                                                                    [layer_input[input_i] * neuron[input_i]
                                                                    for input_i in range(len(layer_input))])
                    # compute the neuron's output based on the layer's activation function
                    z = self.pre_activations[layer_i][neuron_i]
                    a = 0.0
                    match activation.lower():
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
                    self.outputs[layer_i][neuron_i] = a
            # return the righmost (output) layer
            return self.outputs[-1]
        def backward(pre_activations, outputs, X, Y):
            if not (outputs and Y and pre_activations):
                raise Exception("Outputs OR/AND Pre-Activations OR/AND Y\tare empty!")
            signals = {}
            deltas = self.zero_layer_neuron_weight(self.layers)
            n_layers = len(outputs)
            # Calculate errors for each neuron in the output layer:
            for y_i, y in enumerate(Y):
                output = outputs[-1][y_i]
                error = 2 * (output - y)
                destination = (n_layers - 1, y_i)
                signals[destination] = [error]
            for layer_i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_i].layer
                activation = self.layers[layer_i].activation
                for neuron_i in range(0, len(layer)):
                    derivative_so_far = sum(signals.get((layer_i, neuron_i), [0]))
                    # Dead neuron found - do not propagate further:
                    if derivative_so_far == 0.0:
                        continue
                    weights = layer[neuron_i]
                    # compute δa / δz - activation gradient - based on the layer's activation function
                    z = pre_activations[layer_i][neuron_i]
                    activation_gradient = 0.0
                    match activation.lower():
                        case '_no_activation':
                            activation_gradient = 1.0
                        case 'sigmoid' | 'sig':
                            activation_gradient = sigmoid(z, derivative=True)
                        case 'relu':
                            activation_gradient = reLU(z, derivative=True)
                        case 'lrelu' | 'l_relu' | 'leaky_relu' | 'leakyrelu':
                            activation_gradient = LreLU(z, derivative=True)
                        case _:
                            raise Exception("Unknown activation function")
                    # apply the computed activation gradient to the neuron chain derivative
                    derivative_so_far *= activation_gradient
                    for weight_i, weight in enumerate(weights):
                        # compute δz / δw - the change of each weight - and put it in 'deltas':
                        # the last weight is always bias:
                        if weight_i == len(weights) - 1:
                            b_delta = derivative_so_far
                            deltas[layer_i][neuron_i][weight_i] = b_delta
                            continue
                        matching_input = 0.0
                        if layer_i - 1 >= 0:
                            matching_input = outputs[layer_i - 1][weight_i]
                        else:
                            # hit the input layer:
                            matching_input = X[weight_i]
                        w_delta = derivative_so_far * matching_input
                        deltas[layer_i][neuron_i][weight_i] = w_delta
                        # propagate the weight to the previous layer:
                        if layer_i - 1 >= 0:
                            destination = (layer_i - 1, weight_i)
                            if not destination in signals:
                                signals[destination] = []
                            signals[destination].append(derivative_so_far * weight)
            return deltas
        output = forward(X)
        loss = sum([(y_hat - y) ** 2 for y_hat, y in zip(output, Y)]) / len(Y)
        deltas = backward(self.pre_activations, self.outputs, X, Y)
        return [deltas, loss]
    def fit(self, X, Y):
        match self.loss.lower():
            case 'mse':
                pass
            case _:
                raise Exception("Unknown loss function")
        match self.optimizer.lower():
            # Stochastic gradient descent:
            case 'sgd':
                start_time = time.time()
                for instance_i in range(len(Y)):
                    deltas, loss = self.step(X[instance_i], Y[instance_i])
                    if instance_i % 500 == 0:
                        print("\n")
                        print(str(instance_i) + " / " + str(len(Y)))
                        time_elapsed = round(time.time() - start_time, ROUND_PRECISION)
                        start_time = time.time()
                        print("Time elapsed (secs)\t", time_elapsed)
                        print("Y\t", Y[instance_i])
                        print("Output\t", Sequential.round_vector(self.outputs[-1]))
                        print("Loss\t", round(loss, ROUND_PRECISION))
                        print("\n")
                    # update weights and biases after each instance:
                    # w = w - α * Δw
                    # b = w - α * Δb
                    for layer_i, layer in enumerate(self.layers):
                        for neuron_i, neuron in enumerate(layer.layer):
                            for w_i in range(len(neuron)):
                                neuron[w_i] -= self.learning_rate * deltas[layer_i][neuron_i][w_i]
            case _:
                raise Exception("Unknown optimizer")
