from typing import List
from DenseLayer import DenseLayer
class Sequntial:
    def __init__(self, layers: List[DenseLayer], input_size: int):
        self.layers = layers
        self.learning_rate = 0
        self.loss = None
        self.optimizer = None
        self.outputs = [
                            [0.0 for _ in range(layers[layer_i].n_neurons)]
                            for layer_i in range(len(layers))
                        ]
        if not layers:
            raise Exception("Layers cannot be empty!")
        layers[0].compile(input_size)
        for i in range(1, len(layers)):
            layers[i].compile(layers[i - 1].n_neurons)

    @staticmethod
    def _round(vector: List[float], precision = 7):
        for i in range(len(vector)):
            vector[i] = round(vector[i], precision)
    def compile(self, loss: str, optimizer: str, learning_rate: float):
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
    def step(self, X, Y):
        self._round(X)
        self._round(Y)
        # computing neurons' outputs in each layer while moving forward
        def forward(X):
            for layer_i, layer in enumerate(self.layers):
                layer_input = X if layer_i == 0 else self.outputs[layer_i - 1]
                for neuron_i, neuron in enumerate(layer.layer):
                    # vector multiplication without parallelization
                    # the last weight is always bias
                    self.outputs[layer_i][neuron_i] = neuron[-1] + sum([layer_input[input_i] * neuron[input_i]
                                                                   for input_i in range(len(layer_input))])
                self._round(self.outputs[layer_i])
            output = self.outputs[-1]
            return output
        def backward(outputs, X, Y):
            if not (outputs and Y):
                raise Exception("Outputs OR/AND Y are empty!")
            signals = {}
            deltas = [
                        [
                            [0.0 for _ in range(len(weights))]
                            for weights in self.layers[layer_i].layer]
                            for layer_i in range(len(self.layers))
                        ]
            n_layers = len(outputs)
            for y_i, y in enumerate(Y):
                # Calculate errors for each output neuron:
                # Cost function = (output-y)^2
                output = outputs[-1][y_i]
                error = 2 * (output - y)
                destination = (n_layers - 1, y_i)
                signals[destination] = [error]
            for layer_i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_i].layer
                for neuron_i in range(0, len(layer)):
                    derivative_so_far = sum(signals[(layer_i, neuron_i)])
                    weights = layer[neuron_i]
                    for weight_i, weight in enumerate(weights):
                        # the last weight is always bias:
                        if weight_i == len(weights) - 1:
                            b_delta = derivative_so_far
                            deltas[layer_i][neuron_i][weight_i] = b_delta
                            continue
                        # compute the change of each weight and put them in 'deltas':
                        matching_input = 0.0
                        if layer_i - 1 >= 0:
                            matching_input = outputs[layer_i - 1][weight_i]
                        else:
                            # hit the input layer:
                            matching_input = X[weight_i]
                        w_delta = derivative_so_far * matching_input
                        deltas[layer_i][neuron_i][weight_i] = w_delta
                        # propagate weight's derivative to the previous layer:
                        if layer_i - 1 >= 0:
                            destination = (layer_i - 1, weight_i)
                            if not destination in signals:
                                signals[destination] = []
                            signals[destination].append(derivative_so_far * weight)
            return deltas
        output = forward(X)
        loss = sum([(y_hat - y) ** 2 for y_hat, y in zip(output, Y)]) / len(Y)
        deltas = backward(self.outputs, X, Y)
        return [deltas, loss]
    def fit(self, X, Y):
        match self.loss:
            case 'MSE' | 'mse':
                pass
            case _:
                raise Exception("Unknown loss function")
        match self.optimizer:
            # Stochastic gradient descent:
            case 'SGD' | 'sgd':
                for instance_i in range(len(Y)):
                    deltas, loss = self.step(X[instance_i], Y[instance_i])
                    if instance_i % 500 == 0:
                        print("\n")
                        print(str(instance_i) + " / " + str(len(Y)))
                        print("Y\t", Y[instance_i])
                        print("Output\t", self.outputs[-1])
                        print("Loss\t", loss)
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
