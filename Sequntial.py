from typing import List
from DenseLayer import DenseLayer
class Sequntial:
    def __init__(self, layers: List[DenseLayer], input_size: int):
        self.layers = layers
        self.learning_rate = 0
        self.loss = 'mse'
        self.outputs = [
                            [0.0 for _ in range(layers[layer_i].n_neurons)]
                            for layer_i in range(len(layers))
                        ]
        print(self.outputs)
        if not layers:
            raise Exception("Layers cannot be empty!")
        layers[0].compile(input_size)
        for i in range(1, len(layers)):
            layers[i].compile(layers[i - 1].n_neurons)

    def compile(self, loss: str, learning_rate: float):
        self.learning_rate = learning_rate
        self.loss = loss
    def fit_sample(self, X, Y):
        # computing neurons' outputs in each layer while moving forward
        def forward(X):
            for layer_i, layer in enumerate(self.layers):
                layer_input = X if layer_i == 0 else self.outputs[layer_i-1]
                print("Layer input", layer_i, layer_input)
                print(layer.layer)
                for neuron_i, neuron in enumerate(layer.layer):
                    # vector multiplication without parallelization
                    self.outputs[layer_i][neuron_i] = neuron[-1] + sum([layer_input[input_i] * neuron[input_i]
                                                            for input_i in range(len(layer_input))])
        def backward(outputs, X,  Y):
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
                # The cost function used => (output-y)^2
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
                        # compute the change of every weight and put it in 'deltas':
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

        forward(X)
        deltas = backward(self.outputs, X, Y)
        print("X: ", X)
        print("Y: ", Y)
        print("OUTPUTS: ", self.outputs)
        print("DELTAS: ", deltas)
        



        


        

