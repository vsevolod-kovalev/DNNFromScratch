from typing import List
from DenseLayer import DenseLayer
class Sequntial:
    def __init__(self, layers: List[DenseLayer], input_size: int):
        self.layers = layers
        self.learning_rate = 0
        self.loss = 'mse'
        self.layer_outputs = [
                            [0.0 for _ in range(layers[layer_i].n_neurons)]
                            for layer_i in range(len(layers))
                        ]
        print(self.layer_outputs)
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
                layer_input = X if layer_i == 0 else self.layer_outputs[layer_i-1]
                print("Layer input", layer_i, layer_input)
                print(layer.layer)
                for neuron_i, neuron in enumerate(layer.layer):
                    # vector multiplication without parallelization
                    self.layer_outputs[layer_i][neuron_i] = neuron[-1] + sum([layer_input[input_i] * neuron[input_i]
                                                            for input_i in range(len(layer_input))])
        forward(X)
        print(self.layer_outputs)



        


        

