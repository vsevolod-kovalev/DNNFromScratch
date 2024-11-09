import random
class DenseLayer:
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.layer = []
    def compile(self, input_size: int):
        self.layer = [[random.uniform(-0.1, 0.1) for _ in range(input_size + 1)] for i in range(self.n_neurons)]