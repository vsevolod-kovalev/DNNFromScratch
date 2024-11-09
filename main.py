from MNIST import MNIST
from Sequntial import Sequntial
from DenseLayer import DenseLayer

def to_onehot(size: int, y: int):
    return [1.0 if _ == y else 0.0 for _ in range(size)]
def main():
    r = MNIST('datasets/mnist/mnist_train.csv', 'datasets/mnist/mnist_test.csv')
    r.load_data()
    (x_train, y_train), (x_test, y_test) = r.data
    r.print_sample(x_train[0], y_train[0])
    y_train_onehotted = []
    for y in y_train:
        y_train_onehotted.append(
            to_onehot(10, y)
        )
    model = Sequntial([
        DenseLayer(10),
        DenseLayer(10)
    ], input_size = len(x_train[0]))
    model.compile('mse', 'sgd', 0.001)
    model.fit(x_train, y_train_onehotted)

if __name__ == "__main__":
    main()