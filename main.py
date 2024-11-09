from MNIST import MNIST
from Sequntial import Sequntial
from DenseLayer import DenseLayer

def main():
    r = MNIST('datasets/mnist/mnist_train.csv', 'datasets/mnist/mnist_test.csv')
    r.load_data()
    (x_train, y_train), (x_test, y_test) = r.data

    r.print_sample(x_train[0], y_train[0])

    model = Sequntial([
        DenseLayer(8),
        DenseLayer(2)
    ], 3)
    model.compile('mse', 0.01)
    # model.fit_sample(x_train[0], y_train[0])
    model.fit_sample([1,1,1], [300,300])

if __name__ == "__main__":
    main()

    