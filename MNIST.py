class MNIST:
    def __init__(self, train_path: str, test_path: str):
        self.data = []
        self.train_path = train_path
        self.test_path = test_path
    def load_data(self) -> None:
        X_train, Y_train = [], []
        X_test, Y_test = [], []
        def fillXY(X, Y, path: str) -> None:
            f = open(path, "r")
            for line in f:

                line = line.strip().split(',')
                if not line or not line[0].isdigit():
                    continue
                X.append([int(pixel) for pixel in line[1:]])
                Y.append(int(line[0]))
            f.close()
        try:
            fillXY(X_train, Y_train, self.train_path)
            fillXY(X_test, Y_test, self.test_path)
            self.data = [[X_train, Y_train], [X_test, Y_test]]
        except Exception as e:
            raise Exception("Error reading files!") from e
    
r = MNIST('datasets/mnist/mnist_train.csv', 'datasets/mnist/mnist_test.csv')
r.load_data()
(x_train, y_train), (x_test, y_test) = r.data