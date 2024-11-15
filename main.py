from MNIST import MNIST
from Sequential import Sequential
from DenseLayer import DenseLayer

def to_onehot(size: int, y: int):
    return [1.0 if _ == y else 0.0 for _ in range(size)]

def main():
    # Load data
    r = MNIST('datasets/mnist/mnist_train.csv', 'datasets/mnist/mnist_test.csv')
    r.load_data()
    (x_train, y_train), (x_test, y_test) = r.data
    y_train_onehotted = [to_onehot(10, y) for y in y_train]
    
    # Define and compile the model
    model = Sequential([
        DenseLayer(10, activation='relu'),
        DenseLayer(30, activation='relu'),
        DenseLayer(10, activation='softmax')
    ], input_size=len(x_train[0]))
    model.compile('cce', 'sgd', 0.01)
    
    # Train the model
    model.fit(x_train, y_train_onehotted, epochs=2)
    
    # Test on a sample and display results
    print("\nDisplaying sample image with prediction details:\n")
    for i in range(5):
        r.print_sample(x_test[i], y_test[i])
        
        # Get model's prediction
        prob, pred = model.predict_digit(x_test[i])
        actual = y_test[i]
        
        # Display prediction results
        print("Prediction Result")
        print("=================")
        print(f"Predicted Digit : {pred}")
        print(f"Prediction Prob : {prob:.5f}")
        print(f"Actual Digit    : {actual}")
        print("=================")
        
        # Correctness check
        if pred == actual:
            print("✅ Prediction is correct!")
        else:
            print("❌ Prediction is incorrect.")
        print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    main()
