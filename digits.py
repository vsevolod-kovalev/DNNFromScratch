# This file demonstrates the desired functionality of a neural network model
# for recognizing handwritten digits using the MNIST dataset. 

import tensorflow as tf
import numpy as np
def print_sample(x, y):
    for row in x:
        print("".join(f"{value:5.2f}" if value != 0 else "  0  " for value in row))
    print("\nTarget:\t", y)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = np.array([[1.0 if i == value else 0.0 for i in range(10)] for value in y_train])
y_test = np.array([[1.0 if i == value else 0.0 for i in range(10)] for value in y_test])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))
model.fit(x_train, y_train, epochs=5)
print_sample(x_test[0], y_test[0])
prediction = model.predict(x_test[0].reshape(1, 28, 28))
predicted_class = prediction.argmax()
print("Prediction (probabilities for each class):")
for i, prob in enumerate(prediction[0]):
    print(f"Class {i}: {prob:.2f}")