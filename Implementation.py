from NeuralNetwork import NeuralNetwork
import numpy as np


class Implementation:
    def __init__(self, input_size, output_size, hidden_layers, hidden_activation, output_activation, epochs, learning_rate):
        self.nn = NeuralNetwork(
            input_size, output_size, hidden_layers, hidden_activation, output_activation)
        self.epochs = epochs
        self.learning_rate = learning_rate

    def start(self, X, y):
        self.nn.train(X, y, self.epochs, self.learning_rate)
        print("Entrenamiento completado.")
        return self.nn

    def test(self, X_test):
        predictions = self.nn.predict(X_test)
        return predictions
