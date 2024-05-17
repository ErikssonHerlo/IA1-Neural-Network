import numpy as np
from ActivationFunction import sigmoid_activation, tanh_activation, identity_activation, step_activation


class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers, hidden_activation='sigmoid', output_activation='sigmoid'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_activation = self.get_activation_function(
            hidden_activation)
        self.hidden_activation_derivative = self.get_activation_derivative(
            hidden_activation)
        self.output_activation = self.get_activation_function(
            output_activation)
        self.output_activation_derivative = self.get_activation_derivative(
            output_activation)

        # Inicializar pesos y biases
        self.weights = []
        self.biases = []

        # Inicializar pesos y biases para las capas ocultas
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.rand(
                layer_sizes[i], layer_sizes[i+1]) - 0.5)
            self.biases.append(np.random.rand(layer_sizes[i+1]) - 0.5)

    def get_activation_function(self, name):
        if name == 'sigmoid':
            return sigmoid_activation
        elif name == 'tanh':
            return tanh_activation
        elif name == 'identity':
            return identity_activation
        elif name == 'step':
            return step_activation
        else:
            return sigmoid_activation  # Default

    def get_activation_derivative(self, name):
        if name == 'sigmoid':
            return sigmoid_activation.derivative
        elif name == 'tanh':
            return tanh_activation.derivative
        else:
            return sigmoid_activation.derivative  # Default

    def forward(self, X):
        self.activations = [X]
        input_to_layer = X
        for i in range(len(self.weights) - 1):
            output_from_layer = self.hidden_activation(
                np.dot(input_to_layer, self.weights[i]) + self.biases[i])
            self.activations.append(output_from_layer)
            input_to_layer = output_from_layer
        final_output = self.output_activation(
            np.dot(input_to_layer, self.weights[-1]) + self.biases[-1])
        self.activations.append(final_output)
        return final_output

    def backpropagate(self, X, y, learning_rate):
        output = self.forward(X)
        error = y - output
        deltas = [error * self.output_activation_derivative(output)]

        for i in range(len(self.weights) - 2, -1, -1):
            deltas.append(deltas[-1].dot(self.weights[i + 1].T) *
                          self.hidden_activation_derivative(self.activations[i + 1]))

        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * \
                self.activations[i].T.dot(deltas[i])
            self.biases[i] += learning_rate * np.sum(deltas[i], axis=0)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.backpropagate(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.forward(X)))
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        return self.forward(X)
