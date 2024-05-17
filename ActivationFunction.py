import numpy as np


class ActivationFunction:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative

    def __call__(self, x):
        return self.func(x)

# Definir las funciones de activación


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def identity(x):
    return x


def step(x):
    return np.where(x >= 0, 1, 0)


def step_derivative(x):
    return 0  # Step function derivative is not used in backpropagation


# Instanciar las funciones de activación
sigmoid_activation = ActivationFunction(sigmoid, sigmoid_derivative)
tanh_activation = ActivationFunction(tanh, tanh_derivative)
identity_activation = ActivationFunction(identity, lambda x: 1)
step_activation = ActivationFunction(step, step_derivative)
