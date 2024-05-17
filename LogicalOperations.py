import numpy as np


class LogicalOperations:
    def __init__(self):
        self.operations = {
            1: ("AND", self.and_operation),
            2: ("OR", self.or_operation),
            3: ("NOT", self.not_operation),
            4: ("XOR", self.xor_operation)
        }

    def get_operation(self, choice):
        return self.operations.get(choice, (None, None))

    def validate_dimensions(self, X, input_size, y, output_size):
        if X.shape[1] != input_size:
            raise ValueError(
                f"El tamaño de entrada debe ser {input_size}, pero se proporcionó {X.shape[1]}")
        if y.shape[1] != output_size:
            raise ValueError(
                f"El tamaño de salida debe ser {output_size}, pero se proporcionó {y.shape[1]}")
        return X, y

    def and_operation(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])
        return X, y

    def or_operation(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [1]])
        return X, y

    def not_operation(self):
        X = np.array([[0], [1]])
        y = np.array([[1], [0]])
        return X, y

    def xor_operation(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        return X, y

    def get_operation_and_validate(self, choice, input_size, output_size):
        name, func = self.get_operation(choice)
        if func is None:
            raise ValueError("Operación no válida.")
        X, y = func()
        return self.validate_dimensions(X, input_size, y, output_size)
