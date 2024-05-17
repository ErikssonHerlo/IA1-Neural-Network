from NeuralNetwork import NeuralNetwork
from LogicalOperations import LogicalOperations
from ActivationFunction import sigmoid_activation, tanh_activation, identity_activation, step_activation
import numpy as np


def main():
    print("Bienvenido al configurador de Redes Neuronales!")
    input_size = int(input("Ingrese el número de entradas: "))
    output_size = int(input("Ingrese el número de salidas: "))
    hidden_layers = list(map(int, input(
        "Ingrese el número de neuronas en cada capa oculta, separadas por comas: ").split(',')))
    activation_hidden = input(
        "Ingrese la función de activación para las capas ocultas (sigmoid/tanh): ").lower() or 'sigmoid'
    activation_output = input(
        "Ingrese la función de activación para la capa de salida (sigmoid/identity): ").lower() or 'sigmoid'

    nn = NeuralNetwork(input_size, output_size, hidden_layers,
                       activation_hidden, activation_output)

    print("Seleccione la operación lógica para la salida:")
    print("1. AND")
    print("2. OR")
    print("3. NOT")
    print("4. XOR")
    logic_choice = int(
        input("Ingrese el número correspondiente a la operación lógica: "))

    logical_operations = LogicalOperations()
    try:
        X, y = logical_operations.get_operation_and_validate(
            logic_choice, input_size, output_size)
    except ValueError as e:
        print(e)
        return

    epochs = int(input("Ingrese el número de épocas para el entrenamiento: "))
    learning_rate = float(input("Ingrese la tasa de aprendizaje: "))

    nn.train(X, y, epochs, learning_rate)

    print("Entrenamiento completado. Ingrese los datos para la predicción.")
    X_test = np.array(eval(input(
        "Ingrese la matriz de características (X) para predicciones en formato de lista de listas: ")))

    predictions = nn.predict(X_test)
    print("Predicciones:", predictions)


if __name__ == "__main__":
    main()
