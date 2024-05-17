from Implementation import Implementation
import numpy as np


def main():
    print("Bienvenido al configurador de Redes Neuronales!")
    # input_size = int(input("Ingrese el número de entradas: "))
    # output_size = int(input("Ingrese el número de salidas: "))
    # hidden_layers = list(map(int, input(
    #     "Ingrese el número de neuronas en cada capa oculta, separadas por comas: ").split(',')))
    # hidden_activation = input(
    #     "Ingrese la función de activación para las capas ocultas (relu/sigmoid/tanh): ").lower() or 'sigmoid'
    # output_activation = input(
    #     "Ingrese la función de activación para la capa de salida (sigmoid/identity/step): ").lower() or 'sigmoid'
    # epochs = int(
    #     input("Ingrese el número de épocas para el entrenamiento: "))
    # learning_rate = float(input("Ingrese la tasa de aprendizaje: "))

    # X = np.array(eval(input(
    #     "Ingrese la matriz de características de entrenamiento (X) en formato de lista de listas: ")))
    # y = np.array(eval(input(
    #     "Ingrese la matriz de resultados esperados (y) en formato de lista de listas: ")))
    input_size = 2
    output_size = 1
    hidden_layers = [2]
    hidden_activation = 'sigmoid'
    output_activation = 'sigmoid'
    epochs = 1000
    learning_rate = 0.2
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([[0], [0], [0], [1]])
    # y = np.array([[0], [1], [1], [1]])
    y = np.array([[0], [1], [1], [0]])
    implementation = Implementation(input_size, output_size, hidden_layers,
                                    hidden_activation, output_activation, epochs, learning_rate)
    nn = implementation.start(X, y)

    print("Entrenamiento completado. Realizando predicciones con los mismos datos de entrenamiento.")

    predictions = implementation.test(X)

    print("\nEntradas y Predicciones:")
    for i in range(len(X)):
        print(f"Entrada: {X[i]}, Predicción: {predictions[i]}")


if __name__ == "__main__":
    main()
