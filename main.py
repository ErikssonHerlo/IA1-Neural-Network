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

    # Configuración para Operaciones Binarias
    input_size = 2
    output_size = 1
    hidden_layers = [2]
    hidden_activation = 'sigmoid'
    output_activation = 'step'
    epochs = 1000
    learning_rate = 0.2
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])
    # y = np.array([[0], [1], [1], [1]])
    # y = np.array([[0], [1], [1], [0]])

    # # Configuración para Operaciones Numericas
    # input_size = 2
    # output_size = 1
    # hidden_layers = [2]
    # hidden_activation = 'tanh'
    # output_activation = 'identity'
    # epochs = 1000
    # learning_rate = 0.2
    # X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    # y = np.array([[3], [5], [7], [9]])

    if (output_activation == 'identity'):

        # Normalizar los datos manualmente
        X_normalized, X_mean, X_std = normalize(X)
        y_normalized, y_mean, y_std = normalize(y)
        implementation = Implementation(input_size, output_size, hidden_layers,
                                        hidden_activation, output_activation, epochs, learning_rate)
        nn = implementation.start(X_normalized, y_normalized)

        print("Entrenamiento completado. Realizando predicciones con los mismos datos de entrenamiento.")
        predictions_normalized = nn.predict(X_normalized)

        # Desnormalizar las predicciones
        predictions = denormalize(predictions_normalized, y_mean, y_std)
    else:
        implementation = Implementation(input_size, output_size, hidden_layers,
                                        hidden_activation, output_activation, epochs, learning_rate)
        nn = implementation.start(X, y)

        print("Entrenamiento completado. Realizando predicciones con los mismos datos de entrenamiento.")
        predictions = nn.predict(X)

    # Salida General
    print("\nEntradas y Predicciones:")
    for i in range(len(X)):
        print(f"Entrada: {X[i]}, Predicción: {predictions[i]}")

# Normalización y Desnormalización de Datos


def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std, mean, std


def denormalize(data, mean, std):
    return data * std + mean


if __name__ == "__main__":
    main()
