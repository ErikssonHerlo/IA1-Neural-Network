# Implementación de Red Neuronal

Este proyecto implementa una red neuronal capaz de aprender operaciones lógicas como AND, OR y XOR. La red neuronal es configurable en términos de tamaño de entrada, tamaño de salida, capas ocultas y funciones de activación.

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Funciones de Activación](#funciones-de-activación)
   - [Sigmoide](#sigmoide)
   - [Tanh](#tanh)
   - [ReLU](#relu)
   - [Leaky ReLU](#leaky-relu)
   - [Step](#step)
3. [Clase Red Neuronal](#clase-red-neuronal)
   - [Inicialización](#inicialización)
   - [Propagación Hacia Adelante](#propagación-hacia-adelante)
   - [Propagación Hacia Atrás](#propagación-hacia-atrás)
   - [Entrenamiento](#entrenamiento)
   - [Predicción](#predicción)
4. [Funciones de Normalización y Desnormalización](#funciones-de-normalización-y-desnormalización)
5. [Combinaciones Recomendadas](#combinaciones-recomendadas)
   - [Capas Ocultas](#capas-ocultas)
   - [Capas de Salida](#capas-de-salida)
6. [Configuración de Ejemplo](#configuración-de-ejemplo)
7. [Uso](#uso)

## Introducción

Este proyecto permite crear, entrenar y probar una red neuronal para operaciones lógicas simples. Puedes configurar el número de entradas, salidas, capas ocultas y elegir entre varias funciones de activación.

## Funciones de Activación

### Sigmoide

- **Función:** 
  \[
  \sigma(x) = \frac{1}{1 + e^{-x}}
  \]
- **Derivada:**
  \[
  \sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
  \]
- **Características:**
  - Mapea cualquier valor de entrada a un rango entre 0 y 1.
  - Comúnmente utilizada en la capa de salida para problemas de clasificación binaria.
  - Problemas: Puede sufrir de desvanecimiento de gradiente y saturación.
- **Uso Recomendado:** Capa de salida para problemas de clasificación binaria.

### Tanh

- **Función:**
  \[
  \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  \]
- **Derivada:**
  \[
  \tanh'(x) = 1 - \tanh(x)^2
  \]
- **Características:**
  - Mapea los valores de entrada a un rango entre -1 y 1.
  - Preferida sobre la sigmoide para capas ocultas porque su salida está centrada en cero.
  - Problemas: También puede sufrir de desvanecimiento de gradiente.
- **Uso Recomendado:** Capas ocultas.

### ReLU (Rectified Linear Unit)

- **Función:**
  \[
  \text{ReLU}(x) = \max(0, x)
  \]
- **Derivada:**
  \[
  \text{ReLU}'(x) = \begin{cases} 
  1 & \text{si } x > 0 \\
  0 & \text{si } x \leq 0
  \end{cases}
  \]
- **Características:**
  - Mapea cualquier valor de entrada a 0 o el valor mismo si es positivo.
  - Muy popular para capas ocultas debido a su simplicidad y eficiencia.
  - Problemas: Puede sufrir del problema de "neurona muerta" cuando muchas unidades están inactivas.
- **Uso Recomendado:** Capas ocultas.

### Leaky ReLU

- **Función:**
  \[
  \text{Leaky ReLU}(x) = \begin{cases} 
  x & \text{si } x > 0 \\
  \alpha x & \text{si } x \leq 0
  \end{cases}
  \]
- **Derivada:**
  \[
  \text{Leaky ReLU}'(x) = \begin{cases} 
  1 & \text{si } x > 0 \\
  \alpha & \text{si } x \leq 0
  \end{cases}
  \]
- **Características:**
  - Similar a ReLU pero permite un pequeño gradiente cuando la entrada es negativa.
- **Uso Recomendado:** Capas ocultas.

### Step (Función Escalón)

- **Función:**
  \[
  \text{Step}(x) = \begin{cases} 
  1 & \text{si } x \geq 0 \\
  0 & \text{si } x < 0
  \end{cases}
  \]
- **Características:**
  - Produce una salida binaria.
  - No utilizada comúnmente en redes neuronales modernas debido a su derivada cero en todas partes excepto en cero.
- **Uso Recomendado:** No es común en redes neuronales modernas.

## Clase Red Neuronal

### Inicialización

La clase `NeuralNetwork` se inicializa con los siguientes parámetros:
- `input_size`: Número de neuronas de entrada.
- `output_size`: Número de neuronas de salida.
- `hidden_layers`: Lista que contiene el número de neuronas en cada capa oculta.
- `hidden_activation`: Función de activación para las capas ocultas (por defecto es `relu`).
- `output_activation`: Función de activación para la capa de salida (por defecto es `sigmoid` durante el entrenamiento, pero puede cambiarse a `step` para la predicción final).

```python
class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers, hidden_activation='relu', output_activation='sigmoid'):
```

## Inicialización de Pesos y Biases:

- Los pesos y biases se inicializan para cada capa de la red.
- Los pesos se inicializan con pequeños valores aleatorios, y los biases se inicializan en cero.

```python
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            self.biases.append(np.zeros(layer_sizes[i+1]))
```

2. **Funciones de Activación:**
   - Se seleccionan las funciones de activación apropiadas y sus derivadas según los parámetros proporcionados.

```python
        self.hidden_activation = self.get_activation_function(hidden_activation)
        self.hidden_activation_derivative = self.get_activation_derivative(hidden_activation)
        self.output_activation = self.get_activation_function(output_activation)
        self.output_activation_derivative = self.get_activation_derivative(output_activation)
        
```

### Propagación Hacia Adelante

El método `forward` calcula la salida de la red neuronal para una entrada dada `X`. Esto implica pasar la entrada a través de cada capa de la red y aplicar las funciones de activación.

```python
    def forward(self, X):
        self.activations = [X]
        input_to_layer = X
        for i in range(len(self.weights) - 1):
            output_from_layer = self.hidden_activation(np.dot(input_to_layer, self.weights[i]) + self.biases[i])
            self.activations.append(output_from_layer)
            input_to_layer = output_from_layer
        final_output = self.output_activation(np.dot(input_to_layer, self.weights[-1]) + self.biases[-1])
        self.activations.append(final_output)
        return final_output
```

### Propagación Hacia Atrás

El método `backpropagate` ajusta los pesos y biases basados en el error entre la salida predicha y la salida real. Este método realiza los siguientes pasos:

1. **Calcular el Error:**
   - El error se calcula como la diferencia entre la salida real `y` y la salida predicha `output`.

```python
        error = y - output
```

2. **Calcular las Deltas para la Capa de Salida:**
   - La delta para la capa de salida se calcula multiplicando el error por la derivada de la función de activación de la capa de salida.

```python
        deltas = [error * self.output_activation_derivative(output)]
```

3. **Calcular las Deltas para las Capas Ocultas:**
   - Las deltas para las capas ocultas se calculan propagando el error hacia atrás a través de la red, capa por capa.

```python
        for i in range(len(self.weights) - 2, -1, -1):
            deltas.append(deltas[-1].dot(self.weights[i + 1].T) * self.hidden_activation_derivative(self.activations[i + 1]))
        deltas.reverse()
```

4. **Actualizar Pesos y Biases:**
   - Los pesos y biases se actualizan utilizando las deltas calculadas y la tasa de aprendizaje.

```python
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * self.activations[i].T.dot(deltas[i])
            self.biases[i] += learning_rate * np.sum(deltas[i], axis=0)
```

### Entrenamiento

El método `train` entrena la red neuronal durante un número específico de épocas utilizando los datos de entrenamiento proporcionados y la tasa de aprendizaje.

```python
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.backpropagate(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.forward(X)))
                predictions = self.forward(X)
                print(f'Epoch {epoch}, Loss: {loss}')
                print("Predicciones (entrenamiento):")
                for i in range(len(X)):
                    print(f"Entrada: {X[i]}, Predicción: {predictions[i]}")
                print("\nPesos y biases:")
                for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                    print(f"Capa {i} - Pesos:\n{w}\nBiases:\n{b}")
```

### Predicción

El método `predict` utiliza la red neuronal entrenada para hacer predicciones en nuevos datos. La función `step` se aplica a la salida para convertirla en valores binarios.

```python
    def predict(self, X):
        raw_output = self.forward(X)
        return raw_output
```

## Funciones de Normalización y Desnormalización

### Normalización

La función `normalize` se utiliza para escalar los datos de entrada y salida de la red neuronal de modo que tengan una media de 0 y una desviación estándar de 1. Esto es importante porque ayuda a estabilizar el proceso de entrenamiento, asegurando que los gradientes no se vuelvan demasiado grandes o pequeños, lo que podría llevar a problemas de convergencia.

```python
def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std, mean, std
```

**Detalles de la función:**
- **Entrada:** 
  - `data`: Un array de NumPy que contiene los datos a normalizar.
- **Salida:** 
  - Un array de NumPy de los datos normalizados.
  - `mean`: La media de los datos originales.
  - `std`: La desviación estándar de los datos originales.
- **Proceso:**
  - Calcula la media y la desviación estándar de los datos.
  - Escala los datos restando la media y dividiendo por la desviación estándar.

### Desnormalización

La función `denormalize` se utiliza para revertir el proceso de normalización y devolver los datos a su escala original. Esto es útil para interpretar las predicciones de la red neuronal en el mismo rango de valores que los datos de entrada originales.

```python
def denormalize(data, mean, std):
    return data * std + mean
```

**Detalles de la función:**
- **Entrada:** 
  - `data`: Un array de NumPy que contiene los datos normalizados a desnormalizar.
  - `mean`: La media utilizada en el proceso de normalización.
  - `std`: La desviación estándar utilizada en el proceso de normalización.
- **Salida:** 
  - Un array de NumPy de los datos desnormalizados.
- **Proceso:**
  - Multiplica los datos normalizados por la desviación estándar y suma la media para revertir la normalización.

### Uso en Redes Neuronales

Cuando la función de activación de la capa de salida es `identity`, la red neuronal se espera que produzca una salida en la misma escala que los datos de entrada originales. Por esta razón, es crucial normalizar los datos antes de entrenar la red y desnormalizarlos después de obtener las predicciones. Esto asegura que el proceso de entrenamiento sea estable y que las predicciones puedan ser interpretadas correctamente.

## Combinaciones Recomendadas

### Capas Ocultas

1. **ReLU/Leaky ReLU:** Más comunes debido a su eficiencia y buen rendimiento en la práctica.
2. **Tanh:** Buenas para redes más antiguas o cuando las salidas necesitan estar centradas en cero.
3. **Sigmoide:** Menos comunes para capas ocultas debido a problemas de desvanecimiento de gradiente.

### Capas de Salida

1. **Sigmoide:** Para problemas de clasificación binaria.
2. **Identity:** Para problemas de regresión.
3. **Step:** Se aplica a la salida para convertirla en valores binarios, ya que solo devuelve valores 0 y 1.

## Configuración de Ejemplo

Para un problema de clasificación binaria (e.g., operación AND):

- **Entradas:** 2
- **Salidas:** 1
- **Capas Ocultas:** 2 capas con 2 neuronas cada una
- **Función de Activación para Capas Ocultas:** `sigmoid` (o `ReLU`)
- **Función de Activación para la Capa de Salida:** `sigmoid`
- **Épocas:** 1000
- **Tasa de Aprendizaje:** 0.1

Entrada de Ejemplo:
```plaintext
Bienvenido al configurador de Redes Neuronales!
Ingrese el número de entradas: 2
Ingrese el número de salidas: 1
Ingrese el número de neuronas en cada capa oculta, separadas por comas: 2,2
Ingrese la función de activación para las capas ocultas (sigmoid/tanh): sigmoid
Ingrese la función de activación para la capa de salida (sigmoid/identity): sigmoid
Ingrese el número de épocas para el entrenamiento: 1000
Ingrese la tasa de aprendizaje: 0.1
Ingrese la matriz de características de entrenamiento (X) en formato de lista de listas: [[0, 0], [0, 1], [1, 0], [1, 1]]
Ingrese la matriz de resultados esperados (y) en formato de lista de listas: [[0], [0], [0], [1]]
```

## Uso

```python
from NeuralNetwork import NeuralNetwork
import numpy as np

def main():
    print("Bienvenido al configurador de Redes Neuronales!")
    input_size = int(input("Ingrese el número de entradas: "))
    output_size = int(input("Ingrese el número de salidas: "))
    hidden_layers = list(map(int, input("Ingrese el número de neuronas en cada capa oculta, separadas por comas: ").split(',')))
    hidden_activation = input("Ingrese la función de activación para las capas ocultas (sigmoid/tanh): ").lower() or 'sigmoid'
    output_activation = input("Ingrese la función de activación para la capa de salida (sigmoid/identity/step): ").lower() or 'sigmoid'
    epochs = int(input("Ingrese el número de épocas para el entrenamiento: 1000"))
    learning_rate = float(input("Ingrese la tasa de aprendizaje: 0.1"))

    X = np.array(eval(input("Ingrese la matriz de características de entrenamiento (X) en formato de lista de listas: ")))
    y = np.array(eval(input("Ingrese la matriz de resultados esperados (y) en formato de lista de listas: ")))

    nn = NeuralNetwork(input_size, output_size, hidden_layers, hidden_activation, output_activation)
    nn.train(X, y, epochs, learning_rate)

    print("Entrenamiento completado. Realizando predicciones con los mismos datos de entrenamiento.")
    predictions = nn.predict(X)

    print("\nEntradas y Predicciones:")
    for i in range(len(X)):
        print(f"Entrada: {X[i]}, Predicción: {predictions[i]}")

if __name__ == "__main__":
    main()
```

Este README proporciona una explicación detallada del funcionamiento de cada parte de la clase `NeuralNetwork`, incluyendo la inicialización, propagación hacia adelante, propagación hacia atrás, entrenamiento y predicción. Además, se ofrecen recomendaciones para las combinaciones de funciones de activación en capas ocultas y de salida, junto con un ejemplo de configuración y uso del código.
```


