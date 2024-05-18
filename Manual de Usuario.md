
# Manual de Usuario

En este manual se explican los pasos necesarios para configurar y entrenar una red neuronal para aprender la operación lógica AND. A continuación, se describen los valores de entrada necesarios y su propósito.

## Configuración de la Red Neuronal

### Parámetros de Configuración

1. **input_size**: Define el número de entradas de la red neuronal.
    - **Valor**: `2`
    - **Descripción**: La operación lógica AND toma dos entradas binarias (0 o 1).

2. **output_size**: Define el número de salidas de la red neuronal.
    - **Valor**: `1`
    - **Descripción**: La operación lógica AND produce una única salida binaria (0 o 1).

3. **hidden_layers**: Define la estructura de las capas ocultas, es decir, cuántas capas ocultas tiene la red y cuántas neuronas hay en cada capa.
    - **Valor**: `[2]`
    - **Descripción**: Una capa oculta con 2 neuronas.

4. **hidden_activation**: Define la función de activación utilizada en las capas ocultas.
    - **Valor**: `'sigmoid'`
    - **Descripción**: La función sigmoide es adecuada para problemas de clasificación binaria y ayuda a introducir no linealidad en la red.

5. **output_activation**: Define la función de activación utilizada en la capa de salida.
    - **Valor**: `'step'`
    - **Descripción**: La función escalón (step) se utiliza para obtener una salida binaria final (0 o 1), que es ideal para problemas de clasificación binaria como la operación lógica AND.

6. **epochs**: Define el número de épocas de entrenamiento.
    - **Valor**: `1000`
    - **Descripción**: Una época es un ciclo completo a través del conjunto de datos de entrenamiento. Un número mayor de épocas permite que la red ajuste mejor sus pesos, aunque puede llevar más tiempo de entrenamiento.

7. **learning_rate**: Define la tasa de aprendizaje.
    - **Valor**: `0.2`
    - **Descripción**: La tasa de aprendizaje determina el tamaño del paso que la red da durante el ajuste de los pesos. Una tasa de aprendizaje de 0.2 es un valor razonable para empezar.

### Datos de Entrenamiento

1. **X**: Define la matriz de características de entrenamiento.
    - **Valor**: `np.array([[0, 0], [0, 1], [1, 0], [1, 1]])`
    - **Descripción**: Cada fila representa un par de entradas binarias para la operación lógica AND.

2. **y**: Define la matriz de resultados esperados.
    - **Valor**: `np.array([[0], [0], [0], [1]])`
    - **Descripción**: Cada valor en `y` es el resultado esperado para la operación AND de las entradas correspondientes en `X`.

## Flujo para Entrenar y Utilizar la Red Neuronal

### Paso 1: Configuración Inicial

1. **Definir los Parámetros de Configuración**:
    - Establece el tamaño de entrada (`input_size`), el tamaño de salida (`output_size`), la estructura de las capas ocultas (`hidden_layers`), la función de activación para las capas ocultas (`hidden_activation`), la función de activación para la capa de salida (`output_activation`), el número de épocas (`epochs`) y la tasa de aprendizaje (`learning_rate`).

2. **Preparar los Datos de Entrenamiento**:
    - Define la matriz de características de entrenamiento `X` y la matriz de resultados esperados `y` para la operación lógica AND.

### Paso 2: Creación de la Red Neuronal

1. **Inicializar la Red Neuronal**:
    - Crea una instancia de la clase `NeuralNetwork` pasando los parámetros de configuración definidos anteriormente.

### Paso 3: Entrenamiento de la Red Neuronal

1. **Entrenar la Red Neuronal**:
    - Llama al método `train` de la instancia de `NeuralNetwork`, pasando `X`, `y`, el número de épocas y la tasa de aprendizaje.
    - Durante el entrenamiento, la red neuronal ajusta sus pesos y biases para minimizar el error entre las salidas predichas y las salidas esperadas.

### Paso 4: Realización de Predicciones

1. **Realizar Predicciones con los Datos de Entrenamiento**:
    - Llama al método `predict` de la instancia de `NeuralNetwork` pasando `X`.
    - La red neuronal procesará las entradas a través de las capas y aplicará las funciones de activación para producir una salida.

### Paso 5: Evaluación de Resultados

1. **Evaluar las Predicciones**:
    - Compara las salidas predichas con las salidas esperadas para verificar la precisión de la red neuronal.
    - Imprime las entradas y sus respectivas predicciones para ver cómo ha aprendido la red.

### Descripción del Flujo Completo

1. **Configuración Inicial**:
    - Define los parámetros de la red neuronal y prepara los datos de entrenamiento.
2. **Creación de la Red Neuronal**:
    - Inicializa la red neuronal con los parámetros especificados.
3. **Entrenamiento de la Red Neuronal**:
    - Entrena la red neuronal utilizando los datos de entrenamiento durante un número específico de épocas y con una tasa de aprendizaje definida.
4. **Realización de Predicciones**:
    - Utiliza la red neuronal entrenada para realizar predicciones con los datos de entrada.
5. **Evaluación de Resultados**:
    - Compara las predicciones con los resultados esperados y verifica la precisión de la red neuronal.

## Resultado Esperado

Después de seguir estos pasos, deberías obtener los siguientes resultados para la operación lógica AND:

```
Configuración de la Red Neuronal para la Operación Lógica AND
Epoch 0, Loss: 0.2499
Predicciones (entrenamiento):
Entrada: [0 0], Predicción: [0]
Entrada: [0 1], Predicción: [0]
Entrada: [1 0], Predicción: [0]
Entrada: [1 1], Predicción: [1]

...

Entrenamiento completado. Realizando predicciones con los mismos datos de entrenamiento.

Entradas y Predicciones:
Entrada: [0 0], Predicción: [0]
Entrada: [0 1], Predicción: [0]
Entrada: [1 0], Predicción: [0]
Entrada: [1 1], Predicción: [1]
```

Estos resultados muestran que la red neuronal ha aprendido correctamente la operación lógica AND, produciendo las salidas esperadas para cada par de entradas binarias.