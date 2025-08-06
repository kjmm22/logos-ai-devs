# logos-ai-devs

Este repositorio contiene un cuaderno de Google Colab orientado a la
formación de desarrolladores en técnicas de visión artificial. El archivo
`KM_NeuralNetwork.ipynb` implementa, entrena y evalúa una red neuronal
convolucional ligera utilizando PyTorch.

### ¿Qué hace el notebook?

- **Carga de datos**: monta Google Drive, lee los nombres de archivo y sus
  etiquetas desde un CSV y aplica transformaciones de preprocesamiento para
  preparar las imágenes en escala de grises.
- **Modelo**: define `ImprovedNet`, una CNN con aproximadamente 8 300
  parámetros que combina capas convolucionales, `BatchNorm`, `MaxPool` y
  una pequeña red totalmente conectada para clasificar en dos clases.
- **Entrenamiento**: entrena el modelo con `Adam` y ajusta la tasa de
  aprendizaje cuando no se observan mejoras, guardando los pesos
  preentrenados al lograr mayor precisión.
- **Evaluación e inferencia**: carga los pesos guardados, calcula la
  precisión y la desviación estándar sobre un nuevo conjunto de imágenes y
  muestra ejemplos de predicciones junto con sus etiquetas reales.

El notebook alcanza una precisión cercana al 98 % en validación y sirve como
base para experimentar con arquitecturas pequeñas de visión artificial en
Colab.

### Nuevo modelo sin redimensionamiento

El cuaderno también incluye `KM_NoResizeNet`, una variante con aproximadamente
8 300 parámetros que elimina cualquier operación de `Resize` o `Crop` en el

preprocesamiento. Gracias a `nn.AdaptiveAvgPool2d`, el modelo puede recibir
imágenes en su tamaño original. Esta versión reutiliza las mismas rutas de
datos que el modelo base y reporta la *accuracy* de validación en cada época
de entrenamiento.
