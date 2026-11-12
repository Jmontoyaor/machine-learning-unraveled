# Sistema de Clasificación de Dígitos en Tiempo Real con OpenCV y Osciloscopio de Accuracy

> **Versión nueva de la actividad:**
> - En el navegador, sin instalar nada: <https://jmontoyaor.github.io/machine-learning-unraveled/mnist/>
> - En localhost, con entrenamiento: ejecuta `iniciar_app.bat` (Windows) o `iniciar_app.sh` (Linux/macOS) y abre <http://localhost:8501>.
> - Guía completa: [README_WEB.md](README_WEB.md).
>
> Esta guía histórica de la versión con cámara OpenCV se conserva; sus métricas
> antiguas no describen los resultados nuevos.

Este proyecto implementa la solución completa solicitada a partir de los apuntes manuscritos y el cuaderno Jupyter `scratchpad.ipynb`.

---

## 1. Concepto y Requerimientos Cumplidos

A partir de las notas manuscritas:
1. **"La métrica ayuda a revisar iteración por iteración -> ACCURACY -> un osciloscopio":**
   - Se diseñó un panel interactivo estilo **Osciloscopio CRT** (pantalla de fósforo verde con retícula milimétrica, divisiones de tiempo y escala porcentual de 0% a 100%).
   - Monitorea y grafica la señal de **Accuracy / Confianza** cuadro a cuadro (*frame-by-frame / iteración tras iteración*), mostrando el haz de electrones, el histórico de los últimos 120 cuadros, valores estadísticos (Actual, Pico/Peak y Promedio/Avg), y un analizador de espectro de probabilidad para los dígitos del 0 al 9.
   - Adicionalmente, el script de entrenamiento genera automáticamente un gráfico de osciloscopio (`training_osciloscopio_*.png`) con la evolución de la precisión y pérdida por época.

2. **"OpenCV -> La idea es que le pongamos el número al frente de la cámara -> Imagen -> Interfaz -> Diga qué imagen":**
   - Se implementa una aplicación en **OpenCV** (`app_cam.py`) con captura en tiempo real desde la cámara web.
   - Dispone de un recuadro de interés central (ROI) donde se coloca el número escrito en papel.
   - Aplica un pipeline de preprocesamiento avanzado (escala de grises, filtrado Gaussiano, binarización Otsu, extracción del contorno del trazo, escalado a 20x20 con preservación de relación de aspecto y centrado en un lienzo de 28x28 idéntico a MNIST).
   - El modelo de Inteligencia Artificial realiza la inferencia en milisegundos y la interfaz indica en pantalla gigante el dígito detectado junto con su nivel de certeza.

---

## 2. Estructura de Archivos del Proyecto

```text
C:\Users\juanf\Downloads\TAREA\
│
├── .venv/                         # Entorno virtual con Python 3.12, TensorFlow 2.21 y OpenCV 5.0
├── app_cam.py                     # Aplicación principal: Cámara en vivo + OpenCV + Osciloscopio CRT
├── train.py                       # Script de entrenamiento (soporta arquitectura MLP original y CNN)
├── test_cam.py                    # Script de autodiagnóstico de cámara, modelos e inferencia
├── mnist_model.keras              # Modelo activo por defecto (CNN con 99.03% de precisión en test)
├── mnist_model_mlp.keras          # Modelo con arquitectura Dense original de scratchpad.ipynb (97.52%)
├── mnist_model_cnn.keras          # Modelo Convolucional robusto frente a rotaciones y cámara
├── training_osciloscopio_mlp.png  # Gráfico tipo osciloscopio del entrenamiento MLP
├── training_osciloscopio_cnn.png  # Gráfico tipo osciloscopio del entrenamiento CNN
├── scratchpad.ipynb               # Cuaderno Jupyter original con el modelo base
├── WhatsApp Image ... .jpeg       # Imagen con los requerimientos manuscritos originales
└── README.md                      # Esta guía paso a paso
```

---

## 3. Guía de Ejecución Paso a Paso

### Paso 1: Abrir la terminal y activar el entorno virtual
Abre PowerShell o CMD y dirígete a la carpeta del proyecto:
```powershell
cd C:\Users\juanf\Downloads\TAREA
```

Activa el entorno virtual:
- En **PowerShell**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- O en **CMD**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

*(Si PowerShell muestra una advertencia de políticas de ejecución, puedes ejecutar directamente usando la ruta completa del ejecutable: `.\.venv\Scripts\python.exe`)*.

---

### Paso 2: Verificar el sistema y la cámara
Ejecuta el script de diagnóstico para comprobar que la cámara y los modelos cargan correctamente:
```powershell
.\.venv\Scripts\python.exe test_cam.py
```

Salida esperada:
```text
==================================================
 DIAGNÓSTICO DEL SISTEMA (OpenCV + TensorFlow)
==================================================
1. COMPROBACIÓN DE MODELOS:
 [OK] mnist_model.keras encontrado.
 [OK] mnist_model_mlp.keras encontrado.
 [OK] mnist_model_cnn.keras encontrado.
--------------------------------------------------
2. COMPROBACIÓN DE CÁMARAS WEB:
 [OK] Cámara abierta con backend MSMF -> Resolución: 640x480
--------------------------------------------------
3. PRUEBA DE INFERENCIA EN MODELO:
 Dígito real: 7
 Predicción del modelo: 7 con 100.00% de confianza.
 [OK] ¡Inferencia correcta!
==================================================
```

---

### Paso 3: Ejecutar la aplicación de cámara con osciloscopio
Para iniciar la interfaz interactiva con detección en tiempo real:
```powershell
.\.venv\Scripts\python.exe app_cam.py
```

Si deseas probar específicamente con el modelo de red neuronal densa (MLP) original:
```powershell
.\.venv\Scripts\python.exe app_cam.py mnist_model_mlp.keras
```

---

### Paso 4: (Opcional) Reentrenar los modelos
Los modelos ya vienen previamente entrenados y listos para usar en la carpeta, pero si deseas volver a entrenar desde cero:

- **Entrenar la red MLP original (de `scratchpad.ipynb`)**:
  ```powershell
  .\.venv\Scripts\python.exe train.py --type mlp --epochs 15
  ```
- **Entrenar la red Convolucional (CNN)**:
  ```powershell
  .\.venv\Scripts\python.exe train.py --type cnn --epochs 5
  ```

Al finalizar cada entrenamiento, se actualizarán los gráficos `training_osciloscopio_*.png`.

---

## 4. Controles y Teclas en la Aplicación de Cámara

| Tecla | Acción | Descripción |
| :---: | :--- | :--- |
| **`T`** | **Alternar Umbral (Threshold)** | Cambia entre *Tinta negra sobre papel blanco* (por defecto) y *Trazo blanco sobre fondo negro*. |
| **`C`** | **Limpiar Osciloscopio** | Reinicia la memoria de traza del osciloscopio a 0%. |
| **`S`** | **Guardar Captura** | Toma una foto del frame actual junto con el osciloscopio y la guarda como PNG. |
| **`Q`** o **`ESC`** | **Salir** | Cierra la ventana y libera la cámara de forma segura. |

---

## 5. Elementos de la Interfaz Visual

1. **Panel Izquierdo (Visor de Cámara)**:
   - **Cuadro ROI central**: Muestra un recuadro verde donde debes colocar el número escrito.
   - **Miniatura MNIST (28x28)**: En la esquina superior izquierda se muestra la imagen exacta tal como la ve y procesa la red neuronal (centrada y binarizada).
   - **Cartel de Detección**: Texto destacado que dice `"NUMERO: X"` y `"ACCURACY: YY.Y%"`.
   - **Contador de FPS**: Indica la fluidez en tiempo real.

2. **Panel Derecho (El "Osciloscopio")**:
   - **Retícula de voltaje/precisión**: Escala calibrada en 0%, 25%, 50%, 75% y 100%.
   - **Traza verde fosforescente**: Curva continua que registra cómo varía la confianza y estabilidad de la detección a lo largo del tiempo.
   - **Haz de barrido**: Punto brillante que marca la lectura instantánea.
   - **Panel digital**: Muestra los valores `ACTUAL`, `PEAK` (máximo histórico) y `AVG` (promedio).
   - **Distribución de Probabilidades (0 al 9)**: 10 barras horizontales tipo analizador de espectro que muestran la confianza softmax para cada dígito posible.

---

## 6. Consejos para Obtener 100% de Precisión frente a la Cámara

1. **Grosor del trazo**: Escribe el número con un marcador negro o bolígrafo grueso para que el trazo sea visible y definido.
2. **Iluminación**: Procura que la hoja de papel no tenga sombras intensas sobre el recuadro.
3. **Centrado**: Mantén el número dentro del cuadro delimitador verde.
4. **Modo Simulación**: Si en algún momento ejecutas la aplicación en una máquina sin cámara física conectada, `app_cam.py` activará automáticamente el modo de simulación interactivo, permitiéndote cambiar de dígitos del dataset de prueba con la barra espaciadora `[ESPACIO]`.
