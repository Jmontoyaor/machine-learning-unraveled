# machine-learning-unraveled

Repositorio central de notas, demostraciones teóricas, talleres, parciales y actividades prácticas para el curso de **Teoría de Aprendizaje de Máquinas (TAM)** - Universidad Nacional de Colombia.

---

## 🗺️ Grafo de Estudio y Demostraciones Interactivas

- **Mapa Interactivo en Vivo:** [Acceder al Grafo de Demostraciones (GitHub Pages)](https://jmontoyaor.github.io/juan-fernando-montoya-ortiz-machine-learning-unraveled/)
- **Archivo fuente del mapa:** [`docs/index.html`](docs/index.html)
- **Cuaderno de soporte:** [`notebooks/Visualizacion_Demostraciones.ipynb`](notebooks/Visualizacion_Demostraciones.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jmontoyaor/juan-fernando-montoya-ortiz-machine-learning-unraveled/blob/main/notebooks/Visualizacion_Demostraciones.ipynb)

Permite explorar visualmente los grafos de dependencias entre demostraciones, ejercicios resueltos, preguntas de práctica y ejecuciones con código interactivo.

---

## 🔢 Actividad 1: MNIST, red densa vs. red convolucional

- **Abrir en el navegador (sin instalar nada):** [jmontoyaor.github.io/juan-fernando-montoya-ortiz-machine-learning-unraveled/mnist/](https://jmontoyaor.github.io/juan-fernando-montoya-ortiz-machine-learning-unraveled/mnist/) — dibuja, usa la cámara o sube una imagen; muestra la corrección a fondo negro y trazo blanco, y las predicciones de ambas redes.
- **App local con entrenamiento (localhost):** ejecuta [`Actividad/Actividades 1/iniciar_app.bat`](Actividad/Actividades%201/iniciar_app.bat) en Windows o `iniciar_app.sh` en Linux/macOS, y se abrirá [http://localhost:8501](http://localhost:8501). Ese enlace solo funciona mientras el lanzador está corriendo en tu equipo.
- **Documentación:** [`Actividad/Actividades 1/README_WEB.md`](Actividad/Actividades%201/README_WEB.md).

---

## 📂 Estructura del Repositorio

El proyecto se encuentra organizado en las siguientes secciones principales:

```text
machine-learning-unraveled/
├── Talleres/
│   └── Taller 1/
│       ├── Taller1-S3.pdf
│       ├── Taller_Alternativo_TAM_2026_2.pdf
│       └── Taller1_TAM_2026_2.ipynb
│
├── Parcial/
│   └── Parcial 1/
│       ├── Parcial_1_TAM_2026_2_Pearson_RKHS.pdf
│       ├── Conceptos_Por_Que_Funciona_Parcial1_TAM.pdf
│       └── Solucion_Parcial_1_2026_2.ipynb
│
├── Actividad/
│   ├── Actividades 1/
│   │   ├── web_app.py            # app local Streamlit (localhost:8501)
│   │   ├── iniciar_app.bat/.sh   # lanzadores de un paso
│   │   ├── train_mnist.py        # entrenamiento de las redes de referencia
│   │   ├── export_web.py         # exporta los modelos a docs/mnist/
│   │   ├── mnist_app/            # datos, preprocesamiento, modelos, evaluación
│   │   ├── artifacts/            # modelos .keras y resultados .json
│   │   ├── tests/
│   │   ├── train.py, app_cam.py  # versión original con OpenCV (conservada)
│   │   ├── README_WEB.md
│   │   └── README.md
│   └── Actividades 2/
│       └── README.md
│
├── docs/
│   ├── index.html
│   ├── mnist/              # versión web estática de la Actividad 1 (GitHub Pages)
│   └── fig/
└── notebooks/
    └── Visualizacion_Demostraciones.ipynb
```

---

### 1. 📝 [Talleres](Talleres/)
- **[`Talleres/Taller 1/`](Talleres/Taller%201/)**: Guías de taller (`Taller1-S3.pdf`, `Taller_Alternativo_TAM_2026_2.pdf`) y cuaderno de trabajo [`Taller1_TAM_2026_2.ipynb`](Talleres/Taller%201/Taller1_TAM_2026_2.ipynb).

### 2. 📑 [Parcial](Parcial/)
- **[`Parcial/Parcial 1/`](Parcial/Parcial%201/)**: Enunciado de Pearson & RKHS (`Parcial_1_TAM_2026_2_Pearson_RKHS.pdf`), justificación teórica (`Conceptos_Por_Que_Funciona_Parcial1_TAM.pdf`) y cuaderno con solución completa [`Solucion_Parcial_1_2026_2.ipynb`](Parcial/Parcial%201/Solucion_Parcial_1_2026_2.ipynb).

### 3. 🎯 [Actividad](Actividad/)
- **[`Actividad/Actividades 1/`](Actividad/Actividades%201/)**: Clasificador de dígitos MNIST con las dos redes de referencia (densa y convolucional). Incluye una [versión web](https://jmontoyaor.github.io/juan-fernando-montoya-ortiz-machine-learning-unraveled/mnist/), una app local en localhost con entrenamiento y la versión original con OpenCV y osciloscopio.
- **[`Actividad/Actividades 2/`](Actividad/Actividades%202/)**: Espacio reservado para la segunda actividad práctica.

---

## 🚀 Requisitos y Ejecución

Para ejecutar los cuadernos y scripts locales:

```bash
# Crear y activar entorno virtual
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
source .venv/bin/activate

# Instalar dependencias comunes
pip install numpy scipy matplotlib scikit-learn jupyter
```

Para la actividad de visión por computador (OpenCV + TensorFlow / Keras), consultar las instrucciones detalladas en [`Actividad/Actividades 1/README.md`](Actividad/Actividades%201/README.md).
