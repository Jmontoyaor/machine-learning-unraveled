# machine-learning-unraveled

Repositorio central de notas, demostraciones teóricas, talleres, parciales y actividades prácticas para el curso de **Teoría de Aprendizaje de Máquinas (TAM)** - Universidad Nacional de Colombia.

---

## 🗺️ Grafo de Estudio y Demostraciones Interactivas

- **Mapa Interactivo en Vivo:** [Acceder al Grafo de Demostraciones (GitHub Pages)](https://jmontoyaor.github.io/machine-learning-unraveled/)
- **Archivo fuente del mapa:** [`docs/index.html`](docs/index.html)
- **Cuaderno de soporte:** [`notebooks/Visualizacion_Demostraciones.ipynb`](notebooks/Visualizacion_Demostraciones.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jmontoyaor/machine-learning-unraveled/blob/main/notebooks/Visualizacion_Demostraciones.ipynb)

Permite explorar visualmente los grafos de dependencias entre demostraciones, ejercicios resueltos, preguntas de práctica y ejecuciones con código interactivo.

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
│   │   ├── train.py
│   │   ├── app_cam.py
│   │   ├── test_cam.py
│   │   ├── mnist_model_cnn.keras
│   │   ├── mnist_model_mlp.keras
│   │   └── README.md
│   └── Actividades 2/
│       └── README.md
│
├── docs/
│   ├── index.html
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
- **[`Actividad/Actividades 1/`](Actividad/Actividades%201/)**: Clasificador de dígitos manuscritos MNIST en tiempo real utilizando OpenCV y modelos de Deep Learning (MLP y CNN), con osciloscopio interactivo de evolución de precisión durante el entrenamiento.
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
