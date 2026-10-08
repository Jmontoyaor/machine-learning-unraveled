# machine-learning-unraveled

Theory and notes for the Machine Learning Theory course (Teoría de Aprendizaje de Máquina, UNAL).

## Contents

| Path | What it is |
|---|---|
| [`docs/index.html`](https://jmontoyaor.github.io/machine-learning-unraveled/) | Interactive map of the 33 proofs and 9 solved exercises. Select a node to see the proof, practice questions and code. The **Cuaderno (Python)** tab shows the matching notebook cell, its output and its figure, with an **Abrir en Colab** button. |
| [`notebooks/Visualizacion_Demostraciones.ipynb`](notebooks/Visualizacion_Demostraciones.ipynb) | Notebook with one cell per proof or exercise, executed with MNIST. [Open in Colab](https://colab.research.google.com/github/Jmontoyaor/machine-learning-unraveled/blob/main/notebooks/Visualizacion_Demostraciones.ipynb). |
| `docs/fig/` | Figures exported from the notebook, used by the map. |

## Running the notebook

In Colab, run the setup cell first; after that, sections can run in any order. Outside Colab it needs `numpy`, `scipy`, `matplotlib` and `scikit-learn`; without TensorFlow it falls back to the 8×8 scikit-learn digits instead of MNIST, so the MNIST-based results change.
