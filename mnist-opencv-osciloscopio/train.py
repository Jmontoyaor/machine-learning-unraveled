"""
Script de Entrenamiento del Modelo MNIST
=======================================
Entrena la red neuronal descrita en el cuaderno scratchpad.ipynb
y guarda el modelo entrenado junto con la visualización tipo
'osciloscopio' del avance de accuracy y pérdida por iteración.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_model(model_type='mlp'):
    """
    Construye el modelo.
    - 'mlp': La red neuronal densa definida originalmente en scratchpad.ipynb:
             Input(784) -> Dense(64, relu) -> Dense(64, relu) -> Dense(10, softmax)
    - 'cnn': Red convolucional que ofrece mayor robustez frente a la cámara.
    """
    if model_type == 'mlp':
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(784,), name="Input_784"),
            tf.keras.layers.Dense(64, activation='relu', name="Capa_Oculta_1"),
            tf.keras.layers.Dense(64, activation='relu', name="Capa_Oculta_2"),
            tf.keras.layers.Dense(10, activation='softmax', name="Salida_10")
        ])
    else:
        # Arquitectura CNN moderna para mejor tolerancia a rotación/escala de cámara
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1), name="Input_28x28"),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    return model

def plot_oscilloscope_metrics(history, output_path='training_osciloscopio.png'):
    """
    Genera un gráfico con estilo de osciloscopio que muestra
    la evolución del Accuracy y Loss iteración por iteración / época por época.
    """
    epochs = range(1, len(history.history['accuracy']) + 1)
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history.get('val_accuracy')
    val_loss = history.history.get('val_loss')

    # Estilo oscuro tipo pantalla de osciloscopio fósforo verde/cian
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='#0b110e')
    
    for ax in (ax1, ax2):
        ax.set_facecolor('#0f1b14')
        ax.grid(True, which='both', color='#1d382b', linestyle='--', linewidth=0.8)
        ax.tick_params(colors='#4af626')
        for spine in ax.spines.values():
            spine.set_color('#28593a')

    # Traza 1: Accuracy (Canal 1 - Osciloscopio)
    ax1.plot(epochs, acc, color='#00ff88', linewidth=2, marker='o', markersize=4, label='CH1: Train Accuracy')
    if val_acc:
        ax1.plot(epochs, val_acc, color='#00e5ff', linewidth=2, linestyle=':', label='CH2: Val Accuracy')
    ax1.set_title('OSCILOSCOPIO DE ENTRENAMIENTO - CANAL A: ACCURACY [%]', color='#4af626', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', color='#4af626')
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc='lower right', facecolor='#0b110e', edgecolor='#28593a', labelcolor='#4af626')

    # Traza 2: Loss (Canal 2 - Osciloscopio)
    ax2.plot(epochs, loss, color='#ff0055', linewidth=2, marker='s', markersize=4, label='CH1: Train Loss')
    if val_loss:
        ax2.plot(epochs, val_loss, color='#ffb703', linewidth=2, linestyle=':', label='CH2: Val Loss')
    ax2.set_title('OSCILOSCOPIO DE ENTRENAMIENTO - CANAL B: LOSS (PÉRDIDA)', color='#ff6b6b', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Época / Iteración', color='#4af626')
    ax2.set_ylabel('Loss', color='#ff6b6b')
    ax2.legend(loc='upper right', facecolor='#0b110e', edgecolor='#28593a', labelcolor='#ff6b6b')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Gráfico osciloscopio de entrenamiento guardado en: {output_path}")

def train(model_type='mlp', epochs=20, batch_size=32):
    print("=" * 60)
    print(f" Iniciando entrenamiento de MNIST (Modelo: {model_type.upper()})")
    print("=" * 60)

    # 1. Cargar datos MNIST
    print("[*] Cargando dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalizar valores de píxeles a [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    if model_type == 'mlp':
        # Aplanar para red densa (784 features)
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)
    else:
        # Mantener forma de imagen (28, 28, 1) para CNN
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    # 2. Construir modelo
    model = build_model(model_type=model_type)
    model.summary()

    # 3. Compilar
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Entrenar
    print(f"[*] Entrenando por {epochs} épocas...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # 5. Evaluar con test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("=" * 60)
    print(f" [RESULTADO FINAL]")
    print(f" Accuracy en test: {test_acc * 100:.2f}%")
    print(f" Loss en test:     {test_loss:.4f}")
    print("=" * 60)

    # 6. Guardar modelo
    output_filename = os.path.join(BASE_DIR, f"mnist_model_{model_type}.keras")
    model.save(output_filename)
    print(f"[+] Modelo guardado en: {output_filename}")
    
    # También guardamos como mnist_model.keras para la app por defecto
    default_filename = os.path.join(BASE_DIR, "mnist_model.keras")
    model.save(default_filename)
    print(f"[+] Modelo por defecto actualizado: {default_filename}")

    # 7. Generar gráfico tipo osciloscopio
    plot_path = os.path.join(BASE_DIR, f"training_osciloscopio_{model_type}.png")
    plot_oscilloscope_metrics(history, plot_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Entrenar clasificador de dígitos MNIST")
    parser.add_argument('--type', choices=['mlp', 'cnn'], default='mlp', 
                        help="Arquitectura a entrenar: 'mlp' (original) o 'cnn' (alta precisión cámara)")
    parser.add_argument('--epochs', type=int, default=15, help="Número de épocas")
    parser.add_argument('--batch-size', type=int, default=32, help="Tamaño de lote")
    args = parser.parse_args()

    train(model_type=args.type, epochs=args.epochs, batch_size=args.batch_size)
