"""
Script de Verificación de Cámara y Entorno
=========================================
Comprueba:
1. Conexión de cámaras web disponibles en OpenCV.
2. Carga del modelo entrenado (MLP y CNN).
3. Inferencia de prueba con un número aleatorio de MNIST.
"""

import cv2
import numpy as np
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def check_models():
    print("-" * 50)
    print("1. COMPROBACIÓN DE MODELOS:")
    models = ['mnist_model.keras', 'mnist_model_mlp.keras', 'mnist_model_cnn.keras']
    found = 0
    for m in models:
        m_path = os.path.join(BASE_DIR, m)
        if os.path.exists(m_path):
            found += 1
            mod = tf.keras.models.load_model(m_path)
            print(f" [OK] {m} encontrado. Entrada requerida: {mod.input_shape}")
        else:
            print(f" [-] {m} no encontrado.")
    return found > 0

def check_cameras():
    print("-" * 50)
    print("2. COMPROBACIÓN DE CÁMARAS WEB:")
    available = []
    backends = [("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY), ("DSHOW", cv2.CAP_DSHOW)]
    
    for b_name, b_code in backends:
        cap = cv2.VideoCapture(0, b_code)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w, c = frame.shape
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f" [OK] Cámara abierta con backend {b_name} -> Resolución: {w}x{h}")
                available.append((0, b_name))
                cap.release()
                break
            cap.release()
    
    if not available:
        print(" [!] No se detectó ninguna cámara web física activa.")
        print("     (Nota: La app app_cam.py incluye un modo simulación interactivo si no tienes cámara).")
    return available

def test_inference():
    print("-" * 50)
    print("3. PRUEBA DE INFERENCIA EN MODELO:")
    model_path = os.path.join(BASE_DIR, 'mnist_model.keras')
    if not os.path.exists(model_path):
        print(f" [!] No se puede probar: {model_path} no existe.")
        return
    
    model = tf.keras.models.load_model(model_path)
    # Cargar 1 dígito de prueba
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    sample = x_test[0].astype('float32') / 255.0
    true_label = y_test[0]

    if len(model.input_shape) == 2:
        inp = sample.reshape(1, 784)
    else:
        inp = sample.reshape(1, 28, 28, 1)

    preds = model.predict(inp, verbose=0)[0]
    pred_label = int(np.argmax(preds))
    confidence = preds[pred_label] * 100.0

    print(f" Dígito real: {true_label}")
    print(f" Predicción del modelo: {pred_label} con {confidence:.2f}% de confianza.")
    if pred_label == true_label:
        print(" [OK] ¡Inferencia correcta!")
    else:
        print(" [!] Inferencia discrepante.")

if __name__ == '__main__':
    print("==================================================")
    print(" DIAGNÓSTICO DEL SISTEMA (OpenCV + TensorFlow)")
    print("==================================================")
    check_models()
    check_cameras()
    test_inference()
    print("==================================================")
