"""
Aplicación OpenCV con Detección de Dígitos y Osciloscopio de Accuracy
====================================================================
Implementa los requerimientos indicados en la imagen del cuaderno:
1. 'la metrica Ayuda a revisar iteración por iteración -> ACCURACY -> un osciloscopio'
2. 'opencv -> la idea es que le pongamos el numero al frente de camara -> Imagen -> interfaz -> diga que imagen'

Permite capturar en tiempo real con la webcam, preprocesar el número
dibujado, predecir el dígito con el modelo entrenado y visualizar
un osciloscopio en vivo con la evolución de la precisión / confianza.
"""

import os
import sys
import time
from collections import deque
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class DigitClassifierApp:
    def __init__(self, model_path=None, cam_index=0):
        if model_path is None:
            model_path = os.path.join(BASE_DIR, 'mnist_model.keras')
        elif not os.path.isabs(model_path):
            model_path = os.path.join(BASE_DIR, model_path)

        print(f"[*] Cargando modelo desde: {model_path} ...")
        if not os.path.exists(model_path):
            # Buscar alternativas
            mlp_alt = os.path.join(BASE_DIR, 'mnist_model_mlp.keras')
            cnn_alt = os.path.join(BASE_DIR, 'mnist_model_cnn.keras')
            if os.path.exists(mlp_alt):
                model_path = mlp_alt
            elif os.path.exists(cnn_alt):
                model_path = cnn_alt
            else:
                raise FileNotFoundError(
                    f"No se encontró el archivo del modelo '{model_path}'. "
                    "Por favor ejecuta primero: python train.py"
                )

        self.model = tf.keras.models.load_model(model_path)
        self.input_shape = self.model.input_shape
        print(f"[+] Modelo cargado exitosamente. Input shape: {self.input_shape}")

        self.cam_index = cam_index
        self.invert_thresh = True  # True para papel blanco y tinta oscura (lo convierte a blanco sobre negro como MNIST)
        
        # Buffer de historial para el osciloscopio (últimas 120 iteraciones / frames)
        self.history_len = 120
        self.accuracy_history = deque([0.0] * self.history_len, maxlen=self.history_len)
        self.digit_history = deque([None] * self.history_len, maxlen=self.history_len)

        # Estado actual
        self.current_digit = None
        self.current_confidence = 0.0
        self.current_probabilities = np.zeros(10)
        self.fps = 0.0
        self.prev_time = time.time()
        self.roi_has_content = False

    def preprocess_roi(self, roi_bgr):
        """
        Preprocesa el recorte de la cámara para que coincida exactamente con el formato MNIST:
        - Escala de grises
        - Desenfoque para reducir ruido
        - Umbralizado adaptativo / Otsu (número blanco sobre fondo negro)
        - Detección de contornos para aislar el trazo del dígito
        - Centrado y reescalado a 20x20 dentro de un lienzo de 28x28 normalizado a [0, 1]
        """
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Umbralizado Otsu
        if self.invert_thresh:
            # Tinta negra sobre papel blanco -> invertimos a número blanco sobre fondo negro
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Encontrar contornos del dígito
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digit_28 = np.zeros((28, 28), dtype=np.float32)
        has_content = False

        if contours:
            # Obtener el contorno más grande que tenga un área mínima relevante
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 100:  # Umbral mínimo de píxeles para ignorar ruido pequeño
                has_content = True
                x, y, w, h = cv2.boundingRect(c)
                digit_crop = thresh[y:y+h, x:x+w]

                # Escalar para que quepa en un recuadro de 20x20 conservando relación de aspecto
                if w > h:
                    new_w = 20
                    new_h = max(int(h * (20.0 / w)), 1)
                else:
                    new_h = 20
                    new_w = max(int(w * (20.0 / h)), 1)

                resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Centrar en un lienzo negro de 28x28
                pad_top = (28 - new_h) // 2
                pad_bottom = 28 - new_h - pad_top
                pad_left = (28 - new_w) // 2
                pad_right = 28 - new_w - pad_left

                padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, 
                                            cv2.BORDER_CONSTANT, value=0)
                
                # Normalizar a [0, 1]
                digit_28 = padded.astype('float32') / 255.0

        self.roi_has_content = has_content
        return thresh, digit_28

    def predict(self, digit_28):
        """Ejecuta inferencia con el modelo."""
        if not self.roi_has_content:
            self.current_digit = None
            self.current_confidence = 0.0
            self.current_probabilities = np.zeros(10)
            self.accuracy_history.append(0.0)
            self.digit_history.append(None)
            return

        # Adaptar forma según la arquitectura del modelo (MLP o CNN)
        if len(self.input_shape) == 2:  # MLP espera (batch, 784)
            inp = digit_28.reshape(1, 784)
        else:  # CNN espera (batch, 28, 28, 1)
            inp = digit_28.reshape(1, 28, 28, 1)

        preds = self.model.predict(inp, verbose=0)[0]
        self.current_probabilities = preds
        self.current_digit = int(np.argmax(preds))
        self.current_confidence = float(preds[self.current_digit]) * 100.0

        # Guardar en buffer del osciloscopio
        self.accuracy_history.append(self.current_confidence)
        self.digit_history.append(self.current_digit)

    def draw_oscilloscope(self, width=480, height=520):
        """
        Dibuja el panel del osciloscopio:
        - Pantalla oscura estilo fósforo verde CRT
        - Retícula / Grilla milimétrica
        - Señal en tiempo real de la métrica de Accuracy (%) iteración tras iteración
        - Barras de probabilidad para cada dígito (0 al 9)
        - Lectura numérica digital
        """
        # Fondo osciloscopio CRT
        scope = np.zeros((height, width, 3), dtype=np.uint8)
        scope[:] = (15, 22, 16)  # Verde muy oscuro / negro

        # Área de traza del osciloscopio
        plot_x = 30
        plot_y = 60
        plot_w = width - 60
        plot_h = 220

        # Fondo del visor de onda
        cv2.rectangle(scope, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (8, 30, 15), -1)

        # Grilla del osciloscopio (Líneas horizontales: 0%, 25%, 50%, 75%, 100%)
        for i in range(5):
            y_val = plot_y + int(plot_h * (1.0 - i * 0.25))
            cv2.line(scope, (plot_x, y_val), (plot_x + plot_w, y_val), (20, 60, 30), 1)
            pct_label = f"{i * 25}%"
            cv2.putText(scope, pct_label, (5, y_val + 4), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 160, 80), 1)

        # Grilla vertical (divisiones de tiempo)
        divs = 8
        for j in range(divs + 1):
            x_val = plot_x + int(plot_w * (j / divs))
            cv2.line(scope, (x_val, plot_y), (x_val, plot_y + plot_h), (20, 60, 30), 1)

        # Eje central punteado
        cv2.line(scope, (plot_x, plot_y + plot_h // 2), (plot_x + plot_w, plot_y + plot_h // 2), (30, 90, 45), 1)

        # Trazar señal de Accuracy sobre la pantalla
        points = []
        n_pts = len(self.accuracy_history)
        for idx, acc in enumerate(self.accuracy_history):
            px = plot_x + int(plot_w * (idx / max(n_pts - 1, 1)))
            # Mapear 0% - 100% a la altura de la grilla
            clamped_acc = max(0.0, min(100.0, acc))
            py = plot_y + int(plot_h * (1.0 - clamped_acc / 100.0))
            points.append((px, py))

        # Dibujar resplandor y línea principal (efecto fósforo verde/cian)
        if len(points) >= 2:
            pts_np = np.array(points, np.int32).reshape((-1, 1, 2))
            # Resplandor suave
            cv2.polylines(scope, [pts_np], isClosed=False, color=(0, 100, 40), thickness=3, lineType=cv2.LINE_AA)
            # Traza brillante
            cv2.polylines(scope, [pts_np], isClosed=False, color=(40, 255, 120), thickness=2, lineType=cv2.LINE_AA)

            # Punto de barrido actual (haz de electrones del osciloscopio)
            last_pt = points[-1]
            cv2.circle(scope, last_pt, 5, (200, 255, 200), -1, lineType=cv2.LINE_AA)
            cv2.circle(scope, last_pt, 8, (0, 255, 100), 1, lineType=cv2.LINE_AA)

        # Encabezado del osciloscopio
        cv2.putText(scope, "OSCILOSCOPIO // METRICA ACCURACY", (plot_x, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 120), 2)
        cv2.putText(scope, "CH1: CONFIDENCIA TIEMPO REAL [%]", (plot_x, 48), 
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 200, 100), 1)

        # Lectura digital actual debajo de la onda
        stats_y = plot_y + plot_h + 24
        acc_text = f"ACTUAL: {self.current_confidence:5.1f} %"
        max_acc = max(self.accuracy_history) if self.accuracy_history else 0.0
        avg_acc = (sum(self.accuracy_history) / len(self.accuracy_history)) if self.accuracy_history else 0.0
        peak_text = f"PEAK: {max_acc:5.1f} %"
        avg_text  = f"AVG:  {avg_acc:5.1f} %"

        cv2.putText(scope, acc_text, (plot_x, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)
        cv2.putText(scope, peak_text, (plot_x + 150, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 255), 1)
        cv2.putText(scope, avg_text,  (plot_x + 290, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 100), 1)

        # Separador
        cv2.line(scope, (plot_x, stats_y + 12), (plot_x + plot_w, stats_y + 12), (30, 60, 40), 1)

        # Gráfico de barras de probabilidades por cada dígito (0 - 9)
        bars_title_y = stats_y + 32
        cv2.putText(scope, "DISTRIBUCION DE PROBABILIDAD (0 - 9):", (plot_x, bars_title_y), 
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (150, 220, 150), 1)

        bar_start_y = bars_title_y + 16
        bar_height = 12
        bar_spacing = 18
        max_bar_w = plot_w - 70

        for d in range(10):
            by = bar_start_y + d * bar_spacing
            prob = self.current_probabilities[d]
            bw = int(max_bar_w * prob)

            # Etiqueta dígito
            is_winner = (self.current_digit == d and self.roi_has_content)
            label_col = (0, 255, 100) if is_winner else (130, 150, 130)
            cv2.putText(scope, f"[{d}]:", (plot_x, by + 10), cv2.FONT_HERSHEY_PLAIN, 0.85, label_col, 1)

            # Fondo de barra vacía
            cv2.rectangle(scope, (plot_x + 35, by), (plot_x + 35 + max_bar_w, by + bar_height), (20, 40, 25), -1)

            # Barra rellena
            if bw > 0:
                bar_col = (40, 255, 120) if is_winner else (0, 160, 200)
                cv2.rectangle(scope, (plot_x + 35, by), (plot_x + 35 + bw, by + bar_height), bar_col, -1)

            # Porcentaje
            cv2.putText(scope, f"{prob * 100:4.1f}%", (plot_x + 40 + max_bar_w, by + 10), 
                        cv2.FONT_HERSHEY_PLAIN, 0.75, label_col, 1)

        # Borde exterior del osciloscopio
        cv2.rectangle(scope, (0, 0), (width - 1, height - 1), (50, 120, 70), 2)
        return scope

    def open_camera(self):
        """Intenta abrir la cámara con los backends más estables en Windows."""
        backends = [cv2.CAP_MSMF, cv2.CAP_ANY, cv2.CAP_DSHOW]
        for backend in backends:
            cap = cv2.VideoCapture(self.cam_index, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap
                cap.release()
        return None

    def run(self):
        """Bucle principal de ejecución."""
        print("=" * 60)
        print(" Iniciando Sistema OpenCV con Osciloscopio")
        print(" Presiona 'q' o ESC para salir.")
        print(" Presiona 't' para invertir el umbral (fondo blanco / fondo negro).")
        print(" Presiona 'c' para limpiar la señal del osciloscopio.")
        print(" Presiona 's' para guardar una captura con métricas.")
        print("=" * 60)

        cap = self.open_camera()

        if cap is None:
            print(f"[!] ADVERTENCIA: No se pudo abrir la cámara en el índice {self.cam_index}.")
            print("[*] Iniciando en Modo Simulación / Prueba Sintética...")
            self.run_simulation_mode()
            return

        # Configuración de resolución deseada
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        roi_size = 240
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Error leyendo frame de la cámara.")
                break

            # Espejo horizontal para mayor comodidad de interacción
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Calcular cuadro central ROI
            rx1 = max((w - roi_size) // 2, 0)
            ry1 = max((h - roi_size) // 2, 0)
            rx2 = rx1 + roi_size
            ry2 = ry1 + roi_size

            roi = frame[ry1:ry2, rx1:rx2]

            # 1. Preprocesar ROI
            thresh_roi, digit_28 = self.preprocess_roi(roi)

            # 2. Inferencia y actualización de métrica
            self.predict(digit_28)

            # 3. Dibujar cuadro ROI en el frame de la cámara
            box_col = (0, 255, 0) if self.roi_has_content else (0, 165, 255)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), box_col, 2)
            cv2.putText(frame, "COLOCA EL NUMERO AQUI", (rx1, ry1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 1)

            # 4. Mostrar miniatura del dígito procesado (28x28 escalado a 84x84)
            thumb = cv2.resize((digit_28 * 255).astype(np.uint8), (84, 84), interpolation=cv2.INTER_NEAREST)
            thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(thumb_bgr, (0, 0), (83, 83), (0, 255, 0), 1)
            frame[10:94, 10:94] = thumb_bgr
            cv2.putText(frame, "MNIST (28x28)", (10, 106), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)

            # 5. Cartel con el resultado de la predicción en el frame principal
            if self.roi_has_content and self.current_digit is not None:
                pred_label = f"NUMERO: {self.current_digit}"
                conf_label = f"ACCURACY: {self.current_confidence:.1f}%"
                cv2.putText(frame, pred_label, (rx1, ry2 + 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 50), 3)
                cv2.putText(frame, conf_label, (rx1, ry2 + 65), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            else:
                cv2.putText(frame, "ESPERANDO NUMERO...", (rx1, ry2 + 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)

            # FPS
            now = time.time()
            self.fps = 1.0 / max(now - self.prev_time, 0.001)
            self.prev_time = now
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 110, 25), 
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 200), 1)

            # Instrucciones en pantalla
            mode_text = "Tinta negra en papel blanco" if self.invert_thresh else "Trazo blanco en fondo negro"
            cv2.putText(frame, f"[T] Modo: {mode_text}", (10, h - 35), 
                        cv2.FONT_HERSHEY_PLAIN, 0.85, (220, 220, 220), 1)
            cv2.putText(frame, "[Q] Salir | [C] Limpiar osciloscopio | [S] Guardar", (10, h - 15), 
                        cv2.FONT_HERSHEY_PLAIN, 0.85, (180, 180, 180), 1)

            # 6. Renderizar panel de osciloscopio a la derecha
            scope_panel = self.draw_oscilloscope(width=440, height=h)

            # 7. Unir cámara + osciloscopio lado a lado
            combined = np.hstack((frame, scope_panel))

            cv2.imshow("Detector de Numeros con Osciloscopio (OpenCV)", combined)

            # Manejo de teclas
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:  # 'q' o ESC
                break
            elif key == ord('t'):
                self.invert_thresh = not self.invert_thresh
                print(f"[*] Inversión de umbral: {self.invert_thresh}")
            elif key == ord('c'):
                self.accuracy_history = deque([0.0] * self.history_len, maxlen=self.history_len)
                print("[*] Buffer del osciloscopio reiniciado.")
            elif key == ord('s'):
                ts = int(time.time())
                save_fn = f"captura_numero_{ts}.png"
                cv2.imwrite(save_fn, combined)
                print(f"[+] Captura guardada en: {save_fn}")

        cap.release()
        cv2.destroyAllWindows()

    def run_simulation_mode(self):
        """
        Modo de respaldo cuando no hay cámara web física disponible.
        Carga dígitos de prueba del dataset MNIST para demostrar la interfaz,
        el procesamiento y el osciloscopio en tiempo real.
        """
        print("[*] Cargando dígitos de prueba de MNIST para simulación interactiva...")
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        sim_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        current_idx = 0
        total_samples = len(x_test)

        while True:
            sim_frame[:] = (40, 40, 40)
            
            # Dibujar recuadro simulado
            rx1, ry1, rx2, ry2 = 200, 120, 440, 360
            cv2.rectangle(sim_frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.putText(sim_frame, "MODO SIMULACION (SIN CAMARA FISICA)", (120, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

            # Obtener dígito de prueba
            raw_digit = x_test[current_idx]
            ground_truth = y_test[current_idx]

            # Escalar a tamaño del recuadro
            disp_digit = cv2.resize(raw_digit, (200, 200), interpolation=cv2.INTER_NEAREST)
            disp_bgr = cv2.cvtColor(disp_digit, cv2.COLOR_GRAY2BGR)
            sim_frame[140:340, 220:420] = disp_bgr

            # Procesar
            digit_28 = raw_digit.astype('float32') / 255.0
            self.roi_has_content = True
            self.predict(digit_28)

            # Rótulos
            pred_label = f"NUMERO: {self.current_digit} (Real: {ground_truth})"
            conf_label = f"ACCURACY: {self.current_confidence:.1f}%"
            cv2.putText(sim_frame, pred_label, (rx1 - 20, ry2 + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 50), 2)
            cv2.putText(sim_frame, conf_label, (rx1 - 20, ry2 + 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

            cv2.putText(sim_frame, "[ESPACIO] / [N] Siguiente digito | [Q] Salir", (10, 460), 
                        cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 200, 200), 1)

            # Dibujar osciloscopio
            scope_panel = self.draw_oscilloscope(width=440, height=480)
            combined = np.hstack((sim_frame, scope_panel))

            cv2.imshow("Detector de Numeros con Osciloscopio (OpenCV)", combined)
            key = cv2.waitKey(60) & 0xFF
            if key in [ord('q'), 27]:
                break
            elif key in [ord(' '), ord('n')]:
                current_idx = (current_idx + 1) % total_samples

        cv2.destroyAllWindows()

if __name__ == '__main__':
    model_file = 'mnist_model.keras'
    if len(sys.argv) > 1:
        model_file = sys.argv[1]

    app = DigitClassifierApp(model_path=model_file)
    app.run()
