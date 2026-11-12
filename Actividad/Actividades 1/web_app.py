"""Local Streamlit app for Activity 1: dense vs convolutional networks on MNIST.

Run:  streamlit run web_app.py --server.address 127.0.0.1 --server.port 8501
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from mnist_app.config import ACTIVITY_DIR, ARTIFACTS_DIR, MODEL_FILES
from mnist_app.data import load_mnist
from mnist_app.preprocessing import preprocess_image

st.set_page_config(page_title="MNIST: densa y convolucional", layout="wide")
st.title("MNIST: red densa y red convolucional")
st.caption("Actividad 1 · Clasificación de dígitos manuscritos · Arquitecturas fieles a scratchpad.py")

LABELS = {"dense": "Red densa", "cnn": "Red convolucional"}


# ---------------------------------------------------------------- helpers
@st.cache_resource
def load_model(path, modified_ns):
    return tf.keras.models.load_model(path)


@st.cache_data
def mnist_test_samples(count=6):
    _, _, (test_images, test_labels) = load_mnist()
    return test_images[:count], test_labels[:count]


def model_for(kind):
    path = ARTIFACTS_DIR / MODEL_FILES[kind]
    return load_model(str(path), path.stat().st_mtime_ns) if path.exists() else None


def results_for(kind):
    path = ARTIFACTS_DIR / f"{kind}_results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pixel_table(image, decimals=1):
    """Show a 28x28 image as numbers, shading each cell by its intensity."""
    frame = pd.DataFrame(np.round(image, decimals))
    return frame.style.format(f"{{:.{decimals}f}}").background_gradient(cmap="gray", vmin=0, vmax=1)


def pad_square(image):
    """Pad a crop with black so it is displayed without distortion (display only)."""
    h, w = image.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=image.dtype)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = image
    return canvas


def predict(kind, image):
    model = model_for(kind)
    if model is None:
        st.info(f"{LABELS[kind]} sin entrenar. Entrénala en su pestaña: "
                f"`python train_mnist.py --kind {kind}`")
        return None
    batch = image.reshape(1, 784) if kind == "dense" else image[None, ..., None]
    probs = model.predict(batch, verbose=0)[0]
    st.metric("Dígito predicho", int(probs.argmax()), f"{probs.max():.1%} softmax", delta_color="off")
    st.bar_chart(pd.DataFrame({"Probabilidad": probs}, index=[str(i) for i in range(10)]))
    return probs


def train_controls(kind):
    model = model_for(kind)
    job = st.session_state.get(f"job_{kind}")
    running = bool(job and job[0].poll() is None)
    state = "entrenando" if running else "disponible" if model else "sin entrenar"
    st.write(f"Estado del modelo: **{state}**")
    with st.expander("Entrenar o reentrenar", expanded=model is None):
        with st.form(f"train_{kind}"):
            quick = st.checkbox("Modo rápido (2 048 ejemplos, 1 época) para probar el flujo", value=False,
                                key=f"quick_{kind}")
            epochs = st.number_input("Épocas (referencia: 30)", 1, 100, 30, key=f"epochs_{kind}")
            batch = st.number_input("Tamaño de lote (referencia: 32)", 1, 1024, 32, key=f"batch_{kind}")
            seed = st.number_input("Semilla", 0, 999_999, 42, key=f"seed_{kind}")
            submitted = st.form_submit_button("Entrenar en proceso separado", disabled=running)
        if submitted:
            ARTIFACTS_DIR.mkdir(exist_ok=True)
            logfile = ARTIFACTS_DIR / f"{kind}_training.log"
            cmd = [sys.executable, str(ACTIVITY_DIR / "train_mnist.py"), "--kind", kind,
                   "--batch-size", str(batch), "--seed", str(seed)]
            cmd += ["--epochs", "1", "--samples", "2048"] if quick else ["--epochs", str(epochs)]
            with logfile.open("w", encoding="utf-8") as handle:
                proc = subprocess.Popen(cmd, cwd=ACTIVITY_DIR, stdout=handle, stderr=subprocess.STDOUT)
            st.session_state[f"job_{kind}"] = (proc, str(logfile))
            st.success(f"Entrenamiento iniciado (PID {proc.pid}). La app sigue usable mientras tanto.")
            job, running = st.session_state[f"job_{kind}"], True
        if job:
            proc, logfile = job
            code = proc.poll()
            if code is None:
                st.info(f"En curso. Registro: `{logfile}`")
                st.button("Actualizar estado", key=f"refresh_{kind}")
            elif code == 0:
                st.success("Entrenamiento finalizado; el modelo nuevo ya está cargado.")
                st.session_state.pop(f"job_{kind}", None)
            else:
                st.error(f"El entrenamiento falló (código {code}). Revisa `{logfile}`.")


def history_and_metrics(result):
    if not result:
        st.info("Sin resultados guardados todavía.")
        return
    st.caption(f"Entrenado con {result['train_samples']:,} ejemplos · {result['epochs']} épocas · "
               f"lote {result['batch_size']} · semilla {result['seed']} · "
               f"{result['training_seconds']:.0f} s")
    h = result["history"]
    left, right = st.columns(2)
    left.write("Pérdida por época")
    left.line_chart(pd.DataFrame({"Entrenamiento": h["loss"], "Validación": h["val_loss"]},
                                 index=range(1, len(h["loss"]) + 1)))
    right.write("Exactitud por época")
    right.line_chart(pd.DataFrame({"Entrenamiento": h["accuracy"], "Validación": h["val_accuracy"]},
                                  index=range(1, len(h["accuracy"]) + 1)))
    m = result["metrics"]
    a, b, c = st.columns(3)
    a.metric("Exactitud en test", f"{m['accuracy']:.2%}")
    b.metric("Pérdida en test", f"{m['loss']:.4f}")
    c.metric("F1 macro en test", f"{m['classification_report']['macro avg']['f1-score']:.4f}")


def layer_table(model):
    if model is not None:
        st.dataframe(pd.DataFrame([{"Capa": layer.name, "Forma de salida": str(layer.output.shape),
                                    "Parámetros": layer.count_params()} for layer in model.layers]),
                     hide_index=True, width="stretch")
        st.write(f"Total de parámetros: **{model.count_params():,}**")


# ---------------------------------------------------------------- shared input (sidebar)
with st.sidebar:
    st.header("Dígito de entrada")
    st.caption("La misma entrada se envía a ambas redes.")
    mode = st.radio("Fuente", ["Dibujar", "Cargar imagen"], horizontal=True)
    source = None
    if mode == "Dibujar":
        white_on_black = st.toggle("Blanco sobre negro (como MNIST)", value=True)
        background = "#000000" if white_on_black else "#FFFFFF"
        stroke = "#FFFFFF" if white_on_black else "#000000"
        canvas = st_canvas(stroke_color=stroke, stroke_width=20, background_color=background,
                           height=260, width=260, drawing_mode="freedraw",
                           key=f"canvas_{white_on_black}")
        if canvas.image_data is not None:
            source = Image.fromarray(canvas.image_data.astype(np.uint8)[..., :3])
    else:
        uploaded = st.file_uploader("PNG, JPEG o BMP", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded:
            source = Image.open(uploaded)
    polarity = st.selectbox("Polaridad", ["Automática", "Invertir colores", "Mantener colores"],
                            help="Si la detección automática falla, fuerza la inversión manualmente.")

    prepared, info, original = None, None, None
    if source is not None:
        try:
            override = {"Automática": None, "Invertir colores": True, "Mantener colores": False}[polarity]
            original, prepared, info = preprocess_image(source, invert=override)
            st.image(prepared, caption="Lo que recibe la red (28×28)", width=140, clamp=True)
        except ValueError:
            st.warning("Lienzo vacío o sin trazo útil: dibuja o carga un dígito. No se hace predicción.")

polarity_tab, dense_tab, cnn_tab, compare_tab = st.tabs(
    ["Fondo negro y trazo blanco", "Red densa", "Red convolucional", "Comparación"])

# ---------------------------------------------------------------- tab: polarity
with polarity_tab:
    st.header("Cómo «ve» la red un dígito")
    st.markdown(
        "Para la red, una imagen es **una tabla de números**. En MNIST cada píxel vale entre 0 y 1:\n\n"
        "- **0 = negro = fondo** (no hay tinta).\n"
        "- **1 = blanco = trazo** (tinta máxima).\n"
        "- **Valores intermedios = bordes grises** del trazo.\n\n"
        "Blanco y negro **no** significan «reconocido» o «no reconocido»: solo indican dónde hay tinta. "
        "Las redes aprendieron con esta convención, así que **toda entrada se convierte a ella** antes de "
        "predecir. Un dígito negro sobre papel blanco tiene los números al revés (fondo = 1) y la red "
        "vería algo que nunca vio al entrenar.")

    st.subheader("Así son los datos reales de MNIST")
    samples, sample_labels = mnist_test_samples()
    st.image([s for s in samples], width=90, clamp=True,
             caption=[f"Etiqueta {int(y)}" for y in sample_labels])
    with st.expander("Ver los 784 números del primer ejemplo"):
        st.dataframe(pixel_table(samples[0]), width="stretch", height=420)

    st.subheader("Tu entrada, paso a paso")
    if info is None:
        st.info("Dibuja o carga un dígito en la barra lateral para ver la transformación.")
    else:
        steps = info["steps"]
        cols = st.columns(5)
        cols[0].image(steps["grayscale"], caption="1. Escala de grises", width="stretch")
        cols[1].image(steps["polarity"],
                      caption="2. Invertida (fondo claro detectado)" if info["inverted"]
                      else "2. Sin invertir (ya era fondo oscuro)", width="stretch")
        cols[2].image(pad_square(steps["cropped"]), caption="3. Recorte del trazo", width="stretch")
        cols[3].image(steps["boxed"], caption="4. Cabe en 20×20 sin deformar", width="stretch", clamp=True)
        cols[4].image(steps["final"], caption="5. Centrada por masa en 28×28", width="stretch", clamp=True)
        detected = "fondo claro → se invierte" if info["detected_invert"] else "fondo oscuro → no se invierte"
        st.caption(f"Detección automática: {detected}. Decisión aplicada: "
                   f"{'invertir' if info['inverted'] else 'mantener'}.")
        final = steps["final"]
        a, b, c = st.columns(3)
        a.metric("Valor del fondo", f"{final[0, 0]:.2f}")
        b.metric("Valor máximo (trazo)", f"{final.max():.2f}")
        c.metric("Píxeles con tinta", int((final > 0.1).sum()))
        with st.expander("Ver los 784 números que recibe la red"):
            st.dataframe(pixel_table(final), width="stretch", height=420)
    st.markdown(
        "**Fórmulas.** Normalización: $x = \\dfrac{p}{255}$, con $p \\in \\{0,\\dots,255\\}$ (se aplica una sola "
        "vez). Inversión: $p' = 255 - p$. Centro de masa: "
        "$\\bar r = \\dfrac{\\sum r\\,x_{r,c}}{\\sum x_{r,c}}$, $\\bar c = \\dfrac{\\sum c\\,x_{r,c}}{\\sum x_{r,c}}$; "
        "la imagen se desplaza para que $(\\bar r, \\bar c) = (14, 14)$, igual que en MNIST.")

# ---------------------------------------------------------------- tab: dense
with dense_tab:
    st.header("Red densa (perceptrón multicapa)")
    st.markdown(
        "La imagen de 28×28 se **aplana**: las 28 filas se ponen una detrás de otra y queda un vector de "
        "**784 números**. Cada neurona de una capa densa mira **todos** esos números a la vez, los "
        "multiplica por pesos, suma un sesgo y aplica ReLU: $h = \\max(0,\\, W x + b)$. La última capa "
        "usa softmax para dar 10 probabilidades que suman 1: "
        "$p_k = \\dfrac{e^{z_k}}{\\sum_j e^{z_j}}$.")
    st.code("Input(784) → Dense(64, relu) → Dense(64, relu) → Dense(10, softmax)")
    if prepared is not None:
        st.write("Tu dígito convertido en las 784 entradas (fila por fila):")
        st.image(np.repeat(prepared.reshape(1, 784), 20, axis=0), clamp=True, width="stretch")
    dense_model = model_for("dense")
    layer_table(dense_model)
    st.caption("Optimizador Adam · pérdida sparse_categorical_crossentropy · referencia: lote 32, 30 épocas.")
    train_controls("dense")
    st.subheader("Entrenamiento y evaluación")
    history_and_metrics(results_for("dense"))
    st.subheader("Predicción de tu dígito")
    if prepared is not None:
        predict("dense", prepared)
    else:
        st.info("Dibuja o carga un dígito en la barra lateral.")

# ---------------------------------------------------------------- tab: cnn
with cnn_tab:
    st.header("Red convolucional")
    st.markdown(
        "En lugar de aplanar desde el principio, la CNN conserva la forma 28×28. Cada **filtro** es una "
        "pequeña ventana de 3×3 pesos que se desliza por la imagen y responde fuerte donde encuentra su "
        "patrón (un borde, una curva). El resultado es un **mapa de características**. "
        "Sin padding, cada convolución 3×3 reduce el tamaño en 2: 28→26→24. Luego `Flatten` convierte "
        "los 24×24×64 valores en un vector y la capa softmax decide la clase.\n\n"
        "Convolución: $y_{i,j} = \\mathrm{ReLU}\\left(\\sum_{u,v} w_{u,v}\\, x_{i+u,\\,j+v} + b\\right)$.")
    st.code("Input(28×28×1) → Conv2D(32, 3×3, relu) → Conv2D(64, 3×3, relu) → Flatten → Dense(10, softmax)")
    st.caption("Sin pooling ni dropout: arquitectura fiel a scratchpad.py. Por eso casi todos los "
               "parámetros están en la capa final (36 864 × 10).")
    cnn_model = model_for("cnn")
    layer_table(cnn_model)
    train_controls("cnn")
    st.subheader("Entrenamiento y evaluación")
    history_and_metrics(results_for("cnn"))
    st.subheader("Predicción de tu dígito")
    if prepared is not None:
        predict("cnn", prepared)
    else:
        st.info("Dibuja o carga un dígito en la barra lateral.")
    if cnn_model is not None:
        st.subheader("Qué aprendió la primera capa")
        filters = cnn_model.get_layer("conv_32").get_weights()[0][:, :, 0, :16]
        scaled = (filters - filters.min()) / (np.ptp(filters) + 1e-8)
        st.caption("16 de los 32 filtros 3×3 (claro = peso positivo, oscuro = negativo). Son pesos "
                   "aprendidos, no etiquetas con significado humano.")
        st.image([np.kron(scaled[:, :, i], np.ones((10, 10))) for i in range(16)], width=60)
        if prepared is not None:
            activation_model = tf.keras.Model(cnn_model.inputs, cnn_model.get_layer("conv_32").output)
            maps = activation_model.predict(prepared[None, ..., None], verbose=0)[0]
            views = [(maps[:, :, i] - maps[:, :, i].min()) / (np.ptp(maps[:, :, i]) + 1e-8) for i in range(8)]
            st.caption("Mapas de activación de tu dígito (26×26): zonas claras = el filtro encontró su patrón.")
            st.image(views, width=90)

# ---------------------------------------------------------------- tab: comparison
with compare_tab:
    st.header("Comparación sobre el mismo conjunto de prueba")
    st.caption("Ambas redes usan la misma partición: 54 000 entrenamiento · 6 000 validación (del "
               "entrenamiento) · 10 000 prueba reservada.")
    records = [(kind, results_for(kind)) for kind in ("dense", "cnn")]
    if all(result for _, result in records):
        st.dataframe(pd.DataFrame([{
            "Red": LABELS[kind],
            "Exactitud test": f"{r['metrics']['accuracy']:.4f}",
            "Pérdida test": f"{r['metrics']['loss']:.4f}",
            "F1 macro": f"{r['metrics']['classification_report']['macro avg']['f1-score']:.4f}",
            "Tiempo de entrenamiento (s)": f"{r['training_seconds']:.0f}",
            "Parámetros": f"{r['parameters']:,}",
            "Épocas / ejemplos": f"{r['epochs']} / {r['train_samples']:,}",
        } for kind, r in records]), hide_index=True, width="stretch")
        columns = st.columns(2)
        for col, (kind, r) in zip(columns, records):
            with col:
                st.subheader(LABELS[kind])
                report = r["metrics"]["classification_report"]
                st.write("Precisión, recall y F1 por dígito")
                st.dataframe(pd.DataFrame([{"Dígito": i, "Precisión": report[str(i)]["precision"],
                                            "Recall": report[str(i)]["recall"], "F1": report[str(i)]["f1-score"]}
                                           for i in range(10)]).style.format(precision=3),
                             hide_index=True, width="stretch")
                st.write("Matriz de confusión (filas = real, columnas = predicho)")
                st.dataframe(pd.DataFrame(r["metrics"]["confusion_matrix"]).style.background_gradient(cmap="Blues"),
                             width="stretch")
                _, _, (test_images, _) = load_mnist()
                errors = r["metrics"]["errors"][:8]
                st.write("Errores reales del conjunto de prueba")
                st.image([test_images[e["index"]] for e in errors], width=70, clamp=True,
                         caption=[f"real {e['label']} → {e['prediction']}" for e in errors])
    else:
        st.info("Entrena las dos redes para ver la comparación.")

    st.subheader("Tu dígito en ambas redes")
    if prepared is not None:
        left, right = st.columns(2)
        with left:
            st.write(f"**{LABELS['dense']}**")
            predict("dense", prepared)
        with right:
            st.write(f"**{LABELS['cnn']}**")
            predict("cnn", prepared)
    else:
        st.info("Dibuja o carga un dígito en la barra lateral.")
    st.warning("Las métricas de arriba vienen del conjunto de prueba (10 000 imágenes). La predicción de "
               "tu dibujo es un solo caso. Una probabilidad softmax alta **no garantiza** que el resultado "
               "sea correcto ni detecta imágenes que no son dígitos: la red siempre reparte el 100 % entre "
               "las 10 clases.")
