import tensorflow as tf


def build_model(kind):
    """Build the reference architectures from referencia/scratchpad.py."""
    if kind == "dense":
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(784,), name="pixels_784"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_64_a"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_64_b"),
            tf.keras.layers.Dense(10, activation="softmax", name="class_probabilities"),
        ], name="mnist_dense_reference")
    elif kind == "cnn":
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1), name="image_28x28"),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv_32"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv_64"),
            tf.keras.layers.Flatten(name="flatten_features"),
            tf.keras.layers.Dense(10, activation="softmax", name="class_probabilities"),
        ], name="mnist_cnn_reference")
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
