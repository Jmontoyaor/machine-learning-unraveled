from functools import lru_cache

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config import SEED


@lru_cache(maxsize=1)
def load_mnist():
    """Load and prepare a reproducible train/validation/test split once per process."""
    (images, labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    images = images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    train_x, val_x, train_y, val_y = train_test_split(
        images, labels, test_size=0.1, random_state=SEED, stratify=labels
    )
    return (train_x, train_y), (val_x, val_y), (test_images, test_labels)


def for_model(images, kind):
    if kind == "dense":
        return images.reshape((-1, 784)).astype(np.float32, copy=False)
    if kind == "cnn":
        return images[..., np.newaxis].astype(np.float32, copy=False)
    raise ValueError(f"Unknown model kind: {kind}")
