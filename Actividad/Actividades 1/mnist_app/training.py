import json
import random
import time

import numpy as np
import tensorflow as tf

from .config import ARTIFACTS_DIR, MODEL_FILES, PREPROCESSING, SEED
from .data import for_model, load_mnist
from .evaluation import evaluate
from .models import build_model


def train(kind, epochs=30, batch_size=32, seed=SEED, samples=None):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist()
    if samples is not None:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(train_x), size=min(samples, len(train_x)), replace=False))
        train_x, train_y = train_x[chosen], train_y[chosen]
    x_train, x_val, x_test = (for_model(x, kind) for x in (train_x, val_x, test_x))
    model = build_model(kind)
    start = time.perf_counter()
    history = model.fit(
        x_train, train_y, validation_data=(x_val, val_y), epochs=epochs,
        batch_size=batch_size, verbose=2,
    )
    elapsed = time.perf_counter() - start
    metrics = evaluate(model, x_test, test_y)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / MODEL_FILES[kind]
    model.save(model_path)
    result = {
        "kind": kind, "epochs": epochs, "batch_size": batch_size, "seed": seed,
        "train_samples": int(len(train_x)), "validation_samples": int(len(val_x)),
        "test_samples": int(len(test_x)), "training_seconds": elapsed,
        "parameters": int(model.count_params()), "history": history.history,
        "metrics": metrics, "preprocessing": PREPROCESSING,
    }
    (ARTIFACTS_DIR / f"{kind}_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: result[k] for k in ("kind", "epochs", "batch_size", "seed", "train_samples", "training_seconds", "parameters")}, indent=2))
    print(f"Test accuracy: {metrics['accuracy']:.4f}; loss: {metrics['loss']:.4f}")
    print(f"Saved: {model_path}")
    return result
