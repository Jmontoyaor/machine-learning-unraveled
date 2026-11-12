import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(model, images, labels):
    loss, accuracy = model.evaluate(images, labels, verbose=0)
    probabilities = model.predict(images, verbose=0)
    predictions = probabilities.argmax(axis=1)
    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "classification_report": classification_report(
            labels, predictions, labels=list(range(10)), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(10))).tolist(),
        "errors": [
            {"index": int(i), "label": int(labels[i]), "prediction": int(predictions[i]),
             "confidence": float(probabilities[i].max())}
            for i in np.flatnonzero(predictions != labels)[:20]
        ],
    }
