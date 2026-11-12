from pathlib import Path

ACTIVITY_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ACTIVITY_DIR / "artifacts"
DATA_DIR = ACTIVITY_DIR / "data"
SEED = 42
IMAGE_SHAPE = (28, 28)
MODEL_FILES = {"dense": "mnist_dense.keras", "cnn": "mnist_cnn.keras"}
PREPROCESSING = {
    "version": 1,
    "representation": "grayscale float32 [0,1], black background and white foreground",
    "content_box": "crop non-background pixels, preserve aspect ratio, fit in 20x20, center in 28x28",
    "normalization": "divide uint8 intensities by 255 exactly once",
}
