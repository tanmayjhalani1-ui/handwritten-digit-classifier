from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "mnist_cnn.keras"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python -m src.train")
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.mean() > 127:
        pixels = 255 - pixels
    image = Image.fromarray(pixels)
    bbox = image.point(lambda value: 255 if value > 35 else 0).getbbox()
    if bbox:
        image = image.crop(bbox)
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    return np.asarray(canvas, dtype=np.float32)[np.newaxis, ..., np.newaxis]


def predict(image: Image.Image, model=None) -> dict:
    model = model or load_model()
    probabilities = model.predict(preprocess_image(image), verbose=0)[0]
    ranking = np.argsort(probabilities)[::-1]
    return {
        "digit": int(ranking[0]),
        "confidence": float(probabilities[ranking[0]]),
        "probabilities": {str(i): float(probabilities[i]) for i in range(10)},
        "top_three": [
            {"digit": int(index), "probability": float(probabilities[index])}
            for index in ranking[:3]
        ],
    }


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text())
