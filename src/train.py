from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "mnist_cnn.keras"
METADATA_PATH = ARTIFACTS / "metadata.json"
EXPERIMENT_NAME = "handwritten-digit-recognition"


def build_model(variant: str) -> keras.Model:
    inputs = keras.Input(shape=(28, 28, 1), name="digit_image")
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomRotation(0.04)(x)
    x = layers.RandomTranslation(0.06, 0.06)(x)
    filters = [32, 64] if variant == "cnn_v1" else [32, 64, 96]
    for index, width in enumerate(filters):
        x = layers.Conv2D(width, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        if index > 0:
            x = layers.Dropout(0.15)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(10, activation="softmax", name="class_probabilities")(x)
    model = keras.Model(inputs, outputs, name=variant)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(epochs: int = 12, batch_size: int = 128) -> dict:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis].astype("float32")
    x_test = x_test[..., np.newaxis].astype("float32")
    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000, seed=42).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_test[:10000], y_test[:10000])).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    final_test_ds = tf.data.Dataset.from_tensor_slices((x_test[10000:], y_test[10000:])).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri()))
    mlflow.set_experiment(EXPERIMENT_NAME)

    candidates = []
    for variant in ("cnn_v1", "cnn_v2"):
        with mlflow.start_run(run_name=variant) as run:
            model = build_model(variant)
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
            ]
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=2,
            )
            val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
            test_loss, test_accuracy = model.evaluate(final_test_ds, verbose=0)
            params = {
                "architecture": variant,
                "filters": str([32, 64] if variant == "cnn_v1" else [32, 64, 96]),
                "epochs_requested": epochs,
                "batch_size": batch_size,
                "optimizer": "Adam",
                "learning_rate": 1e-3,
                "augmentation": "rotation_0.04_translation_0.06",
                "validation_samples": 10000,
                "test_samples": 10000,
            }
            mlflow.log_params(params)
            mlflow.log_metrics({
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "epochs_completed": len(history.history["loss"]),
            })
            mlflow.tensorflow.log_model(model, artifact_path="model")
            candidates.append({
                "variant": variant,
                "run_id": run.info.run_id,
                "val_accuracy": float(val_accuracy),
                "val_loss": float(val_loss),
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss),
            })

    best = max(candidates, key=lambda item: item["val_accuracy"])
    best_model = mlflow.tensorflow.load_model(f"runs:/{best['run_id']}/model")
    best_model.save(MODEL_PATH)
    metadata = {
        "task": "10-class handwritten digit classification",
        "dataset": "Keras MNIST",
        "input_shape": [28, 28, 1],
        "class_names": [str(i) for i in range(10)],
        "selected_model": best["variant"],
        "selection_metric": "validation_accuracy",
        "candidates": candidates,
        "test_set_note": "A fixed 10,000-image holdout from the original MNIST test split; candidate test metrics are reported for transparency, selection uses validation accuracy only.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    train(args.epochs, args.batch_size)
