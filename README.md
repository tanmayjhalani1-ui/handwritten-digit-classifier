# Handwritten Digit Recognition

A production-oriented MNIST digit-recognition application built with TensorFlow/Keras. It compares CNN architectures, tracks training runs with MLflow, serves predictions through Streamlit and FastAPI, and ships with Docker and GitHub Actions CI.

## Capabilities

- Two CNN candidates with batch normalization, dropout and lightweight image augmentation
- Validation-based architecture selection; a separate fixed holdout for final evaluation
- MLflow experiment tracking for parameters, validation/test metrics and model artifacts
- Consistent image preprocessing for uploaded handwritten digits
- Streamlit prediction dashboard with confidence scores and class probabilities
- FastAPI prediction endpoint and health check
- Docker deployment and automated architecture/import checks

## Dataset

Uses the MNIST dataset distributed by `tensorflow.keras.datasets`. The dataset is downloaded automatically on the first training run; it is not duplicated in this repository. MNIST contains 60,000 training images and 10,000 test images of 28×28 grayscale digits across ten classes.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --epochs 12 --batch-size 128
streamlit run app.py
```

Training writes `artifacts/mnist_cnn.keras` and `artifacts/metadata.json`. It compares `cnn_v1` and `cnn_v2` using validation accuracy. The 10,000-image holdout is not used for model selection. Model weights are generated locally and are excluded from Git.

## MLflow

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open the MLflow UI at `http://127.0.0.1:5000`. Each architecture is logged as a separate run under `handwritten-digit-recognition`, including parameters, metrics and the trained model artifact. These are tracked candidate runs; a Model Registry version/promotion workflow is not configured.

## FastAPI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- `GET /health`
- `POST /predict` — multipart form upload using the `file` field (PNG/JPEG)

Interactive API docs: `http://127.0.0.1:8000/docs`.

## Docker

Train the model locally first so that `artifacts/` contains the model and metadata, then run:

```bash
docker compose up --build
```

Open `http://localhost:8501`.

## Repository structure

```text
├── api/
│   └── main.py
├── src/
│   ├── train.py
│   └── inference.py
├── artifacts/                 # generated locally; gitignored
├── app.py
├── digitclassifier.py         # original app, preserved
├── digitclassification.ipynb  # original notebook, preserved
├── dataset                    # original dataset note, preserved
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Notes and limitations

MNIST consists of centered, normalized grayscale digit images. Real-world photos, unusual handwriting, colored backgrounds, multiple digits or non-centered writing may reduce accuracy despite preprocessing. This repository does not claim performance on external handwriting datasets. The original notebook and Streamlit script are retained as historical project files; the maintained application is `app.py`.
