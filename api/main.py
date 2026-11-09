from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference import load_model, predict

app = FastAPI(title="Handwritten Digit Recognition API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "handwritten-digit-recognition"}


@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="Upload a PNG or JPEG image.")
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller.")
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    try:
        result = predict(image, load_model())
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return result
