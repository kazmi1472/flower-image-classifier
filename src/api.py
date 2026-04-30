"""FastAPI inference service.

  uvicorn src.api:app --host 0.0.0.0 --port 8000

  GET  /health   — readiness + labels
  POST /predict  — multipart file upload, returns top-k
"""

from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .predict import FlowerPredictor
from .utils import get_logger

logger = get_logger("api")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MiB

predictor: FlowerPredictor | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global predictor
    try:
        predictor = FlowerPredictor.from_config(os.environ.get("FLOWERML_CONFIG"))
        logger.info("model loaded — labels: %s", list(predictor.labels.values()))
    except FileNotFoundError as exc:
        # Allow boot without a model so /health can report not-ready.
        logger.error("model not loaded: %s", exc)
        predictor = None
    yield


app = FastAPI(title="Flower Recognition API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    if predictor is None:
        return {"status": "model_unavailable", "ready": False}
    return {"status": "ok", "ready": True, "labels": list(predictor.labels.values())}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    if predictor is None:
        raise HTTPException(503, "Model not loaded — train one first")
    if top_k < 1 or top_k > len(predictor.labels):
        raise HTTPException(400, f"top_k must be in 1..{len(predictor.labels)}")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large")

    try:
        Image.open(io.BytesIO(raw)).verify()
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc

    try:
        t0 = time.perf_counter()
        result = predictor.predict(raw, top_k=top_k)
        result["inference_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return result
    except Exception:
        logger.exception("inference failed")
        raise HTTPException(500, "Inference error")
