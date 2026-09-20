from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", BASE_DIR / "final_mobilenetv2_breast_cancer.keras")
)
IMG_SIZE = (224, 224)
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG"}

BACK4APP_APP_ID = os.getenv("BACK4APP_APP_ID", "").strip()
BACK4APP_REST_API_KEY = os.getenv("BACK4APP_REST_API_KEY", "").strip()
BACK4APP_SERVER_URL = os.getenv(
    "BACK4APP_SERVER_URL", "https://parseapi.back4app.com"
).rstrip("/")
BACK4APP_CLASS = os.getenv("BACK4APP_CLASS", "PredictionHistory").strip()


def back4app_configured():
    return bool(BACK4APP_APP_ID and BACK4APP_REST_API_KEY)


def back4app_headers():
    return {
        "X-Parse-Application-Id": BACK4APP_APP_ID,
        "X-Parse-REST-API-Key": BACK4APP_REST_API_KEY,
        "Content-Type": "application/json",
    }


def save_prediction(record: dict):
    response = requests.post(
        f"{BACK4APP_SERVER_URL}/classes/{BACK4APP_CLASS}",
        json=record,
        headers=back4app_headers(),
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("objectId")


def fetch_history(user_id: str, limit: int = 50):
    params = {
        "where": json.dumps({"userId": user_id}),
        "order": "-createdAt",
        "limit": max(1, min(limit, 100)),
    }
    response = requests.get(
        f"{BACK4APP_SERVER_URL}/classes/{BACK4APP_CLASS}",
        params=params,
        headers=back4app_headers(),
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("results", [])


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    app.state.model = tf.keras.models.load_model(str(MODEL_PATH))
    yield


app = FastAPI(
    title="Breast Cancer Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [
    x.strip()
    for x in os.getenv("FRONTEND_ORIGINS", "*").split(",")
    if x.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def prepare_image(image_bytes: bytes):
    """Validate basic file integrity/format and prepare image for MobileNetV2."""
    try:
        # Verify the file is a readable JPEG or PNG.
        with Image.open(BytesIO(image_bytes)) as img:
            if img.format not in ALLOWED_FORMATS:
                raise ValueError("Unsupported image format.")
            img.verify()

        # Reopen after verify(), then convert and resize.
        with Image.open(BytesIO(image_bytes)) as img:
            if img.format not in ALLOWED_FORMATS:
                raise ValueError("Unsupported image format.")
            image = img.convert("RGB").resize(IMG_SIZE)

        array = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
        return preprocess_input(array)

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            "Invalid image file. Please upload a readable JPEG or PNG."
        ) from exc


@app.get("/")
def root():
    return {
        "message": "Breast Cancer Detection API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_id: str | None = Query(default=None),
):
    image_bytes = await file.read()

    # Invalid file cases return a consistent "invalid" result.
    if not image_bytes:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "status": "invalid",
                "prediction": "invalid",
                "confidence": 0.0,
                "raw_scores": {"benign": 0.0, "invalid": 1.0, "malignant": 0.0},
                "message": "No image was uploaded.",
            },
        )

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "status": "invalid",
                "prediction": "invalid",
                "confidence": 0.0,
                "raw_scores": {"benign": 0.0, "invalid": 1.0, "malignant": 0.0},
                "message": "Image exceeds the 10 MB limit.",
            },
        )

    try:
        processed = prepare_image(image_bytes)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "status": "invalid",
                "prediction": "invalid",
                "confidence": 0.0,
                "raw_scores": {"benign": 0.0, "invalid": 1.0, "malignant": 0.0},
                "message": str(exc),
            },
        )

    output = await run_in_threadpool(
        app.state.model.predict, processed, verbose=0
    )
    raw_score = float(output[0][0])
    label, confidence = (
        ("malignant", raw_score)
        if raw_score > 0.5
        else ("benign", 1.0 - raw_score)
    )

    result = {
        "success": True,
        "status": "ok",
        "filename": file.filename,
        "prediction": label,
        "confidence": round(confidence, 4),
        "raw_score": round(raw_score, 4),
        "raw_scores": {
            "benign": round(1.0 - raw_score, 4),
            "invalid": 0.0,
            "malignant": round(raw_score, 4),
        },
        "note": (
            "For educational and research purposes only. "
            "Not a medical diagnosis."
        ),
    }

    # Store result metadata only; uploaded medical images are not retained.
    if user_id and back4app_configured():
        record = {
            "userId": user_id,
            "filename": (file.filename or "uploaded_image")[:200],
            "prediction": label,
            "confidence": round(confidence, 4),
            "rawScore": round(raw_score, 4),
            "createdAtClient": datetime.now(timezone.utc).isoformat(),
        }
        try:
            object_id = await run_in_threadpool(save_prediction, record)
            result["history_saved"] = bool(object_id)
            if object_id:
                result["history_id"] = object_id
        except requests.RequestException:
            result["history_saved"] = False
            result["history_warning"] = (
                "Prediction completed, but history could not be saved."
            )
    else:
        result["history_saved"] = False
        if user_id:
            result["history_warning"] = (
                "Back4App credentials are not configured."
            )

    return result


@app.get("/history")
async def history(
    user_id: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=100),
):
    if not back4app_configured():
        raise HTTPException(
            status_code=503, detail="Back4App is not configured."
        )

    try:
        records = await run_in_threadpool(fetch_history, user_id, limit)
        return {
            "success": True,
            "count": len(records),
            "results": records,
        }
    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve history from Back4App.",
        )
