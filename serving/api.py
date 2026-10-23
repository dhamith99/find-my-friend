"""
Find My Friend - model serving API (FastAPI).

Loads the @champion version of all three registered models from the MLflow
Model Registry at start-up. Interactive docs: http://127.0.0.1:8000/docs

    uvicorn serving.api:app --port 8000          (from the project root)

Endpoints
  GET  /health                 service status + loaded model versions
  GET  /models                 registered models, versions and their test scores
  POST /predict?model=resnet18 one model   (form field "file" = image)
  POST /predict_all            all three models on the same image
  POST /reload                 load the newest @champion versions without restarting
  GET  /stats                  live monitoring: predictions per label, confidence, latency

Every prediction is appended to logs/predictions.jsonl (time, model, version,
label, confidence, latency) - simple production monitoring. No images are stored.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference import FacePredictor

predictor: FacePredictor | None = None
LOG_PATH = Path(os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl"))
_log_lock = threading.Lock()
_recent = deque(maxlen=500)          # last predictions, for /stats
_started = time.time()


_request_no = 0


def log_predictions(out: dict, source: str):
    global _request_no
    with _log_lock:
        _request_no += 1
        base = {"ts": round(time.time(), 3), "request": _request_no, "source": source}
    if not out["face_found"]:
        rows = [{**base, "face_found": False}]
    else:
        rows = [{**base, "face_found": True, "model": r["model"], "version": r["version"],
                 "label": r["label"], "raw_label": r["raw_label"],
                 "confidence": r["confidence"], "latency_ms": r["latency_ms"]}
                for r in out["results"]]
    with _log_lock:
        _recent.extend(rows)
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
        except OSError:
            pass  # logging must never break predictions


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = FacePredictor()
    yield


app = FastAPI(title="Find My Friend API",
              description="Recognises dhamith, pawara, uchitha or unknown from a photo, "
                          "using three models served from the MLflow Model Registry.",
              version="1.0", lifespan=lifespan)


async def read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "The uploaded file is not an image") from None


def check_model(name: str):
    if name not in predictor.models:
        raise HTTPException(404, f"Model '{name}' is not loaded. Loaded: {list(predictor.models)}. "
                                 f"Errors: {predictor.errors}")


@app.get("/health")
def health():
    return {"status": "ok" if predictor.models else "no models loaded",
            "alias": predictor.settings["alias"],
            "models": {n: m.version for n, m in predictor.models.items()},
            "errors": predictor.errors,
            "unknown_threshold": predictor.settings["unknown_threshold"]}


@app.get("/models")
def models():
    return [{"model": n, "registered_name": f"fmf-{n}", "version": m.version,
             "alias": predictor.settings["alias"], "run_id": m.run_id,
             "classes": m.classes, "tags": m.tags}
            for n, m in predictor.models.items()]


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  model: str = Query("resnet18", description="custom_cnn | mobilenet_v3_small | resnet18"),
                  detect_face: bool = Query(True, description="false if the image is already a face crop"),
                  threshold: float | None = Query(None, ge=0, le=1,
                                                  description="override the unknown threshold")):
    check_model(model)
    img = await read_image(file)
    out = predictor.predict(img, [model], detect_face, threshold)
    log_predictions(out, "predict")
    if not out["face_found"]:
        return {"face_found": False, "box": None, "result": None}
    return {"face_found": True, "box": out["box"], "result": out["results"][0]}


@app.post("/predict_all")
async def predict_all(file: UploadFile = File(...), detect_face: bool = Query(True),
                      threshold: float | None = Query(None, ge=0, le=1)):
    img = await read_image(file)
    out = predictor.predict(img, None, detect_face, threshold)
    log_predictions(out, "predict_all")
    return out


@app.post("/reload")
def reload():
    versions = predictor.reload()
    return {"reloaded": versions, "errors": predictor.errors}


@app.get("/stats")
def stats():
    """Summary of the most recent predictions (per model): label counts, confidence, latency."""
    with _log_lock:
        rows = list(_recent)
    faces = [r for r in rows if r["face_found"]]
    per_model = {}
    for name in predictor.models:
        mr = [r for r in faces if r["model"] == name]
        if not mr:
            continue
        per_model[name] = {
            "predictions": len(mr),
            "labels": dict(Counter(r["label"] for r in mr)),
            "mean_confidence": round(sum(r["confidence"] for r in mr) / len(mr), 4),
            "below_threshold_rate": round(sum(r["label"] != r["raw_label"] for r in mr) / len(mr), 4),
            "mean_latency_ms": round(sum(r["latency_ms"] for r in mr) / len(mr), 2),
        }
    requests = {r["request"] for r in rows}
    no_face = {r["request"] for r in rows if not r["face_found"]}
    return {"uptime_s": round(time.time() - _started), "requests": len(requests),
            "no_face_rate": round(len(no_face) / max(1, len(requests)), 4),
            "models": per_model}
