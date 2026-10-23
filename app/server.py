"""
Find My Friend - real-time web app.

Serves the webcam page and forwards each frame to the model API, so the
browser only talks to this server (no CORS issues, works the same in Docker).

    uvicorn app.server:app --port 8501        then open http://127.0.0.1:8501

API_URL (env) points at the model API - http://127.0.0.1:8000 locally,
http://api:8000 inside Docker Compose.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Find My Friend - app", docs_url=None, redoc_url=None)
client = httpx.AsyncClient(base_url=API_URL, timeout=20)


async def forward(method: str, path: str, **kwargs) -> JSONResponse:
    try:
        r = await client.request(method, path, **kwargs)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Model API not reachable at {API_URL}: {e}") from None
    return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
async def health():
    return await forward("GET", "/health")


@app.get("/api/stats")
async def stats():
    return await forward("GET", "/stats")


@app.post("/api/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """?model=all -> /predict_all, otherwise /predict?model=<name>. threshold is passed through."""
    params = dict(request.query_params)
    model = params.pop("model", "all")
    files = {"file": (file.filename or "frame.jpg", await file.read(), file.content_type or "image/jpeg")}
    if model == "all":
        return await forward("POST", "/predict_all", files=files, params=params)
    return await forward("POST", "/predict", files=files, params={**params, "model": model})
