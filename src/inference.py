"""
Shared inference code - used by the FastAPI service and the real-time app.

The preprocessing here is exactly the same as in training:
  find the largest face (YuNet) -> square crop with the same margin -> resize
  -> scale to 0..1 -> ImageNet normalisation
so the models see live images the same way they saw the training data.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from PIL import Image

from src.data.face import FaceDetector, crop_square
from src.models.factory import MODEL_NAMES


@dataclass
class LoadedModel:
    name: str
    version: str
    run_id: str
    model: torch.nn.Module
    classes: list[str]
    img_size: int
    mean: list[float]
    std: list[float]
    tags: dict = field(default_factory=dict)


def load_settings(path: str = "params.yaml") -> dict:
    """params.yaml values, overridable with environment variables (used in Docker)."""
    p = yaml.safe_load(Path(path).read_text()) if Path(path).exists() else {}
    face, serve = p.get("face", {}), p.get("serve", {})
    return {
        "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI",
                                       p.get("mlflow", {}).get("tracking_uri", "http://127.0.0.1:5000")),
        "alias": os.environ.get("MODEL_ALIAS", serve.get("alias", "champion")),
        "unknown_threshold": float(os.environ.get("UNKNOWN_THRESHOLD",
                                                  serve.get("unknown_threshold", 0.6))),
        "face_model": os.environ.get("FACE_MODEL_PATH",
                                     face.get("model_path", "assets/face_detection_yunet_2023mar.onnx")),
        "face_score": float(face.get("score_threshold", 0.5)),
        "face_margin": float(face.get("margin", 0.2)),
        "face_max_side": int(face.get("max_side", 1280)),
    }


def load_champion(name: str, alias: str) -> LoadedModel:
    """Load fmf-<name>@<alias> from the MLflow Model Registry, plus its preprocess.json."""
    client = mlflow.MlflowClient()
    reg_name = f"fmf-{name}"
    mv = client.get_model_version_by_alias(reg_name, alias)
    model = mlflow.pytorch.load_model(f"models:/{reg_name}@{alias}", map_location="cpu").eval()
    meta_path = mlflow.artifacts.download_artifacts(f"runs:/{mv.run_id}/preprocess.json")
    meta = json.loads(Path(meta_path).read_text())
    return LoadedModel(name=name, version=str(mv.version), run_id=mv.run_id, model=model,
                       classes=meta["classes"], img_size=meta["img_size"],
                       mean=meta["mean"], std=meta["std"], tags=dict(mv.tags))


def to_tensor(face: Image.Image, lm: LoadedModel) -> torch.Tensor:
    face = face.resize((lm.img_size, lm.img_size), Image.BICUBIC)
    x = torch.from_numpy(np.array(face, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(lm.mean).view(3, 1, 1)
    std = torch.tensor(lm.std).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


class FacePredictor:
    """Holds the face detector and the three registered models."""

    def __init__(self, settings: dict | None = None, model_names=MODEL_NAMES):
        self.settings = settings or load_settings()
        mlflow.set_tracking_uri(self.settings["tracking_uri"])
        self.detector = FaceDetector(self.settings["face_model"], self.settings["face_score"],
                                     self.settings["face_max_side"])
        self.model_names = list(model_names)
        self.models: dict[str, LoadedModel] = {}
        self.errors: dict[str, str] = {}
        self.reload()

    def reload(self) -> dict:
        """(Re)load every model's current @champion version from the registry."""
        models, errors = {}, {}
        for name in self.model_names:
            try:
                models[name] = load_champion(name, self.settings["alias"])
            except Exception as e:  # keep serving the other models
                errors[name] = str(e)
        self.models, self.errors = models, errors
        return {n: m.version for n, m in models.items()}

    def find_face(self, img: Image.Image):
        box = self.detector.largest_face(np.asarray(img))
        if box is None:
            return None, None
        return crop_square(img, box, self.settings["face_margin"]), [round(v, 1) for v in box[:4]]

    @torch.no_grad()
    def classify(self, face: Image.Image, name: str, threshold: float | None = None) -> dict:
        lm = self.models[name]
        threshold = self.settings["unknown_threshold"] if threshold is None else threshold
        t = time.perf_counter()
        probs = torch.softmax(lm.model(to_tensor(face, lm)), 1)[0].numpy()
        latency = (time.perf_counter() - t) * 1000
        top = int(probs.argmax())
        raw_label, conf = lm.classes[top], float(probs[top])
        # open-set rule: if the model is not confident, call it "unknown"
        label = raw_label if conf >= threshold else "unknown"
        return {
            "model": name, "version": lm.version,
            "label": label, "raw_label": raw_label, "confidence": round(conf, 4),
            "below_threshold": conf < threshold, "threshold": threshold,
            "probabilities": {c: round(float(p), 4) for c, p in zip(lm.classes, probs)},
            "latency_ms": round(latency, 2),
        }

    def predict(self, img: Image.Image, names=None, detect_face: bool = True,
                threshold: float | None = None) -> dict:
        img = img.convert("RGB")
        if detect_face:
            face, box = self.find_face(img)
            if face is None:
                return {"face_found": False, "box": None, "results": []}
        else:
            face, box = img, None
        names = names or list(self.models)
        return {"face_found": True, "box": box,
                "results": [self.classify(face, n, threshold) for n in names]}
