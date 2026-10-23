"""
Send one photo to the three MLflow model servers (scripts/serve_mlflow_models.sh).

    python -m serving.mlflow_client raw/dhamith/IMG_..._TIMEBURST44.jpg

MLflow's server only runs the network itself, so this client does the same
face crop + normalisation as training before sending the tensor as JSON.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np
from PIL import Image, ImageOps

from src.data.dataset import MEAN, STD
from src.data.face import FaceDetector, crop_square
from src.inference import LoadedModel, load_settings, to_tensor

CLASSES = ["dhamith", "pawara", "uchitha", "unknown"]
SERVERS = {"custom_cnn": 5001, "mobilenet_v3_small": 5002, "resnet18": 5003}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    s = load_settings()
    img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    box = FaceDetector(s["face_model"], s["face_score"], s["face_max_side"]).largest_face(np.asarray(img))
    if box is None:
        raise SystemExit("No face found in the image")
    face = crop_square(img, box, s["face_margin"])
    spec = LoadedModel("", "", "", None, CLASSES, 224, MEAN, STD)
    payload = json.dumps({"inputs": to_tensor(face, spec).numpy().tolist()}).encode()

    for name, port in SERVERS.items():
        req = urllib.request.Request(f"http://{args.host}:{port}/invocations", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                logits = np.array(json.loads(r.read())["predictions"][0])
        except Exception as e:
            print(f"{name:20s} error: {e}")
            continue
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        top = int(probs.argmax())
        print(f"{name:20s} -> {CLASSES[top]:8s} ({probs[top]:.1%})")


if __name__ == "__main__":
    main()
