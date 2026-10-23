"""
Face detection + square crop, shared by the dataset script and the live app.

Uses OpenCV's YuNet detector (small ONNX model, fast on CPU, handles turned
heads better than the old Haar cascades).
Download the model once with:  bash scripts/download_face_model.sh
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class FaceDetector:
    def __init__(self, model_path: str | Path, score_threshold: float = 0.6,
                 max_side: int = 1280):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run: bash scripts/download_face_model.sh"
            )
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000
        )
        self.max_side = max_side

    def largest_face(self, rgb: np.ndarray):
        """Return (x, y, w, h, score) of the biggest face in an RGB image, or None."""
        h, w = rgb.shape[:2]
        # Phone photos are ~4000px wide; detecting on a smaller copy is faster
        scale = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(rgb, (round(w * scale), round(h * scale))) if scale < 1 else rgb
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

        self.detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = self.detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None

        best = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = (best[:4] / scale).tolist()
        return x, y, fw, fh, float(best[14])


def crop_square(img: Image.Image, box, margin: float = 0.2) -> Image.Image:
    """Square crop around the face box, with extra space on each side, kept inside the image."""
    x, y, w, h = box[:4]
    W, H = img.size
    cx, cy = x + w / 2, y + h / 2
    side = min(max(w, h) * (1 + 2 * margin), W, H)
    x0 = min(max(cx - side / 2, 0), W - side)
    y0 = min(max(cy - side / 2, 0), H - side)
    return img.crop((round(x0), round(y0), round(x0 + side), round(y0 + side)))
