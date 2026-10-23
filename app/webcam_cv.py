"""
Desktop alternative to the web app: OpenCV window + laptop webcam -> model API.

    python -m app.webcam_cv                     (ResNet18)
    python -m app.webcam_cv --model all         (all three, label from ResNet18)
Press q to quit. Needs the API running (bash scripts/serve_api.sh).
"""
from __future__ import annotations

import argparse
import time

import cv2
import httpx

COLORS = {"resnet18": (214, 120, 42), "mobilenet_v3_small": (52, 104, 235),
          "custom_cnn": (122, 175, 27)}   # BGR, same colours as the report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    client = httpx.Client(base_url=args.api, timeout=20, trust_env=False)
    params = {} if args.threshold is None else {"threshold": args.threshold}
    path = "/predict_all" if args.model == "all" else "/predict"
    if args.model != "all":
        params["model"] = args.model

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                     # mirror like a selfie camera
        small = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        scale = frame.shape[1] / 640
        _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        t = time.perf_counter()
        try:
            files = {"file": ("f.jpg", jpg.tobytes(), "image/jpeg")}
            d = client.post(path, files=files, params=params).json()
        except httpx.HTTPError as e:
            cv2.putText(frame, f"API error: {e}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            d = {"face_found": False}
        rtt = (time.perf_counter() - t) * 1000

        if d.get("face_found"):
            results = d.get("results") or [d["result"]]
            r = next((m for m in results if m["model"] == "resnet18"), results[0])
            x, y, w, h = (int(v * scale) for v in d["box"])
            color = COLORS.get(r["model"], (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(frame, f"{r['label']} {r['confidence']:.0%}", (x, max(30, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            for i, rr in enumerate([m for m in results if m is not r], 1):
                cv2.putText(frame, f"{rr['model']}: {rr['label']} {rr['confidence']:.0%}", (20, 30 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.get(rr["model"], (255, 255, 255)), 2)
        cv2.putText(frame, f"{rtt:.0f} ms", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.imshow("Find My Friend (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
