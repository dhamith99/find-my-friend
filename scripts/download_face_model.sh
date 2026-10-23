#!/usr/bin/env bash
# Download the YuNet face detector (about 230 KB) into assets/.
# Run from the project root:  bash scripts/download_face_model.sh
set -euo pipefail

MODEL="assets/face_detection_yunet_2023mar.onnx"
# media.githubusercontent.com serves the real file (the model is stored with Git LFS)
URL="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

mkdir -p assets
if [ -f "$MODEL" ] && [ "$(stat -c%s "$MODEL")" -gt 100000 ]; then
  echo "Already downloaded: $MODEL"
  exit 0
fi

echo "Downloading YuNet face detector..."
curl -fL --progress-bar -o "$MODEL" "$URL"

SIZE=$(stat -c%s "$MODEL")
if [ "$SIZE" -lt 100000 ]; then
  echo "ERROR: file is only $SIZE bytes (probably a Git LFS pointer, not the model)."
  echo "Download it by hand from: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"
  rm -f "$MODEL"
  exit 1
fi
echo "Saved $MODEL ($SIZE bytes)"
