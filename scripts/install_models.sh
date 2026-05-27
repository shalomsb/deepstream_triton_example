#!/usr/bin/env bash
set -e

YOLO_MODEL_DIR=/triton/model_repo/yolo26x/1

# Skip if model already exists
if [ -f "${YOLO_MODEL_DIR}/model.plan" ]; then
    echo "YOLO26x model already present. Skipping."
    exit 0
fi

pip install ultralytics onnx onnxslim onnxruntime --quiet

# --- Export YOLO26x ONNX and build TensorRT engine ---
cd /tmp
if [ ! -f yolo26x.pt ]; then
    echo "Downloading yolo26x.pt..."
    python3 -c "from ultralytics import YOLO; YOLO('yolo26x.pt')"
fi

echo "Exporting YOLO26x to ONNX..."
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo26x.pt')
model.export(format='onnx', imgsz=640, dynamic=True, opset=17, simplify=True)
"

echo "Building YOLO26x TensorRT engine..."
mkdir -p "${YOLO_MODEL_DIR}"
trtexec --onnx=/tmp/yolo26x.onnx \
    --saveEngine="${YOLO_MODEL_DIR}/model.plan" \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640 \
    --fp16 \
    --memPoolSize=workspace:4096

# Cleanup
rm -f /tmp/yolo26x.pt /tmp/yolo26x.onnx 2>/dev/null || true

echo "YOLO26x model setup complete."
