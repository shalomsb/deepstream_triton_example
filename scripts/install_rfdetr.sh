#!/usr/bin/env bash
set -e

RFDETR_MODEL_DIR=/triton/model_repo/rfdetr_large/1

if [ -f "${RFDETR_MODEL_DIR}/model.plan" ]; then
    echo "RF-DETR Large model already present. Skipping."
    exit 0
fi

pip install rfdetr onnx onnxruntime --quiet

mkdir -p /tmp/rfdetr
cd /tmp/rfdetr

echo "Exporting RF-DETR Large to ONNX..."
python3 -c "
from rfdetr import RFDETRLarge
m = RFDETRLarge()
m.export(output_dir='/tmp/rfdetr', dynamic_batch=True, opset_version=17)
"

# rfdetr's exporter drops 'inference_model.onnx' in output_dir.
ONNX_PATH=$(ls /tmp/rfdetr/*.onnx | head -n1)
echo "ONNX: ${ONNX_PATH}"

# Print I/O so the user can verify names match the Triton config.
python3 -c "
import onnx
m = onnx.load('${ONNX_PATH}')
print('--- inputs ---')
for t in m.graph.input:
    dims = [d.dim_value or d.dim_param or '?' for d in t.type.tensor_type.shape.dim]
    print(f'  {t.name}: {dims}')
print('--- outputs ---')
for t in m.graph.output:
    dims = [d.dim_value or d.dim_param or '?' for d in t.type.tensor_type.shape.dim]
    print(f'  {t.name}: {dims}')
"

echo "Building RF-DETR Large TensorRT engine..."
mkdir -p "${RFDETR_MODEL_DIR}"
trtexec --onnx="${ONNX_PATH}" \
    --saveEngine="${RFDETR_MODEL_DIR}/model.plan" \
    --minShapes=input:1x3x704x704 \
    --optShapes=input:4x3x704x704 \
    --maxShapes=input:8x3x704x704 \
    --fp16 \
    --memPoolSize=workspace:4096

rm -rf /tmp/rfdetr 2>/dev/null || true

echo "RF-DETR Large model setup complete."
