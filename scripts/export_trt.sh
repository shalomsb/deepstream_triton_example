
#!/usr/bin/env bash

ONNX_PATH=$1
OUT_MODEL_NAME=$2

# Deformable DETR
trtexec --onnx=${ONNX_PATH} \
        --maxShapes=inputs:1x3x544x960 \
        --minShapes=inputs:1x3x544x960 \
        --optShapes=inputs:1x3x544x960 \
        --memPoolSize=workspace:4096 \
        --saveEngine=/deepstream/${OUT_MODEL_NAME}.engine