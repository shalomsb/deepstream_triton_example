#!/usr/bin/env bash
# Build the custom nvinferserver bbox parser for yolo26x_ensemble.
# Runs INSIDE the container (needs DeepStream headers). Wired into entrypoint -b,
# or run manually after `./docker/launch.sh -d`:  bash /opt/scripts/build_parser.sh
set -e

PARSER_DIR=/deepstream/parser
DS_DIR=/opt/nvidia/deepstream/deepstream

# detect installed CUDA toolkit version (e.g. 12.8 / 13.0); fall back to 12.8
CUDA_VER=$(ls -d /usr/local/cuda-*/ 2>/dev/null \
  | grep -oP 'cuda-\K[0-9]+\.[0-9]+' | sort -V | tail -1)
CUDA_VER=${CUDA_VER:-12.8}

echo "Building yolo26x_ensemble bbox parser (CUDA_VER=${CUDA_VER})..."
make -C "${PARSER_DIR}" clean
make -C "${PARSER_DIR}" DEEPSTREAM_DIR="${DS_DIR}" CUDA_VER="${CUDA_VER}"

LIB="${PARSER_DIR}/libnvds_yolo26_ensemble_parser.so"
if nm -D "${LIB}" | grep -q NvDsInferParseCustomYolo26Ensemble; then
    echo "OK: ${LIB} (NvDsInferParseCustomYolo26Ensemble exported)"
else
    echo "ERROR: parser symbol not exported from ${LIB}" >&2
    exit 1
fi
