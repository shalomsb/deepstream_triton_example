#!/usr/bin/env bash

apt-get update && \
    apt-get install -y pkg-config && \
    apt-get install -y git && \
    apt-get install -y zlib1g && \
    apt-get install -y zlib1g-dev && \
    apt --only-upgrade -y install openssl && \
    apt --only-upgrade -y install libpmix2

DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install ffmpeg libsm6 libxext6  -y

TRT_VERSION_MAJOR=10
TRT_VERSION_MINOR=8
TRT_VERSION_PATCH=0
TRT_VERSION_BUILD=40

TRT_VERSION_MAJOR_MINOR=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR
TRT_VERSION_MAJOR_MINOR_PATCH=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR.$TRT_VERSION_PATCH
TRT_VERSION_FULL=$TRT_VERSION_MAJOR_MINOR_PATCH.$TRT_VERSION_BUILD

CUDA_VERSION_MAJOR=12
CUDA_VERSION_MINOR=8
CUDA_VERSION_PATCH=0
CUDA_VERSION_BUILD=038
CUDA_VERSION_MAJOR_MINOR=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDA_VERSION_FULL=$CUDA_VERSION_MAJOR_MINOR.$CUDA_VERSION_PATCH.$CUDA_VERSION_BUILD
CUDNN_VERSION=9.7.0.66

export TRT_VERSION=$TRT_VERSION_FULL+cuda$CUDA_VERSION_FULL

cd /opt
export TRT_TAG="release/$TRT_VERSION_MAJOR_MINOR"
mkdir trt_oss_src
cd trt_oss_src
echo "$PWD Building TRT OSS..."
git clone -b $TRT_TAG https://github.com/nvidia/TensorRT TensorRT
cd TensorRT
git submodule update --init --recursive
mkdir -p build && cd build 
cmake .. \
    -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu \
    -DTRT_OUT_DIR=`pwd`/out \
    -DCUDA_VERSION=$CUDA_VERSION_MAJOR_MINOR \
    -DCUDNN_VERSION=$CUDNN_VERSION
make -j$(nproc)

cp out/libnvinfer_plugin.so.$TRT_VERSION_MAJOR_MINOR_PATCH /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.$TRT_VERSION_MAJOR_MINOR_PATCH
cp out/libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a
cp out/libnvonnxparser.so.$TRT_VERSION_MAJOR_MINOR_PATCH /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.$TRT_VERSION_MAJOR_MINOR_PATCH
cp out/trtexec /usr/local/bin/
cd ../../../
rm -rf trt_oss_src
