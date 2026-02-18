#!/usr/bin/env bash

DETR_MODEL_DIR=/triton/model_repo/ddetr/1
# if the models are already present, exit
if [ -f ${DETR_MODEL_DIR}/model_vit_tiny.plan ] && [ -f ${DETR_MODEL_DIR}/model_resnet50.plan ]; then
    echo "DETR models already present. Exiting installation script."
    exit 0
fi

cd /deepstream/

ONNX_DETR_MODEL_FILE_NAME=ddetr_vit_tiny.onnx
# For VIT backbone:
WGET_URL='https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/pretrained_deformable_detr_coco/ddetr_gc_vit_tiny_deployable_v1.0/files?redirect=true&path=dd_gcvit_tiny_ep50.onnx'
wget --content-disposition ${WGET_URL} --output-document ${ONNX_DETR_MODEL_FILE_NAME}

# Export the VIT model to TRT (pass full path)
/opt/scripts/export_trt.sh /deepstream/${ONNX_DETR_MODEL_FILE_NAME} ddetr_vit_tiny_model
# Place the TRT model in the target model directory for safe-keeping
mv /deepstream/ddetr_vit_tiny_model.engine /triton/model_repo/ddetr/1/model_vit_tiny.plan

# For ResNet 50 Backbone:
ONNX_DETR_MODEL_FILE_NAME=ddetr_resnet50.onnx
WGET_URL='https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/pretrained_deformable_detr_coco/ddetr_resnet_50_deployable_v1.0/files?redirect=true&path=dd_resnet50_ep50.onnx'
wget --content-disposition ${WGET_URL} --output-document ${ONNX_DETR_MODEL_FILE_NAME}

# Export the Resnet 50 model to TRT (pass full path)
/opt/scripts/export_trt.sh /deepstream/${ONNX_DETR_MODEL_FILE_NAME} ddetr_resnet_50_model

# Place the model in the target model repository
mv /deepstream/ddetr_resnet_50_model.engine /triton/model_repo/ddetr/1/model_resnet50.plan
# Place the TRT model in the target model directory as target model plan
cp /triton/model_repo/ddetr/1/model_vit_tiny.plan /triton/model_repo/ddetr/1/model.plan

# Setup DETR pre-process module TRT
cd /triton/model_repo/ddetr_preprocess/
python3 export_preprocess_trt.py
mv ddter_1batch_960tw_544th_-1w_-1h.plan 1/model.plan
# rm ddter_4batch_960tw_544th_-1w_-1h.onnx.data
rm ddter_1batch_960tw_544th_-1w_-1h.onnx
