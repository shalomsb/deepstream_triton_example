// Custom nvinferserver bbox parser for the yolo26x_ensemble.
//
// WHY THIS EXISTS (the tracker fix, "Option A" in psm/FINDINGS.md):
//   The raw-tensor path (postprocess { other {} } + output_tensor_meta) makes
//   nvinferserver call attachTensorOutputMeta(), which NEVER sets
//   frameMeta->bInferDone. nvtracker gates on that flag
//   (nvtracker_proc.cpp: detectionDone = bInferDone) and so assigns object_id=0
//   to everything. Routing detections through the *detection* postprocess path
//   (this parser) makes nvinferserver call attachDetectionMetadata() instead,
//   which DOES set bInferDone -> the tracker tracks.
//
// The ensemble already ran NMS, so this is a pure reformat of its three output
// layers into NvDsInferObjectDetectionInfo. No clustering needed downstream.
//
//   ensemble_labels  INT64 [300]     class id per detection
//   ensemble_scores  FP32  [300]     confidence per detection
//   ensemble_boxes   FP32  [300,4]   xywh in the 640x640 yolo network space
//
// Boxes are emitted in networkInfo pixel space (the ensemble input dims, e.g.
// 1920x1080); nvinferserver then rescales networkInfo -> frame surface.

#include <cstdint>
#include <algorithm>
#include <vector>
#include <cstring>
#include "nvdsinfer_custom_impl.h"

// The yolo26x sub-model runs at a fixed 640x640; ensemble_boxes are in that space.
static const float kYoloNet = 640.0f;
static const int kNumDetections = 300;

extern "C" bool NvDsInferParseCustomYolo26Ensemble(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

static const NvDsInferLayerInfo *findLayer(
    std::vector<NvDsInferLayerInfo> const &layers, const char *name)
{
    for (auto const &l : layers) {
        if (l.layerName && std::strcmp(l.layerName, name) == 0)
            return &l;
    }
    return nullptr;
}

extern "C" bool NvDsInferParseCustomYolo26Ensemble(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    const NvDsInferLayerInfo *labelsLayer = findLayer(outputLayersInfo, "ensemble_labels");
    const NvDsInferLayerInfo *scoresLayer = findLayer(outputLayersInfo, "ensemble_scores");
    const NvDsInferLayerInfo *boxesLayer  = findLayer(outputLayersInfo, "ensemble_boxes");

    if (!labelsLayer || !scoresLayer || !boxesLayer) {
        // names must match triton/model_repo/yolo26x_ensemble/config.pbtxt outputs
        return false;
    }

    const int64_t *labels = reinterpret_cast<const int64_t *>(labelsLayer->buffer);
    const float   *scores = reinterpret_cast<const float *>(scoresLayer->buffer);
    const float   *boxes  = reinterpret_cast<const float *>(boxesLayer->buffer);
    if (!labels || !scores || !boxes)
        return false;

    // ensemble boxes are 640-space; scale to networkInfo (ensemble input) pixels.
    const float sx = static_cast<float>(networkInfo.width)  / kYoloNet;
    const float sy = static_cast<float>(networkInfo.height) / kYoloNet;
    const float maxX = static_cast<float>(networkInfo.width)  - 1.0f;
    const float maxY = static_cast<float>(networkInfo.height) - 1.0f;

    const unsigned int numClasses = detectionParams.numClassesConfigured;

    for (int i = 0; i < kNumDetections; ++i) {
        const float conf = scores[i];
        const int   cls  = static_cast<int>(labels[i]);

        if (cls < 0 || (numClasses && static_cast<unsigned int>(cls) >= numClasses))
            continue;

        // per-class pre-cluster threshold (falls back to 0 if unset)
        float thresh = 0.0f;
        if (static_cast<size_t>(cls) < detectionParams.perClassPreclusterThreshold.size())
            thresh = detectionParams.perClassPreclusterThreshold[cls];
        if (conf < thresh)
            continue;

        float x = boxes[i * 4 + 0] * sx;   // left
        float y = boxes[i * 4 + 1] * sy;   // top
        float w = boxes[i * 4 + 2] * sx;   // width
        float h = boxes[i * 4 + 3] * sy;   // height

        // clip to network frame
        x = std::min(std::max(x, 0.0f), maxX);
        y = std::min(std::max(y, 0.0f), maxY);
        if (w <= 0.0f || h <= 0.0f)
            continue;
        if (x + w > maxX) w = maxX - x;
        if (y + h > maxY) h = maxY - y;

        NvDsInferObjectDetectionInfo obj = {};   // zero-init (mandatory across DS versions)
        obj.classId = static_cast<unsigned int>(cls);
        obj.left = x;
        obj.top = y;
        obj.width = w;
        obj.height = h;
        obj.detectionConfidence = conf;
        objectList.push_back(obj);
    }

    return true;
}

// validates the function matches the NvDsInferParseCustomFunc prototype
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolo26Ensemble);
