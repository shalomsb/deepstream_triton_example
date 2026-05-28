# YOLO26x Object Detection with DeepStream + Triton

Real-time object detection and tracking using YOLO26x on DeepStream with a Triton Inference Server ensemble pipeline and TensorRT optimization.

## Features

- **YOLO26x Detection**: 80 COCO classes, post-NMS output (300 detections per frame)
- **Triton Ensemble Pipeline**: Preprocess (Python) -> YOLO26x (TensorRT) -> Postprocess (Python)
- **Dual Tracker Modes**: NvDCF (accurate) or IOU (fast)
- **Multi-source Support**: USB cameras (V4L2), RTSP streams, video files
- **ds_pipeline Toolkit**: Clean DeepStream pipeline utilities for element creation, metadata iteration, and OSD

---

## Quick Start

### Requirements

- NVIDIA GPU with CUDA support
- Docker with NVIDIA runtime (`nvidia-container-toolkit`)
- DeepStream 9.0 base image

### Build

```bash
# Build Docker image + export YOLO26x TensorRT engine
./docker/launch.sh -b
```

This will:
- Build the Docker image with DeepStream 9.0 + Triton
- Download YOLO26x weights and export to TensorRT
- Build the Triton Python backend with GPU support

### Run

```bash
# Run the DeepStream application
./docker/launch.sh -r

# Development shell
./docker/launch.sh -d

# Triton server only
./docker/launch.sh -s
```

---

## Configuration

### `deepstream/configs/config.yaml`

```yaml
source: file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4

streammux:
  width: 1920
  height: 1080
  batch_size: 1

pgie:
  config_file: configs/config_infer.txt
  preprocess_width: 640
  preprocess_height: 640
  conf_threshold: 0.25
  labels_file: /deepstream/labels.txt

tracker:
  config_file: configs/tracker_config_dcf.txt

file_source:
  loop: true
```

### Source Options

```yaml
# USB camera
source: /dev/video0

# Video file
source: file:///deepstream/streams/video.mp4

# RTSP stream
source: rtsp://192.168.1.100:554/stream1
```

### Tracker Selection

**NvDCF** (default, better accuracy):
```yaml
tracker:
  config_file: configs/tracker_config_dcf.txt
```

**IOU** (faster, simpler):
```yaml
tracker:
  config_file: configs/tracker_config_iou.txt
```

---

## Architecture

```
                         INPUT SOURCES
        USB Camera  |  RTSP Stream  |  Video File
                         |
                         v
  +--------------------------------------------------+
  |              DEEPSTREAM PIPELINE                  |
  |                                                   |
  |  nvstreammux -> nvinferserver -> nvtracker ->      |
  |  nvvideoconvert -> nvdsosd -> display              |
  |                       |                            |
  |                       v                            |
  |            TRITON ENSEMBLE                         |
  |  +------------------------------------------+     |
  |  | yolo26_preprocess (Python)               |     |
  |  |   UINT8 [3,1080,1920] -> FP32 [3,640,640]|     |
  |  |   cv2.resize + /255 normalize            |     |
  |  +------------------------------------------+     |
  |                       |                            |
  |  +------------------------------------------+     |
  |  | yolo26x (TensorRT FP16)                 |     |
  |  |   FP32 [3,640,640] -> FP32 [300,6]      |     |
  |  |   Post-NMS: (x1,y1,x2,y2,conf,cls)     |     |
  |  +------------------------------------------+     |
  |                       |                            |
  |  +------------------------------------------+     |
  |  | yolo26_postprocess (Python)              |     |
  |  |   Split into labels, scores, boxes       |     |
  |  |   Convert x1y1x2y2 -> xywh              |     |
  |  +------------------------------------------+     |
  |                                                   |
  +--------------------------------------------------+
                         |
                         v
                CALLBACK PROBES
    pgie_src_probe: Parse tensors -> NvDsObjectMeta
    osd_probe: Set labels + display count
```

### Triton Model Repository

```
triton/model_repo/
├── yolo26_preprocess/       # Python: resize + normalize
│   ├── config.pbtxt
│   └── 1/model.py
├── yolo26x/                 # TensorRT: YOLO26x inference
│   ├── config.pbtxt
│   └── 1/model.plan         # Built by install_models.sh
├── yolo26_postprocess/      # Python: split outputs
│   ├── config.pbtxt
│   └── 1/model.py
└── yolo26x_ensemble/        # Ensemble orchestration
    └── config.pbtxt
```

---

## Project Structure

```
deepstream_triton_example/
├── deepstream/
│   ├── main.py                # Entry point
│   ├── config.py              # YAML config loader (extends AppConfig)
│   ├── callbacks.py           # Tensor parsing + OSD probes
│   ├── constants.py           # Preprocess dimensions
│   ├── labels.txt             # 80-class COCO labels
│   ├── configs/
│   │   ├── config.yaml        # Main app config
│   │   ├── config_infer.txt   # nvinferserver config
│   │   ├── tracker_config_dcf.txt
│   │   ├── tracker_config_iou.txt
│   │   ├── config_tracker_NvDCF_perf.yml
│   │   └── config_tracker_IOU.yml
│   ├── ds_pipeline/           # DeepStream pipeline utilities
│   │   ├── _elements.py       # Element factory functions
│   │   ├── bins.py            # Source/output bin compositions
│   │   ├── config.py          # AppConfig base class
│   │   ├── meta.py            # Metadata iteration helpers
│   │   ├── osd.py             # OSD text/label helpers
│   │   ├── pipeline.py        # Pipeline lifecycle (run, link)
│   │   ├── rtsp.py            # RTSP server utilities
│   │   └── logger.py          # Logger
│   └── common/
│       ├── bus_call.py        # GStreamer bus message handler
│       └── platform_info.py   # GPU/platform detection
├── triton/model_repo/         # Triton model configs + code
├── docker/
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── launch.sh
│   └── requirements.txt
└── scripts/
    └── install_models.sh                    # YOLO26x export + TRT build
```

---

## References

- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
