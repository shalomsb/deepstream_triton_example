# DeepStream + Triton: YOLO26x and RF-DETR Large

Real-time multi-stream object detection and tracking on DeepStream 9.0 with Triton Inference Server. Two interchangeable detector ensembles:

- **YOLO26x** — 80 COCO classes, post-NMS output, FP16 TRT
- **RF-DETR Large** — DETR-style (no NMS), 90 COCO classes (+ background), FP16 TRT, letterbox preprocess

Both run as Triton ensembles with GPU-resident Python pre/postprocess (PyTorch + DLPack zero-copy). Switch between them with a single line in `config.yaml`.

## Features

- Up to **8 simultaneous streams**, any mix of file / RTSP / USB. Auto-sized streammux batch + tiler grid.
- **GPU-resident ensembles** (`KIND_GPU` + `FORCE_CPU_ONLY_{INPUT,OUTPUT}_TENSORS=no`) using `torch.utils.dlpack` for zero-copy in/out.
- **Dynamic-batch TRT engines** (min=1, opt=4, max=8) FP16.
- **Per-stream FPS** printed every 5 s by `common/FPS.PERF_DATA`.
- **`ds_pipeline` toolkit** wraps GStreamer/pyds element factories, metadata iteration, OSD helpers.
- **NvDCF tracker** by default (per-stream-native), IOU available for fast-and-simple.

---

## Quick Start

### Requirements

- NVIDIA GPU (any modern RTX/Tesla; tested on RTX 5070 Laptop, compute 12.0)
- Docker with `nvidia-container-toolkit`
- ~10 GB free disk for the image + engines

### Build (one-time)

```bash
./docker/launch.sh -b
```

Builds the Docker image (DeepStream 9.0 + Triton + PyTorch CUDA), then runs the two export scripts inside the container:

- `scripts/install_models.sh`   → downloads YOLO26x and produces `triton/model_repo/yolo26x/1/model.plan`
- `scripts/install_rfdetr.sh`   → pip-installs `rfdetr`, exports RF-DETR Large with `dynamic_batch=True`, produces `triton/model_repo/rfdetr_large/1/model.plan`

Both engines are built with `--minShapes=...1x... --optShapes=...4x... --maxShapes=...8x... --fp16`.

### Run

```bash
./docker/launch.sh -r      # run the DeepStream app
./docker/launch.sh -d      # dev shell inside the container
./docker/launch.sh -s      # Triton server only (HTTP 8000, gRPC 8001, metrics 8002)
```

---

## Configuration

### Top-level: `deepstream/configs/config.yaml`

```yaml
sources:
  - file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264
  # add up to 8 entries: file URI, rtsp://, http(s)://, /dev/videoN

streammux:
  width: 1280
  height: 720
  # batch_size: omitted -> defaults to len(sources), clamped 1..8

tiler:
  width: 1280
  height: 720
  # rows/cols: omitted -> auto-grid from len(sources)
  #   1 -> 1x1, 2 -> 1x2, 3-4 -> 2x2, 5-6 -> 2x3, 7-8 -> 2x4

pgie:
  # Swap this single line to switch detector.
  config_file: configs/pgie_rfdetr.yaml   # or: configs/pgie_yolo26x.yaml

tracker:
  config_file: configs/tracker_config_dcf.txt

file_source:
  loop: true
```

### Per-detector: `deepstream/configs/pgie_*.yaml`

Each file self-contains every per-model knob:

```yaml
# pgie_yolo26x.yaml                       # pgie_rfdetr.yaml
infer_config: config_infer.txt            # config_infer_rfdetr.txt
network_width: 640                        # 704
network_height: 640                       # 704
conf_threshold: 0.25                      # 0.1
maintain_aspect_ratio: false              # true   (letterbox + ImageNet norm)
labels_file: /deepstream/labels.txt       # /deepstream/labels_coco91.txt
```

`maintain_aspect_ratio` tells the DS-side parser whether the ensemble boxes are in plain network-resize space (YOLO26x) or in letterboxed network-pixel space (RF-DETR) — the parser inverts accordingly using streammux dims.

### Tracker

```yaml
tracker:
  config_file: configs/tracker_config_dcf.txt   # NvDCF, per-stream (default)
  # config_file: configs/tracker_config_iou.txt # IOU, fast/simple
```

---

## Architecture

```
                            INPUT (1..8 streams)
       file / RTSP / USB (mixed)
                |
                v
  +-----------------------------------------------------+
  |              DEEPSTREAM PIPELINE                    |
  |                                                     |
  |  N x nvurisrcbin/v4l2 -> nvstreammux -> q ->        |
  |  nvinferserver -> q -> nvtracker -> q ->            |
  |  nvmultistreamtiler -> q -> nvvideoconvert ->       |
  |  q -> nvdsosd -> nveglglessink (sync=0, qos=0)      |
  |                       |                             |
  |                       v                             |
  |              TRITON ENSEMBLE                        |
  |          (one of yolo26x_ensemble                   |
  |           or rfdetr_large_ensemble)                 |
  |                                                     |
  +-----------------------------------------------------+
                |
                v
       CALLBACK PROBES
  pgie_src_probe  : ensemble tensors -> NvDsObjectMeta
                    (linear scale OR letterbox inverse)
  osd_probe       : per-object labels (skip "background"/"N/A")
  PERF_DATA       : per-stream fps every 5 s
```

### YOLO26x ensemble  (`yolo26x_ensemble`)

```
input_images UINT8 [3,1080,1920]
   |
   v   yolo26_preprocess  (Python KIND_GPU + DLPack)
       cv-free: F.interpolate to 640x640, /255  -> FP32
   |
   v   yolo26x            (TRT FP16, dynamic batch 1..8)
       [3,640,640] -> output0 [300,6]  (x1,y1,x2,y2,conf,cls; post-NMS)
   |
   v   yolo26_postprocess (Python KIND_GPU + DLPack)
       Split + xyxy->xywh, all vectorised over the batch
   |
   v
ensemble_labels [300]  ensemble_scores [300]  ensemble_boxes [300,4]
```

### RF-DETR Large ensemble  (`rfdetr_large_ensemble`)

```
input_images UINT8 [3,720,1280]   (matches streammux dims)
   |
   v   rfdetr_preprocess  (Python KIND_GPU + DLPack)
       Letterbox via F.interpolate + F.pad, ImageNet (x/255 - mean)/std
       -> FP32 [3,704,704]
   |
   v   rfdetr_large       (TRT FP16, dynamic batch 1..8)
       dets [300,4]  (cxcywh-normalized)
       labels [300,91] (raw logits)
   |
   v   rfdetr_postprocess (Python KIND_GPU + DLPack)
       sigmoid + top-K over 300 x 91 (official RF-DETR algorithm)
       -> labels [300], scores [300], boxes [300,4] in 704-px NETWORK space
       (letterbox inverse is applied DS-side; ensemble stays source-agnostic)
   |
   v
ensemble_labels [300]  ensemble_scores [300]  ensemble_boxes [300,4]
```

Both ensembles expose the **same output layer names** so the DS parser is shared.

### Triton model repository

```
triton/model_repo/
├── yolo26_preprocess/        # Python KIND_GPU, DLPack
├── yolo26x/                  # TensorRT FP16 (1/4/8 dynamic batch)
├── yolo26_postprocess/       # Python KIND_GPU, DLPack
├── yolo26x_ensemble/         # ensemble: pre -> yolo26x -> post
├── rfdetr_preprocess/        # Python KIND_GPU, DLPack
├── rfdetr_large/             # TensorRT FP16 (1/4/8 dynamic batch)
├── rfdetr_postprocess/       # Python KIND_GPU, DLPack
└── rfdetr_large_ensemble/    # ensemble: pre -> rfdetr_large -> post
```

---

## Project structure

```
deepstream_triton_example/
├── deepstream/
│   ├── main.py                       # entry point: source bins, tiler, pgie/tracker/osd
│   ├── config.py                     # Config: sources/streammux/tiler/tracker/pgie
│   ├── callbacks.py                  # pgie_src_probe + osd_probe (shared by both detectors)
│   ├── labels.txt                    # 80-class COCO (YOLO26x)
│   ├── labels_coco91.txt             # 91-class with bg + N/A gaps (RF-DETR)
│   ├── configs/
│   │   ├── config.yaml               # main app config
│   │   ├── pgie_yolo26x.yaml         # per-detector knobs (YOLO26x)
│   │   ├── pgie_rfdetr.yaml          # per-detector knobs (RF-DETR)
│   │   ├── config_infer.txt          # nvinferserver -> yolo26x_ensemble
│   │   ├── config_infer_rfdetr.txt   # nvinferserver -> rfdetr_large_ensemble
│   │   ├── tracker_config_dcf.txt    # NvDCF tracker
│   │   ├── tracker_config_iou.txt    # IOU tracker
│   │   └── config_tracker_*.yml      # tracker low-level libs
│   ├── ds_pipeline/                  # reusable DS toolkit (factories, meta iter, OSD, ...)
│   └── common/
│       ├── FPS.py                    # PERF_DATA per-stream fps
│       ├── bus_call.py
│       └── platform_info.py
├── triton/model_repo/                # see above
├── docker/{Dockerfile, entrypoint.sh, launch.sh, requirements.txt}
└── scripts/
    ├── install_models.sh             # YOLO26x ONNX export + trtexec
    └── install_rfdetr.sh             # RF-DETR Large ONNX export + trtexec
```

---

## References

- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Roboflow RF-DETR](https://github.com/roboflow/rf-detr)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
