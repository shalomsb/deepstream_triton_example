# TAO DETR Object Detection with DeepStream

Real-time object detection and tracking using NVIDIA TAO Deformable DETR (Detection Transformer) on DeepStream with Triton Inference Server and TensorRT optimization.

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
- [Configuration](#configuration)
- [Model Selection](#model-selection)
- [Interactive Features](#interactive-features)
- [Performance Optimization](#performance-optimization)
- [NVIDIA SDK Stack](#nvidia-sdk-stack)
- [Architecture](#architecture)

---

## Features

- **Real-time Object Detection**: 91 COCO classes detection using Deformable DETR
- **Dual Tracking Modes**: Choose between NvDCF (accurate) or IOU (fast) tracker
- **Shadow Tracking**: Maintains tracking during occlusions with fade visualization
- **Interactive Tracking**: Click-to-track specific objects with zoom window
- **Multi-source Support**: USB cameras (V4L2), RTSP streams, and video files with auto-validation
- **File Loop Control**: Configure video files to loop or play once
- **Fully Configurable**: Control detection threshold, display resolution, FPS reporting, and more via YAML
- **Optimized for Jetson**: Unified memory architecture for zero-copy GPU processing
- **Performance Tuning**: Inference interval control for FPS optimization

---

## Quick Start

### Requirements

**Hardware:**
- NVIDIA Jetson Orin AGX 64GB (or compatible Jetson device)
- USB camera, RTSP camera, or video file for input

**Software:**
- JetPack 6.2.1 (includes CUDA 12.6, cuDNN, TensorRT)
- DeepStream SDK 7.1
- Docker runtime with NVIDIA GPU support

**Verified Configuration:**
```
Platform: Jetson Orin AGX
JetPack: 6.2.1
DeepStream: 7.1-triton-multiarch
CUDA: 12.6
TensorRT: 10.3
Python: 3.10
```

### Installation

**1. Navigate to project directory:**
```bash
cd /.../TaoDetr
```

**2. Build Docker container:**
```bash
./launch.sh -b
```

This will:
- Build the Docker image with DeepStream 7.1 + Triton
- Install dependencies (PyTorch, torchvision, Python packages)
- Setup custom PyDS bindings for tracker metadata
- Download DETR models, convert them to TensorRT, and create preprocessing model as TensorRT for optimized performance
- Install Triton Python backend for GPU

**Build time:** ~5-15 minutes (first time only)

### Usage

**Run the application:**
```bash
./launch.sh -r
```

**Development mode (interactive shell):**
```bash
./launch.sh -d
```

**Default behavior:**
- Opens `/dev/video0` (USB camera)
- Displays video with bounding boxes and labels
- Runs at configured inference interval (see [Performance Optimization](#performance-optimization))

**Click to track an object:**
- Click on any detected object in the display window
- The selected object will be highlighted in red
- A zoom window appears in the top-right corner
- Tracking continues even during occlusions (shadow tracking)

---

## Configuration

### Main Configuration File: `app/configs/default_config.yaml`

```yaml
streammux:
  width: 1920              # Output resolution width
  height: 1080             # Output resolution height
  MUXER_BATCH_TIMEOUT_USEC: 33000  # Timeout (μs) for incomplete batches

pgie:
  config_file: config_infer.txt    # Triton inference config
  interval: 1                       # Inference interval (0=every frame, 1=every other frame)
  confidence_threshold: 0.3         # Detection confidence threshold (0.0-1.0)

tracker:
  use_dcf_tracker: true             # true = NvDCF (better accuracy), false = IOU (faster)
  shadow_fade_frames: 30            # Shadow tracking fade duration (frames)

display:
  tiled_output_width: 1920          # Display window width
  tiled_output_height: 1080         # Display window height
  fixed_crop_size: 400              # Zoom window size for tracked objects (pixels)
  fps_print_interval: 5.0           # Seconds between FPS reports

source: /dev/video0                 # Input source (see options below)

file_source:
  loop: true                        # Enable looping for file:// sources
```

### Source Configuration

**USB Camera (V4L2):**
```yaml
source: /dev/video0
```

**Video File:**
```yaml
source: file:///app/streams/video_chase.mp4

file_source:
  loop: true   # Video loops continuously when it reaches the end
  loop: false  # Video stops after playing once
```

**Note:** File sources are automatically validated on startup. If the file doesn't exist, the application will log an error and exit.

**RTSP Stream:**
```yaml
# Example: IP camera with authentication
source: rtsp://username:password@192.168.1.100:554/stream1

# Example: Public RTSP stream (no authentication)
source: rtsp://192.168.1.100:8554/live

# Example: H.264 stream from camera
source: rtsp://camera.local:554/h264
```

**Note:** RTSP streams are treated as live sources and automatically set the pipeline to live mode for optimal latency.

### Inference Configuration: `app/configs/config_infer.txt`

Key parameters:
- `unique_id: 1` - Primary inference engine ID (must match PyDS metadata)
- `max_batch_size: 1` - Single stream processing
- `model_name: "ddetr_ensemble"` - Triton ensemble model
- `pinned_memory_pool_byte_size` - CPU pinned memory allocation
- `cuda_device_memory` - GPU memory pool size

### Tracker Configuration

#### Choosing Between NvDCF and IOU Trackers

The application supports two tracking algorithms that can be selected via `default_config.yaml`:

**NvDCF (Discriminative Correlation Filter) - Default:**
```yaml
tracker:
  use_dcf_tracker: true
```

**Features:**
- ✅ Better accuracy for complex tracking scenarios
- ✅ Uses appearance models and motion prediction
- ✅ Robust occlusion handling (shadow tracking)
- ✅ Better for crowded scenes with overlapping objects
- ⚠️ Slightly higher computational cost

**Best for:** Production environments, crowded scenes, occlusion-heavy scenarios

**IOU (Intersection Over Union) Tracker:**
```yaml
tracker:
  use_dcf_tracker: false
```

**Features:**
- ✅ Faster and simpler algorithm
- ✅ Lower computational overhead
- ✅ Good for scenarios where objects don't overlap much
- ✅ Uses bounding box overlap for data association
- ⚠️ Less robust to occlusions

**Best for:** High-speed applications, simple tracking scenarios, testing/debugging

#### Tracker Configuration Files

Based on your selection, the application automatically uses:
- `tracker_config_dcf.txt` → points to `config_tracker_NvDCF_perf.yml`
- `tracker_config_iou.txt` → points to `config_tracker_IOU.yml`

**Shadow tracking parameters (NvDCF only):**
```yaml
shadowTrackingAge: 30          # Frames to track without detection
minDetectorConfidence: 0.3     # Minimum confidence threshold
outputShadowTracks: 1          # Enable shadow tracking metadata output
outputTerminatedTracks: 1      # Enable terminated tracks metadata output
```

### Display Configuration

Control display output and visualization parameters:

```yaml
display:
  tiled_output_width: 1920      # Display resolution width
  tiled_output_height: 1080     # Display resolution height
  fixed_crop_size: 400          # Zoom window size for tracked objects
  fps_print_interval: 5.0       # FPS report frequency (seconds)
```

**Adjusting display resolution:**
- Lower resolutions (e.g., 1280x720) reduce GPU load
- Higher resolutions provide better visualization detail
- Must match streammux resolution for best results

**Zoom window size:**
- Default: 400x400 pixels (top-right corner)
- Displays selected tracked object with aspect-ratio preservation
- Set to 0 to disable zoom window feature

### Detection Threshold

Fine-tune detection sensitivity:

```yaml
pgie:
  confidence_threshold: 0.3     # Range: 0.0 (all detections) to 1.0 (only certain)
```

**Lower threshold (0.2-0.3):**
- Detects more objects (higher recall)
- May include false positives
- Good for ensuring no objects are missed

**Higher threshold (0.4-0.6):**
- Only high-confidence detections (higher precision)
- May miss some objects
- Good for reducing false alarms

---

## Model Selection

### Available Backbones

NVIDIA TAO provides two DETR model variants:

#### 1. **ResNet-50 Backbone** (Default)
- **Size:** ~175MB ONNX / ~180MB TensorRT
- **Speed:** ~15 FPS (interval=0) ~27 FPS (interval=1) on Jetson Orin AGX
- **Best for:** Real-time applications, balanced performance

#### 2. **ViT (Vision Transformer) Backbone**
- **Size:** ~204MB ONNX / ~205MB TensorRT
- **Speed:** ~12 FPS (interval=0) ~20 FPS (interval=1) ~24 FPS (interval=2) on Jetson Orin AGX
- **Best for:** Accuracy-critical applications

### How to Switch Models

During build time, both models are automatically downloaded and converted to TensorRT:
- `app/triton_models/ddetr/1/model_resnet50.plan` (default, active as `model.plan`)
- `app/triton_models/ddetr/1/model_vit_tiny.plan` 

**To switch between models:**

```bash
cd app/triton_models/ddetr/1/

# Delete current active model
rm model.plan

# Switch to ResNet-50
cp model_resnet50.plan model.plan

# OR switch to ViT
cp model_vit_tiny.plan model.plan
```

**Restart the application:**
```bash
./launch.sh -r
```

### Model Comparison

| Feature | ResNet-50 | ViT |
|---------|-----------|-----|
| **Inference Speed** | ⚡⚡⚡ Fast | ⚡⚡ Moderate |
| **Accuracy** | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Better |
| **Small Objects** | ⭐⭐ Fair | ⭐⭐⭐ Good |
| **Recommended For** | Real-time tracking | Precision tasks |

---

## Interactive Features

### Click-to-Track

**How it works:**
1. Run the application (`./launch.sh -r`)
2. Click on any detected object in the display window
3. The object is now tracked with:
   - Red bounding box (primary tracking)
   - Track ID and class label
   - Zoom window in top-right corner (400x400px)
   - Shadow age indicator

**Shadow Tracking:**
- When an object is occluded or temporarily not detected:
  - Tracker predicts position using motion models
  - Bounding box alpha fades over 30 frames (configurable)
  - `shadow_age=N` shows frames since last detection
  - Automatically terminates if lost permanently

**Implementation details:**
- Coordinate transformation from display space to frame space
- Shadow metadata from NvDCF tracker (`NVDS_TRACKER_SHADOW_LIST_META`)
- Aspect-ratio preserving zoom with letterboxing

---

## Performance Optimization

### Inference Interval Tuning

The `pgie.interval` parameter controls how often detection runs:

**In `app/configs/default_config.yaml`:**
```yaml
pgie:
  interval: 0  # Run detection every frame
  interval: 1  # Run detection every 2nd frame (skip 1)
  interval: 4  # Run detection every 5th frame (skip 4)
```

**How it works:**
- DeepStream skips inference for intermediate frames
- Tracker fills gaps using motion prediction
- Objects remain tracked during skipped frames
- Trade-off: Lower interval = higher FPS, but delayed detection of new objects

**Recommendation:**
- **interval=1** for most use cases (~50% FPS boost, minimal accuracy loss)
- **interval=0** for rapidly moving objects or dense scenes
- **interval=2** for using ViT model

### Memory Optimization

**NVBUF_MEM_CUDA_UNIFIED (Type 4):**
Critical for Jetson performance - enables zero-copy GPU access via unified memory architecture:

```python
# In deepstream.py
streammux.set_property("nvbuf-memory-type", 4)  # Unified memory
```

---

## NVIDIA SDK Stack

This application leverages multiple NVIDIA SDKs for optimized performance:

### 1. **DeepStream SDK**
- **Purpose:** Video analytics framework
- **Features used:**
  - `nvstreammux`: Multi-source batching and synchronization
  - `nvinferserver`: Triton integration for inference
  - `nvtracker`: NvDCF multi-object tracking with shadow tracking
  - `nvdsosd`: On-screen display for bounding boxes and labels
  - `nvvideoconvert`: Color space and format conversion
  - `nvmultistreamtiler`: Multi-stream tiling for display

**Pipeline:**
```
source → nvstreammux → nvinferserver (Triton) → nvtracker →
nvvideoconvert → tiler → osd → display
```

### 2. **Triton Inference Server**
- **Purpose:** Scalable inference serving
- **Architecture:** Ensemble pipeline
  - `ddetr_preprocess`: TensorRT backend (resize, normalize)
  - `ddetr`: TensorRT backend (detection model)
  - `ddetr_postprocess`: Python backend (NMS, filtering)
  - `ddetr_ensemble`: Orchestration

**Benefits:**
- Dynamic batching (future multi-stream support)
- Model versioning and A/B testing
- GPU/CPU memory pooling

**Configuration:** `app/triton_models/*/config.pbtxt`

### 3. **TensorRT**
- **Purpose:** Deep learning inference optimization
- **Optimizations:**
  - Layer fusion (reduce memory bandwidth)
  - Kernel auto-tuning for Ampere architecture
  - Dynamic shape optimization (544x960 input)



## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT SOURCES                         │
│  USB Camera (V4L2)  │  RTSP Stream  │  Video File           │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    DEEPSTREAM PIPELINE                       │
│                                                              │
│  nvstreammux (Batching & Sync)                              │
│       │                                                      │
│       ▼                                                      │
│  nvinferserver ──────► TRITON INFERENCE SERVER              │
│       │                     │                                │
│       │                     ├─► ddetr_preprocess (Python)   │
│       │                     ├─► ddetr (TensorRT)            │
│       │                     └─► ddetr_postprocess (Python)  │
│       │                                                      │
│       ▼                                                      │
│  nvtracker (NvDCF Multi-Object Tracking)                    │
│       │                                                      │
│       ▼                                                      │
│  nvvideoconvert → tiler → osd → display                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    CUSTOM CALLBACKS                          │
│                                                              │
│  • pgie_src_pad_buffer_probe: Extract DETR results          │
│  • fps_tracker_probe: Performance monitoring                │
│  • click_probe: Interactive object selection                │
│  • draw_probe: Shadow tracking & zoom window                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

**1. Input Processing:**
- Video frames captured from source (camera/file/RTSP)
- `nvstreammux` converts to NVMM (unified memory) buffers
- Frames resized to 1920x1080 for display

**2. Inference:**
- Triton preprocessing: Resize to 960x544, normalize RGB
- TensorRT engine: Run DETR model on GPU
- Postprocessing: Filter detections (confidence > 0.3), scale bboxes

**3. Tracking:**
- NvDCF tracker assigns unique IDs to objects
- Kalman filtering for motion prediction
- Shadow tracking maintains IDs during occlusions

**4. Visualization:**
- OSD overlays bounding boxes and labels
- Custom draw probe adds zoom window for selected object
- Display sink renders to screen (nveglglessink)

### Memory Management

**Zero-Copy Pipeline (Jetson Orin):**
```
Camera → NVMM Buffer (Unified Memory)
              ↓
         GPU Inference (no copy)
              ↓
         Tracker (no copy)
              ↓
         Display (no copy)
```

**Only CPU access when:**
- Drawing zoom window (pixel manipulation)
- Custom metadata processing
- Uses `get_nvds_buf_surface` + `unmap_nvds_buf_surface`

---

## Troubleshooting

### Common Issues

**1. "Unable to get GstBuffer" errors:**
- Check camera permissions: `sudo chmod 777 /dev/video0`
- Verify camera is not in use: `lsof /dev/video0`

**2. Low FPS:**
- Increase inference interval: `pgie.interval: 2`
- Reduce resolution: `streammux.width: 1280, height: 720`

**3. Tracker loses objects frequently:**
- Decrease `pgie.interval` (run detection more often)
- Adjust tracker confidence: `minDetectorConfidence: 0.2`
- Increase shadow tracking age in tracker config

**4. Docker GPU access issues:**
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Check DeepStream container
./launch.sh -d
# Inside container:
nvidia-smi
gst-inspect-1.0 nvvideoconvert
```

---

## Project Structure

```
TaoDetr/
├── app/
│   ├── configs/
│   │   ├── default_config.yaml            # Main configuration
│   │   ├── config_infer.txt               # Triton inference config
│   │   ├── tracker_config_dcf.txt         # NvDCF tracker config
│   │   ├── tracker_config_iou.txt         # IOU tracker config
│   │   ├── config_tracker_NvDCF_perf.yml  # NvDCF settings
│   │   └── config_tracker_IOU.yml         # IOU settings
│   ├── triton_models/
│   │   ├── ddetr_preprocess/              # Preprocessing model
│   │   ├── ddetr/                         # TensorRT detection model
│   │   ├── ddetr_postprocess/             # Postprocessing model
│   │   └── ddetr_ensemble/                # Ensemble pipeline
│   ├── deepstream.py                      # Main pipeline
│   ├── ds_callbacks.py                    # GStreamer probes
│   ├── ds_bins.py                         # Source bin factory
│   ├── ds_utils.py                        # Metadata utilities
│   ├── labels.txt                         # COCO class names (91 classes)
│   ├── config.py                          # Configuration loader
│   ├── constants.py                       # System constants
│   └── utils.py                           # Helper functions
├── scripts/
│   ├── install_models.sh                  # Model download
│   ├── export_trt.sh                      # TensorRT conversion
│   └── install_triton_python_backend_gpu.sh
├── Dockerfile                             # Container build
├── launch.sh                              # Build/run script
└── README.md                              # This file
```

---


## References

- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [TAO Toolkit](https://developer.nvidia.com/tao-toolkit)
- [Deformable DETR Paper](https://arxiv.org/abs/2010.04159)
