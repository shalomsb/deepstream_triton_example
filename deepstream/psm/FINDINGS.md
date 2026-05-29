# pyservicemaker findings (psm/ learning ladder)

A step-by-step exploration of NVIDIA's **pyservicemaker** (the high-level `Flow`/`Pipeline`
API in DeepStream 9.0), rebuilding this repo's YOLO26x + Triton pipeline one feature at a
time. Each `mainN.py` adds exactly one concept on top of the previous.

> The `service-maker/` source tree referenced below is NVIDIA proprietary SDK source and is
> **not** committed here. Find it inside the container at
> `/opt/nvidia/deepstream/deepstream-9.0/service-maker/` and the installed package at
> `/usr/local/lib/python3.12/dist-packages/pyservicemaker/`.

## The ladder

| File | Adds | pyservicemaker concept |
|------|------|------------------------|
| `main1.py` | decode → display | `Flow(p).capture([uri]).render()()` — one `nvurisrcbin`, no mux |
| `main2.py` | multi-stream tile | `.batch_capture([...])` = N×`nvurisrcbin` + `nvstreammux`, auto-tile in `render()` |
| `main3.py` | Triton inference | `.infer(cfg, with_triton=True)` → `nvinferserver` (False → plain `nvinfer`) |
| `main4.py` | tensors → boxes | `BatchMetadataOperator` probe: `frame.tensor_items` → `get_layers()` → DLPack → `acquire_object_meta` → `append` |
| `main5.py` | per-box label text | set `obj.text_params` (raw `osd.TextParams`) |
| `main6.py` | tracker | `.track(ll_lib_file=…, ll_config_file=…)` — **see limitation below** |
| `runner.py` | graceful Ctrl-C | run pipeline in a child process; stop via `pipeline.stop()` for clean Triton unload |

Mapping vs the classic pyds pipeline:

- `frame.tensor_items` ≈ `iter_output_tensors(frame)`
- `tmeta.as_tensor_output().get_layers()` ≈ `pyds.get_nvds_LayerInfo` + `ctypes` cast
- `batch_meta.acquire_object_meta()` + `frame.append(obj)` ≈ `add_obj_meta(...)`

## Gotcha 1 — Ctrl-C and graceful Triton shutdown

`Flow(...)()` blocks in a C++ `wait()` that ignores Python's SIGINT, so a bare run can't be
Ctrl-C'd. The NVIDIA samples wrap it in `multiprocessing.Process` + `terminate()` — but
`terminate()` is SIGTERM (hard kill), so the Triton Python backends log
*"Non-graceful termination detected"*.

`runner.py` fixes this: run the pipeline in a child; the child **ignores SIGINT**; on Ctrl-C
the parent sets an `Event`; a child watcher thread calls `pipeline.stop()` (PLAYING → NULL)
so `nvinferserver` unloads Triton cleanly, then `wait()` returns and the child exits 0.
`build()` must **return** the Flow (no trailing `()`) so `runner` owns start/stop.

## Gotcha 2 — `osd.TextParams` field names differ from the `osd.Text` wrapper

`obj.text_params` returns a **raw `osd.TextParams`**, whose fields are NOT the same as the
friendly `osd.Text()` the samples use for HUD text:

| `osd.Text` (HUD wrapper) | `osd.TextParams` (raw, `obj.text_params`) |
|--------------------------|-------------------------------------------|
| `.font` | **`.font_params`** (a `Font`: `.name`/`.size`/`.color`) |
| `.set_bg_color` | **`.set_bg_clr`** |
| `.bg_color` | **`.text_bg_clr`** |

Both share `display_text`, `x_offset`, `y_offset`. pybind `def_readwrite` struct members can
return a **copy**, so use get → mutate → assign-back:
`tp = obj.text_params; …; obj.text_params = tp`.

Discover field names without a GPU: `strings _pydeepstream.so | grep ':ivar'`.

## Limitation — the tracker can't ID detections that come from a tensor-output probe

**Symptom:** with `main6.py` (`.track(...)`), every object shows `ID:0` — no tracking.

**Root cause (verified in DeepStream 9.0 plugin source):**

- `gst-nvinferserver/gstnvinferserver_meta_utils.cpp` sets `frameMeta->bInferDone = TRUE`
  **only** in `attachDetectionMetadata` / `attachClassificationMetadata` /
  `attachSegmentationMetadata`. The raw-tensor path **`attachTensorOutputMeta`** never sets it.
- `gst-nvtracker/nvtracker_proc.cpp:1446`:
  `pObjList->detectionDone = frameMeta.bInferDone ? true : false;`
  → with `bInferDone == FALSE` the tracker runs prediction-only and **ignores** our
  probe-created detections → `object_id` stays at the default `0`.

The classic pyds pipeline (`deepstream/callbacks.py`) compensates with
`frame.bInferDone = True` — that line is **load-bearing**, not defensive.

**Why pyservicemaker can't fix it in pure Python** (all three escape hatches are closed):

1. `FrameMetadata` (C++ `metadata.hpp`) exposes only read accessors + `append()` — **no
   `bInferDone` setter**, no native-pointer accessor.
2. `pipeline[name]` returns a pyservicemaker C++ node wrapper, **not** a `gi.Gst.Element`, so
   you can't attach a classic `gi.Gst` pad probe to use `pyds`.
3. `BufferProbe::IBufferOperator::handleBuffer` hands you a pyservicemaker `Buffer&` whose
   `GstBuffer` is an opaque `OpaqueBuffer*` (no Python address). `gi` interop is inbound only
   (`Buffer(gi.Gst.Buffer)`). So **no `pyds.gst_buffer_get_nvds_batch_meta` bridge** either.

`pyds` IS installed (`/opt/pyds_env/.../pyds.so`) but is **unreachable** from inside a
pyservicemaker pipeline.

**Working fixes (both require C++):**

- **A — native nvinferserver detection parsing:** add a custom `NvDsInferParse*.so` +
  `postprocess { detection {} }` so `nvinferserver` attaches `NvDsObjectMeta` *and* sets
  `bInferDone` itself. Cleanest at runtime; also benefits the classic pipeline.
- **B — C++ service-maker buffer-probe module:** compile a small module (template:
  `service-maker/sources/modules/sample_video_probe/`) that sets `bInferDone`.

For the learning ladder we stop at detections-only under pyservicemaker; **stable tracking
lives in the production pyds `main.py`**, which already sets `bInferDone` correctly.
