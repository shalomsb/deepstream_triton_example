# LEVEL 4: read the ensemble's RAW output tensors and turn them into boxes.
#          classic equivalent: callbacks.py:pgie_src_probe (pyds + ctypes).
#          pyservicemaker equivalent below uses frame.tensor_items + DLPack.

from pyservicemaker import Pipeline, Flow, BatchMetadataOperator, Probe, osd
from runner import run            # graceful Ctrl-C + clean Triton unload
import torch

# nvurisrcbin wants a URI, not a bare path -> file://<abspath>
VIDEO = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
SOURCES = [VIDEO, VIDEO]
INFER = "/deepstream/configs/config_infer.txt"

# yolo26x_ensemble facts (same as classic callbacks.py):
NETWORK = 640                 # ensemble input is square 640, PLAIN resize (no letterbox)
CONF = 0.25                   # conf_threshold
STREAM_W, STREAM_H = 1920, 1080   # must match batch_capture mux dims (its default)

# Tensor (DLPack) -> numpy. output_mem_type is CPU in config_infer.txt, so this is cheap.
def _np(t):
    return torch.utils.dlpack.from_dlpack(t).cpu().numpy()

class DetectionConverter(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        # boxes come back in 640x640 space; YOLO squashes (no aspect ratio) -> non-uniform scale
        sx, sy = STREAM_W / NETWORK, STREAM_H / NETWORK
        for frame in batch_meta.frame_items:
            n = 0
            # frame.tensor_items == iter_output_tensors(frame) in classic callbacks.py
            for tmeta in frame.tensor_items:
                # get_layers() == the pyds.get_nvds_LayerInfo + ctypes cast block
                layers = tmeta.as_tensor_output().get_layers()
                labels = _np(layers["ensemble_labels"])   # int64 [300]
                scores = _np(layers["ensemble_scores"])    # f32   [300]
                boxes = _np(layers["ensemble_boxes"])      # f32   [300,4] xywh @ 640
                for i in range(scores.shape[0]):
                    if scores[i] < CONF:
                        continue
                    x, y, w, h = boxes[i]
                    # batch_meta.acquire_object_meta + frame.append == add_obj_meta
                    obj = batch_meta.acquire_object_meta()
                    obj.class_id = int(labels[i])
                    obj.confidence = float(scores[i])
                    obj.rect_params.left = float(x) * sx
                    obj.rect_params.top = float(y) * sy
                    obj.rect_params.width = float(w) * sx
                    obj.rect_params.height = float(h) * sy
                    obj.rect_params.border_width = 2
                    obj.rect_params.border_color = osd.Color(1.0, 0.0, 0.0, 1.0)
                    frame.append(obj)
                    n += 1
            print(f"src={frame.pad_index} frame={frame.frame_number} objects={n}", flush=True)

# build() RETURNS the Flow (no trailing ()) -- runner.run owns start/stop/Ctrl-C
def build():
    pipeline = Pipeline("psm-main4")
    return (
        Flow(pipeline)
        # N x nvurisrcbin + 1 nvstreammux
        .batch_capture(SOURCES)
        # nvinferserver -> yolo26x_ensemble (emits raw tensors)
        .infer(INFER, with_triton=True)
        # attach converter on nvinferserver src: tensors -> NvDsObjectMeta (boxes)
        .attach(Probe("convert", DetectionConverter()))
        # auto-tile + nvvideoconvert + nvdsosd (draws the boxes) + nveglglessink
        .render()
    )

if __name__ == "__main__":
    run(build)
