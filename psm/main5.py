# LEVEL 5: + per-box class label & confidence text.
#          classic equivalent: osd_probe in callbacks.py (set_obj_label).
#          delta vs main4: load labels.txt, set obj.text_params on each detection.

from pyservicemaker import Pipeline, Flow, BatchMetadataOperator, Probe, osd
from runner import run
import torch

VIDEO = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
SOURCES = [VIDEO, VIDEO]
INFER = "/deepstream/configs/config_infer.txt"
LABELS_FILE = "/deepstream/labels.txt"        # 80 COCO classes, sequential (YOLO26x)

NETWORK = 640
CONF = 0.25
STREAM_W, STREAM_H = 1920, 1080

# load labels once (class_id -> name). yolo class_id is a direct 0..79 index.
with open(LABELS_FILE) as f:
    LABELS = [ln.strip() for ln in f if ln.strip()]

def _np(t):
    return torch.utils.dlpack.from_dlpack(t).cpu().numpy()

class DetectionConverter(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        sx, sy = STREAM_W / NETWORK, STREAM_H / NETWORK
        for frame in batch_meta.frame_items:
            for tmeta in frame.tensor_items:
                layers = tmeta.as_tensor_output().get_layers()
                labels = _np(layers["ensemble_labels"])
                scores = _np(layers["ensemble_scores"])
                boxes = _np(layers["ensemble_boxes"])
                for i in range(scores.shape[0]):
                    if scores[i] < CONF:
                        continue
                    x, y, w, h = boxes[i]
                    cid = int(labels[i])
                    conf = float(scores[i])
                    left, top = float(x) * sx, float(y) * sy

                    obj = batch_meta.acquire_object_meta()
                    obj.class_id = cid
                    obj.confidence = conf
                    obj.rect_params.left = left
                    obj.rect_params.top = top
                    obj.rect_params.width = float(w) * sx
                    obj.rect_params.height = float(h) * sy
                    obj.rect_params.border_width = 2
                    obj.rect_params.border_color = osd.Color(1.0, 0.0, 0.0, 1.0)

                    # --- class + confidence text on the box ---
                    # raw osd.TextParams field names (NOT the osd.Text wrapper's):
                    #   font_params (a Font), set_bg_clr (bool), text_bg_clr (Color)
                    name = LABELS[cid] if 0 <= cid < len(LABELS) else str(cid)
                    tp = obj.text_params                    # get -> mutate -> assign back
                    tp.display_text = f"{name} {conf:.2f}".encode("ascii")
                    tp.x_offset = int(left)
                    tp.y_offset = max(0, int(top) - 18)      # sit just above the box
                    fp = tp.font_params
                    fp.name = osd.FontFamily.Serif
                    fp.size = 14
                    fp.color = osd.Color(1.0, 1.0, 1.0, 1.0)
                    tp.font_params = fp
                    tp.set_bg_clr = True                     # solid background -> readable
                    tp.text_bg_clr = osd.Color(0.0, 0.0, 0.0, 1.0)
                    obj.text_params = tp

                    frame.append(obj)

def build():
    pipeline = Pipeline("psm-main5")
    return (
        Flow(pipeline)
        # N x nvurisrcbin + 1 nvstreammux
        .batch_capture(SOURCES)
        # nvinferserver -> yolo26x_ensemble (raw tensors)
        .infer(INFER, with_triton=True)
        # tensors -> NvDsObjectMeta with box + label text
        .attach(Probe("convert", DetectionConverter()))
        # auto-tile + nvvideoconvert + nvdsosd (draws boxes AND text) + nveglglessink
        .render()
    )

if __name__ == "__main__":
    run(build)
