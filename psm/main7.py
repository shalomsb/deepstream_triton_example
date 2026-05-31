# LEVEL 7: + per-frame HUD overlay. classic equivalent: in the osd-sink probe,
#          nvds_acquire_display_meta_from_pool() -> fill NvOSD_TextParams ->
#          nvds_add_display_meta_to_frame().
#          NOTE the osd field-name split (verified in deepstream-dev/service_maker_api.md):
#            main5 per-OBJECT label uses obj.text_params -> RAW osd.TextParams
#                  (font_params / set_bg_clr / text_bg_clr)
#            main7 per-FRAME HUD    uses osd.Text()      -> FRIENDLY wrapper
#                  (font        / set_bg_color / bg_color)
#          chain: streammux -> nvinferserver -[conv]-> nvtracker -[label+hud]-> render

from pyservicemaker import Pipeline, Flow, BatchMetadataOperator, Probe, osd
from runner import run
import torch

VIDEO = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
SOURCES = [VIDEO, VIDEO]
INFER = "/deepstream/configs/config_infer.txt"
LABELS_FILE = "/deepstream/labels.txt"

# nvtracker low-level lib + config (same values as configs/tracker_config_dcf.txt)
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
TRACKER_CFG = "/deepstream/configs/config_tracker_NvDCF_perf.yml"

NETWORK = 640
CONF = 0.25
STREAM_W, STREAM_H = 1920, 1080

with open(LABELS_FILE) as f:
    LABELS = [ln.strip() for ln in f if ln.strip()]

def _np(t):
    return torch.utils.dlpack.from_dlpack(t).cpu().numpy()

# per-OBJECT label helper (RAW osd.TextParams field names: font_params/set_bg_clr/text_bg_clr)
def _set_label(obj, text):
    tp = obj.text_params
    tp.display_text = text.encode("ascii")
    tp.x_offset = int(obj.rect_params.left)
    tp.y_offset = max(0, int(obj.rect_params.top) - 18)
    fp = tp.font_params
    fp.name = osd.FontFamily.Serif
    fp.size = 14
    fp.color = osd.Color(1.0, 1.0, 1.0, 1.0)
    tp.font_params = fp
    tp.set_bg_clr = True
    tp.text_bg_clr = osd.Color(0.0, 0.0, 0.0, 1.0)
    obj.text_params = tp

# PROBE 1 (nvinferserver src): tensors -> NvDsObjectMeta boxes (NO text yet)
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

# PROBE 2 (nvtracker src): per-object labels + per-frame HUD
class LabelHudProbe(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for frame in batch_meta.frame_items:
            # per-object label: name + conf + tracker ID (same as main6)
            count = 0
            for obj in frame.object_items:
                count += 1
                cid = obj.class_id
                name = LABELS[cid] if 0 <= cid < len(LABELS) else str(cid)
                _set_label(obj, f"{name} {obj.confidence:.2f} ID:{obj.object_id}")

            # --- main7 delta: per-frame HUD ---
            # nvds_acquire_display_meta_from_pool()
            dmeta = batch_meta.acquire_display_meta()
            # NvOSD_TextParams -> osd.Text() FRIENDLY wrapper (font/set_bg_color/bg_color)
            text = osd.Text()
            text.display_text = (
                f"src {frame.source_id}  frame {frame.frame_number}  objs {count}"
            ).encode("ascii")
            text.x_offset = 12
            text.y_offset = 12
            # friendly nested struct: assign fields directly (no get/mutate/assign-back)
            text.font.name = osd.FontFamily.Serif
            text.font.size = 16
            text.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
            text.set_bg_color = True
            text.bg_color = osd.Color(0.0, 0.0, 0.0, 0.6)
            # NvOSD display element list -> add_text()
            dmeta.add_text(text)
            # nvds_add_display_meta_to_frame()
            frame.append(dmeta)

def build():
    pipeline = Pipeline("psm-main7")
    return (
        Flow(pipeline)
        # N x nvurisrcbin + 1 nvstreammux
        .batch_capture(SOURCES)
        # nvinferserver -> yolo26x_ensemble (raw tensors)
        .infer(INFER, with_triton=True)
        # PROBE 1: tensors -> boxes (must run before the tracker)
        .attach(Probe("convert", DetectionConverter()))
        # nvtrackerbin: ll-lib-file / ll-config-file (kwargs: _ -> -)
        .track(ll_lib_file=TRACKER_LIB, ll_config_file=TRACKER_CFG,
               tracker_width=640, tracker_height=384)
        # PROBE 2: labels + HUD (after tracker so object_id exists)
        .attach(Probe("label_hud", LabelHudProbe()))
        # auto-tile + nvvideoconvert + nvdsosd + nveglglessink
        .render()
    )

if __name__ == "__main__":
    run(build)
