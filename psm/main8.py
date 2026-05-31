# LEVEL 8: working tracker via NATIVE detection parsing ("Option A").
#          THE DELTA vs main7: we stop converting raw tensors in a Python probe.
#          Instead nvinferserver runs the DETECTION postprocess path with a custom
#          bbox parser (.so), so IT creates the NvDsObjectMeta AND sets bInferDone
#          -> nvtracker assigns real object_ids (main6's ID:0 problem is fixed).
#
#          So DetectionConverter is GONE; nvinferserver does its job:
#            main6/7:  infer(other{}) -[DetectionConverter probe]-> track
#            main8:    infer(detection{custom_parse}) ------------> track
#
#          Build the parser, then run from a container shell:
#            ./docker/launch.sh -b      # builds models + scripts/build_parser.sh
#            ./docker/launch.sh -d      # drop into the container
#            cd /psm && python3 main8.py
#          (-r launches the OLD pyds main.py, not this rung)
#          chain: streammux -> nvinferserver(detect) -> nvtracker -[label+hud]-> render

from pyservicemaker import Pipeline, Flow, BatchMetadataOperator, Probe, osd
from runner import run

# nvurisrcbin wants a URI, not a bare path -> file://<abspath>
VIDEO = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
SOURCES = [VIDEO, VIDEO]
# detection-path config (custom parser) instead of the raw-tensor config_infer.txt
INFER = "/deepstream/configs/config_infer_detection.txt"
LABELS_FILE = "/deepstream/labels.txt"

# nvtracker low-level lib + config (same values as configs/tracker_config_dcf.txt)
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
TRACKER_CFG = "/deepstream/configs/config_tracker_NvDCF_perf.yml"

with open(LABELS_FILE) as f:
    LABELS = [ln.strip() for ln in f if ln.strip()]

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

# single probe after the tracker: objects already exist (made by nvinferserver),
# object_id is now real -> draw name + conf + ID, plus the per-frame HUD.
class LabelHudProbe(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for frame in batch_meta.frame_items:
            count = 0
            for obj in frame.object_items:
                count += 1
                cid = obj.class_id
                name = LABELS[cid] if 0 <= cid < len(LABELS) else str(cid)
                _set_label(obj, f"{name} {obj.confidence:.2f} ID:{obj.object_id}")

            # per-frame HUD via the FRIENDLY osd.Text wrapper
            dmeta = batch_meta.acquire_display_meta()
            text = osd.Text()
            text.display_text = (
                f"src {frame.source_id}  frame {frame.frame_number}  objs {count}"
            ).encode("ascii")
            text.x_offset = 12
            text.y_offset = 12
            text.font.name = osd.FontFamily.Serif
            text.font.size = 16
            text.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
            text.set_bg_color = True
            text.bg_color = osd.Color(0.0, 0.0, 0.0, 0.6)
            dmeta.add_text(text)
            frame.append(dmeta)

def build():
    pipeline = Pipeline("psm-main8")
    return (
        Flow(pipeline)
        # N x nvurisrcbin + 1 nvstreammux
        .batch_capture(SOURCES)
        # nvinferserver -> yolo26x_ensemble, DETECTION postprocess + custom parser:
        # attaches NvDsObjectMeta and sets bInferDone itself (no converter probe)
        .infer(INFER, with_triton=True)
        # nvtrackerbin: now sees bInferDone=TRUE -> real object_ids
        .track(ll_lib_file=TRACKER_LIB, ll_config_file=TRACKER_CFG,
               tracker_width=640, tracker_height=384)
        # labels (with working ID) + HUD
        .attach(Probe("label_hud", LabelHudProbe()))
        # auto-tile + nvvideoconvert + nvdsosd + nveglglessink
        .render()
    )

if __name__ == "__main__":
    run(build)
