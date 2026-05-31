# LEVEL 9: + FPS printing. classic equivalent: the GstFPSCounter / the perf
#          callback in deepstream_python_apps (PERF_DATA + a GLib timeout that
#          prints per-stream fps every 5s).
#          THE DELTA vs main8: one line — attach the BUILT-IN "measure_fps_probe".
#          pyservicemaker ships probe modules you attach by NAME (no class needed):
#            .attach(what="measure_fps_probe", name="fps_probe")
#          It prints fps to stdout. Attach it BEFORE render() (on the tracker src
#          pad) so it sees the batched per-stream frames, not the tiled output.
#          (measure_fps_probe must sit on an element with a src pad — never a sink.)
#
#          Build the parser, then run from a container shell:
#            ./docker/launch.sh -b      # builds models + scripts/build_parser.sh
#            ./docker/launch.sh -d      # drop into the container
#            cd /psm && python3 main9.py
#          chain: streammux -> nvinferserver(detect) -> nvtracker -[label+hud][fps]-> render

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

# single probe after the tracker: labels (with real object_id) + per-frame HUD
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
    pipeline = Pipeline("psm-main9")
    return (
        Flow(pipeline)
        # N x nvurisrcbin + 1 nvstreammux
        .batch_capture(SOURCES)
        # nvinferserver (detection postprocess + custom parser) -> sets bInferDone
        .infer(INFER, with_triton=True)
        # nvtrackerbin -> real object_ids
        .track(ll_lib_file=TRACKER_LIB, ll_config_file=TRACKER_CFG,
               tracker_width=640, tracker_height=384)
        # labels (with working ID) + HUD
        .attach(Probe("label_hud", LabelHudProbe()))
        # main9 delta: built-in FPS probe (prints per-stream fps to stdout)
        .attach(what="measure_fps_probe", name="fps_probe")
        # auto-tile + nvvideoconvert + nvdsosd + nveglglessink
        .render()
    )

if __name__ == "__main__":
    run(build)
