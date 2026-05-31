# LEVEL 3: N x nvurisrcbin -> nvstreammux -> nvinferserver(Triton) -> (tiler) -> OSD -> sink
#          NOTE: this ensemble emits RAW TENSORS (output_tensor_meta:true), not boxes.
#                so inference runs but NOTHING draws yet. main4 adds the probe that converts.

from pyservicemaker import Pipeline, Flow
from multiprocessing import Process

# nvurisrcbin wants a URI, not a bare path -> file://<abspath>
VIDEO = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
# 2 URIs -> source_id 0, 1
SOURCES = [VIDEO, VIDEO]
# backend { triton { model_name: "yolo26x_ensemble" } }  -- in-process Triton, no separate server
INFER = "/deepstream/configs/config_infer.txt"

def run():
    pipeline = Pipeline("psm-main3")
    (
        Flow(pipeline)
        # N x nvurisrcbin + 1 nvstreammux (batch-size=N, 1920x1080 == ensemble input H1080 W1920)
        .batch_capture(SOURCES)
        # with_triton=True -> nvinferserver (False would build plain nvinfer, wrong for this config)
        .infer(INFER, with_triton=True)
        # auto-tile (nvmultistreamtiler) + nvvideoconvert + nvdsosd + nveglglessink
        .render()
    )()  # set_state(PLAYING) + wait for EOS  (blocking, in child)

# run the blocking GLib loop in a child so SIGINT reaches PYTHON in the parent
if __name__ == "__main__":
    p = Process(target=run)
    try:
        p.start()
        # parent blocks in Python -> Ctrl-C works
        p.join()
    except KeyboardInterrupt:
        # Ctrl-C kills the child cleanly
        p.terminate()
