# LEVEL 2: N x nvurisrcbin -> nvstreammux -> (tiler) -> nvvideoconvert -> nvdsosd -> nveglglessink

from pyservicemaker import Pipeline, Flow
from multiprocessing import Process

VIDEO = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
# 2 URIs -> source_id 0, 1
SOURCES = [VIDEO, VIDEO, VIDEO, VIDEO, VIDEO, VIDEO, VIDEO, VIDEO]

def run():
    pipeline = Pipeline("psm-main2")
    (
        Flow(pipeline)
        # N x nvurisrcbin + 1 nvstreammux, link vsrc_%u -> sink_%u (batch-size=N, 1920x1080, 33ms)
        .batch_capture(SOURCES)
        # batched flow auto-tiled (nvmultistreamtiler) then nvvideoconvert + nvdsosd + nveglglessink
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
