# LEVEL 1: nvurisrcbin -> nvvideoconvert -> nvdsosd -> nveglglessink

from pyservicemaker import Pipeline, Flow
from multiprocessing import Process

VIDEO = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"

def run():
    # Gst.Pipeline.new("psm-main1")
    pipeline = Pipeline("psm-main1")
    (
        Flow(pipeline)
        # 1 nvurisrcbin (demux + NVDEC). no mux for 1 source.
        .capture([VIDEO])
        # nvvideoconvert + nvdsosd + nveglglessink
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
