# Graceful run harness for the psm learning ladder.
#
# WHY this exists:
#   Flow(...)() blocks in a C++ wait() that does NOT react to Ctrl-C, so we run
#   the pipeline in a child process (SIGINT then reaches Python in the parent).
#   But terminate() = SIGTERM = hard kill -> nvinferserver never unloads Triton
#   -> "Non-graceful termination detected" from the pb_stub backends.
#
#   Instead, on Ctrl-C we ask the child to STOP THE PIPELINE (PLAYING -> NULL).
#   That unloads Triton cleanly and lets wait() return so the child exits 0.

import signal
import threading
import multiprocessing


def _child(build, stop_event):
    # Ctrl-C hits the whole process group, so the child gets SIGINT too. Ignore it
    # here -- ONLY the parent handles Ctrl-C and drives the graceful stop via the
    # event below. (Otherwise the child's wait() ALSO raises KeyboardInterrupt and
    # dumps a duplicate traceback after the clean shutdown already ran.)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # build() returns a Flow but does NOT activate it (no trailing ()).
    pipeline = build().pipeline
    # The main thread is about to block in wait(). A daemon watcher thread waits
    # for the stop signal and calls stop() from OUTSIDE that blocked call
    # (GStreamer state changes are thread-safe -> stop() from another thread is ok).
    def watcher():
        stop_event.wait()
        pipeline.stop()             # PLAYING -> NULL: Triton unloads gracefully
    threading.Thread(target=watcher, daemon=True).start()
    pipeline.start().wait()         # blocks until EOS or stop()


def run(build):
    """Run a Flow-building function with graceful Ctrl-C + clean Triton unload.
       `build` must RETURN a Flow (don't call it with ())."""
    stop_event = multiprocessing.Event()
    p = multiprocessing.Process(target=_child, args=(build, stop_event))
    p.start()
    try:
        p.join()                    # parent blocks in Python -> Ctrl-C works
    except KeyboardInterrupt:
        stop_event.set()            # ask child to stop the pipeline gracefully
        p.join(timeout=10)          # give Triton time to unload
        if p.is_alive():
            p.terminate()           # fallback hard kill only if it hangs
