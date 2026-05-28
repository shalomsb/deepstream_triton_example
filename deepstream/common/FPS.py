import time
from threading import Lock


class _StreamFPS:
    def __init__(self):
        self.start = time.time()
        self.count = 0
        self.first = True

    def tick(self):
        if self.first:
            self.start = time.time()
            self.first = False
        else:
            self.count += 1

    def read_and_reset(self):
        now = time.time()
        elapsed = now - self.start
        fps = self.count / elapsed if elapsed > 0 else 0.0
        self.count = 0
        self.start = now
        return round(fps, 2)


class PERF_DATA:
    """Per-stream FPS counter. Call update_fps(stream_index) from a probe and
    register perf_print_callback with GLib.timeout_add to print at an interval."""

    def __init__(self, num_streams):
        self._lock = Lock()
        self._streams = {f"stream{i}": _StreamFPS() for i in range(num_streams)}

    def update_fps(self, stream_index):
        with self._lock:
            s = self._streams.get(stream_index)
            if s is not None:
                s.tick()

    def perf_print_callback(self):
        with self._lock:
            snap = {k: s.read_and_reset() for k, s in self._streams.items()}
        print(f"**PERF: {snap}")
        return True
