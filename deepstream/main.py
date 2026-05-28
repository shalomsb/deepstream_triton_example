#!/usr/bin/env python3

import sys

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.platform_info import PlatformInfo
from common.FPS import PERF_DATA
from ds_pipeline import (
    Logger,
    create_pipeline, create_source_bin, create_streammux,
    create_pgie_inferserver, create_tracker, create_tiler,
    create_nvvidconv, create_nvosd, create_sink, create_queue,
    run_pipeline, link_chain,
)
from config import Config
from callbacks import pgie_src_probe, osd_probe


def main():
    config = Config()
    logger = Logger("deepstream-yolo26x")
    platform_info = PlatformInfo()
    Gst.init(None)

    pipeline = create_pipeline("deepstream-yolo26x", logger)

    source_bins = [
        create_source_bin(i, uri, logger, file_loop=config.file_loop)
        for i, uri in enumerate(config.sources)
    ]
    streammux = create_streammux("yolo26x", batch_size=config.streammux_batch_size,
                                 width=config.streammux_width,
                                 height=config.streammux_height, logger=logger)
    pgie = create_pgie_inferserver("yolo26x", config.pgie_config, logger)
    pgie.set_property("batch-size", len(config.sources))
    tracker = create_tracker("yolo26x", config.tracker_config, logger)
    tiler = create_tiler("yolo26x", config.tiler_rows, config.tiler_cols,
                         config.tiler_width, config.tiler_height, platform_info, logger)
    nvvidconv = create_nvvidconv("yolo26x", logger)
    nvosd = create_nvosd("yolo26x", logger)
    sink = create_sink("yolo26x", platform_info, logger)
    sink.set_property("qos", 0)
    sink.set_property("sync", 0)

    queues = [create_queue(f"q{i}", logger) for i in range(5)]

    for el in source_bins + [streammux, pgie, tracker, tiler, nvvidconv, nvosd, sink] + queues:
        pipeline.add(el)

    for i, source_bin in enumerate(source_bins):
        srcpad = source_bin.get_static_pad("src")
        sinkpad = streammux.request_pad_simple(f"sink_{i}")
        srcpad.link(sinkpad)

    link_chain(streammux, queues[0], pgie, queues[1], tracker, queues[2],
               tiler, queues[3], nvvidconv, queues[4], nvosd, sink)

    perf_data = PERF_DATA(len(config.sources))
    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, pgie_src_probe, (config, perf_data))
    nvosd.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, osd_probe, config)
    GLib.timeout_add(5000, perf_data.perf_print_callback)

    run_pipeline(pipeline, logger)


if __name__ == '__main__':
    sys.exit(main())
