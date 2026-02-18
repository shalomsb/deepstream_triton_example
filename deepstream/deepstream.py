#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
import os

sys.path.append('../')
import gi

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import math

from ds_callbacks import pgie_src_pad_buffer_probe, fps_tracker_probe
from ds_bins import create_source_bin
from ds_utils import configure_tracker
from utils import create_gst_element, is_platform_aarch64, bus_call
from logger import CustomLogger
from config import Config
def main(config: Config, logger: CustomLogger):

    
    sources = []
    sources.append(config.source)

    # Validate file:// sources exist before starting pipeline
    for source in sources:
        if source.startswith("file://"):
            file_path = source.replace("file://", "")
            if not os.path.exists(file_path):
                logger.error(f"File source does not exist: {file_path}")
                sys.exit(1)
            logger.info(f"File source found: {file_path}")

    number_sources = len(sources)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create Pipeline element
    pipeline = Gst.Pipeline()
    is_live = False

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = create_gst_element("nvstreammux", "Stream-muxer", logger)

    pipeline.add(streammux)
    for i in range(number_sources):
        uri_name = sources[i]
        # Detect live sources (RTSP streams, V4L2/USB cameras) for pipeline optimization
        if uri_name.find("rtsp://") == 0 or uri_name.find("v4l2://") == 0 or uri_name.find("/dev/video") == 0:
            is_live = True

        # Normalize v4l2:// URIs to /dev/video format for source bin compatibility
        if uri_name.startswith("v4l2://"):
            uri_name = uri_name.replace("v4l2://", "")

        source_bin = create_source_bin(i, uri_name, logger, config.file_loop)

        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        srcpad = source_bin.get_static_pad("src")
        srcpad.link(sinkpad)
    
    if is_live:
        streammux.set_property('live-source', 1)

    streammux.set_property('width', config.streammux_input_width)
    streammux.set_property('height', config.streammux_input_height)
    streammux.set_property('batch-size', number_sources)
    # Timeout (μs) to wait for full batch before pushing incomplete batch downstream
    streammux.set_property('batched-push-timeout', config.MUXER_BATCH_TIMEOUT_USEC)
    # Memory type: Jetson uses Type 4 (CUDA_UNIFIED) for zero-copy, x86 uses Type 0 (DEFAULT)
    if is_platform_aarch64():
        streammux.set_property("nvbuf-memory-type", 4)  # NVBUF_MEM_CUDA_UNIFIED for Jetson
    else:
        streammux.set_property("nvbuf-memory-type", 0)  # NVBUF_MEM_DEFAULT for x86

    pgie = create_gst_element("nvinferserver", "primary-inference", logger)

    pgie.set_property('config-file-path', config.pgie_config_file)
    # Inference interval: run detection every N frames (e.g., 0=every frame, 1=every other frame)
    pgie.set_property('interval', config.pgie_interval)
    pgie.set_property("batch-size", number_sources)

    tracker = create_gst_element("nvtracker", "tracker", logger)
    configure_tracker(tracker=tracker, tracker_config_file=config.tracker_config_file)

    nvvidconv1 = create_gst_element("nvvideoconvert", "convertor1", logger)
    # Memory type: Jetson uses Type 4 (CUDA_UNIFIED), x86 uses Type 0 (DEFAULT)
    if is_platform_aarch64():
        nvvidconv1.set_property("nvbuf-memory-type", 4)  # NVBUF_MEM_CUDA_UNIFIED for Jetson
    else:
        nvvidconv1.set_property("nvbuf-memory-type", 0)  # NVBUF_MEM_DEFAULT for x86

    # NVMM (NVIDIA Multimedia Memory) with RGBA format for OSD overlay compatibility
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = create_gst_element("capsfilter", "filter1", logger)
    filter1.set_property("caps", caps1)

    # Display mode - create tiler, nvvidconv, nvosd and display sink
    tiler = create_gst_element("nvmultistreamtiler", "nvtiler", logger)
    # Type 0 = NVBUF_MEM_DEFAULT: System memory for display output compatibility
    tiler.set_property("nvbuf-memory-type", 0)
    nvvidconv = create_gst_element("nvvideoconvert", "convertor", logger)
    # Type 0 for final conversion to system memory before OSD and display
    nvvidconv.set_property("nvbuf-memory-type", 0)
    nvosd = create_gst_element("nvdsosd", "onscreendisplay", logger)


    if is_platform_aarch64():
        sink = create_gst_element("nv3dsink", "nv3d-sink", logger)
    else:
        sink = create_gst_element("nveglglessink", "nvvideo-renderer", logger)

    # Configure tiler for display
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", config.tiled_output_width)
    tiler.set_property("height", config.tiled_output_height)

    # Display sink properties: optimized for maximum throughput over real-time playback
    sink.set_property("sync", False)   # No clock sync - processes frames as fast as GPU can handle
    sink.set_property("qos", False)    # Disables frame dropping for quality-of-service management
    sink.set_property("async", False)  # Synchronous processing reduces pipeline latency
    
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(sink)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)

    # Link pipeline: streammux -> pgie -> tracker -> nvvidconv1
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv1)

    nvvidconv1.link(filter1)

    # Display: filter1 -> tiler -> nvvidconv -> nvosd -> sink
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # create an event loop and feed gstreamer bus messages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pgie_src_pad = pgie.get_static_pad("src")
    if pgie_src_pad:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, (config, logger))

    # Add FPS tracking probe
    tiler_sink_pad = tiler.get_static_pad("sink")
    if tiler_sink_pad is not None:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, fps_tracker_probe, (config, logger))
    else:
        logger.error("Unable to get tiler sink pad")

    # List the sources
    logger.info("Now playing...")
    for i, source in enumerate(sources):
        logger.info(f"{i}: {source}")

    logger.info("Starting pipeline")
    # start play back and listen to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    logger.info("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
