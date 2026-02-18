# DeepStream & Gstreamer imports
import pyds
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import configparser

# General python imports
import ctypes
import numpy as np
import traceback
import time
from collections import deque

# ==========================================================================================
# ==========================================================================================
# ======================== DS METADATA & TRITON UTILS =======================================
# ==========================================================================================
# ==========================================================================================

UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF

class DetectedObject:
    """Represents a detected object from DDETR model"""
    def __init__(self, bbox, confidence, class_id, class_name):
        self.bbox = bbox  # [x, y, w, h]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    def __str__(self):
        return f"class: {self.class_name} ({self.class_id}) -- bbox: {self.bbox} -- confidence: {self.confidence:.2f}"

    def __repr__(self):
        return str(self)


def get_triton_layer_data(layer_info, data_type, logger):
    """Extract numpy array from Triton layer buffer"""
    result_array = np.array([])
    try:
        if data_type == "float32":
            actual_ctypes_dtype = ctypes.c_float
        elif data_type == "int64":
            actual_ctypes_dtype = ctypes.c_int64
        elif data_type == "uint8":
            actual_ctypes_dtype = ctypes.c_uint8
        else:
            logger.error(f"Unknown data type: {data_type}")
            return result_array

        client_ptr = ctypes.cast(pyds.get_ptr(layer_info.buffer), ctypes.POINTER(actual_ctypes_dtype))
        if client_ptr:
            result_array = np.ctypeslib.as_array(client_ptr, shape=(*np.trim_zeros(layer_info.inferDims.d, 'b'),))
        else:
            logger.error(f"Triton pointer is NULL for {layer_info.layerName}")
    except:
        logger.error(traceback.format_exc())

    return result_array


def add_object_data_to_ds_meta(batch_meta, frame_meta, detected_obj: DetectedObject,
                               unique_engine_id, orig_frame_width, orig_frame_height):
    """Add detected object to DeepStream metadata"""
    new_obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)

    # Extract bbox data from model
    box_x1, box_y1, box_w, box_h = detected_obj.bbox

    rect_params = new_obj_meta.rect_params
    # Clip box top-left to be inside frame min boundaries
    rect_params.left = max(0, int(box_x1))
    rect_params.top = max(0, int(box_y1))
    # Clip box width & height to keep box inside frame max boundaries
    rect_params.width = min(max(1, int(box_w)), orig_frame_width - rect_params.left)
    rect_params.height = min(max(1, int(box_h)), orig_frame_height - rect_params.top)

    # Store the object ID and confidence
    new_obj_meta.object_id = UNTRACKED_OBJECT_ID
    new_obj_meta.confidence = float(detected_obj.confidence)

    # Set border color based on class (you can customize this)
    rect_params.border_color.set(0, 1, 0, 1)  # Green
    rect_params.border_width = 3
    rect_params.has_bg_color = 0

    # Store which model engine provided the results
    new_obj_meta.unique_component_id = unique_engine_id

    # Class ID and label string
    new_obj_meta.class_id = int(detected_obj.class_id)
    new_obj_meta.obj_label = str(detected_obj.class_name)

    # Text parameters for label display
    txt_params = new_obj_meta.text_params
    if txt_params.display_text:
        pyds.free_buffer(txt_params.display_text)

    # Set offset for per-target text
    txt_params.x_offset = max(0, int(rect_params.left))
    txt_params.y_offset = max(0, int(rect_params.top) - 10)
    # Set target text
    txt_params.display_text = f"{new_obj_meta.obj_label}: {detected_obj.confidence:.2f}"

    # Text style
    txt_params.font_params.font_name = "Serif"
    txt_params.font_params.font_size = 10
    txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

    txt_params.set_bg_clr = 1
    txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

    # Finally, add the new object to DeepStream metadata
    pyds.nvds_add_obj_meta_to_frame(frame_meta, new_obj_meta, None)


# ==========================================================================================
# ==========================================================================================
# ======================== FPS TRACKING ====================================================
# ==========================================================================================
# ==========================================================================================

class StreamFPSTracker:
    """Simple FPS tracker with moving average over a time window"""
    def __init__(self, window_seconds=1.0):
        self.window_seconds = window_seconds
        self.frame_timestamps = deque()

    def add_frame(self):
        current_time = time.time()
        self.frame_timestamps.append(current_time)
        # Remove timestamps older than window
        cutoff_time = current_time - self.window_seconds
        while self.frame_timestamps and self.frame_timestamps[0] < cutoff_time:
            self.frame_timestamps.popleft()

    def get_fps(self):
        # Calculate actual time span between first and last frame in window
        if self.has_enough_data():
            time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if time_span > 0.0:
                # frames - 1 because we count intervals between frames
                return (len(self.frame_timestamps) - 1) / time_span
        return 0.0

    def has_enough_data(self):
        """Check if we have enough data for meaningful FPS calculation"""
        # Wait at until we have at-least two frames
        return len(self.frame_timestamps) >= 2
    

def configure_tracker(tracker, tracker_config_file):
     #Set properties of tracker
    config = configparser.ConfigParser()
    config.read(tracker_config_file)
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
