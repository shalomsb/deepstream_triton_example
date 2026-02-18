# DeepStream & Gstreamer imports
import pyds
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# General python imports
import numpy as np
import traceback
import time

# Application specific imports
from ds_utils import get_triton_layer_data, add_object_data_to_ds_meta, DetectedObject
from constants import Constants
from utils import read_labels_file
import os

# Load COCO class labels from labels.txt file
# Strips whitespace and newlines to ensure correct class retrieval
_current_dir = os.path.dirname(os.path.abspath(__file__))
_labels_file = os.path.join(_current_dir, "labels.txt")
COCO_CLASSES = read_labels_file(_labels_file)

# Global variables for FPS tracking
fps_trackers = {}
last_fps_print_time = 0


# ============================================================================
# DDETR MODEL OUTPUT PROCESSING
# ============================================================================

def get_ddetr_model_output(output_tensor_meta, logger=None):
    """Extract DDETR model outputs from Triton tensor metadata"""
    ddetr_results = {}

    # Extract DDETR results from triton output layers
    # Expected outputs: ensemble_labels, ensemble_scores, ensemble_boxes
    for i in range(output_tensor_meta.num_output_layers):
        layer_info = pyds.get_nvds_LayerInfo(output_tensor_meta, i)
        layer_name = layer_info.layerName

        if layer_name == "ensemble_labels":
            # INT64 tensor with shape [batch, num_objects]
            layer_data = get_triton_layer_data(layer_info, "int64", logger)
        elif layer_name == "ensemble_scores":
            # FP32 tensor with shape [batch, num_objects]
            layer_data = get_triton_layer_data(layer_info, "float32", logger)
        elif layer_name == "ensemble_boxes":
            # FP32 tensor with shape [batch, num_objects, 4]
            layer_data = get_triton_layer_data(layer_info, "float32", logger)
        else:
            logger.warning(f"Unknown layer: {layer_name}\n")
            layer_data = None

        ddetr_results[layer_name] = layer_data

    return ddetr_results


def parse_ddetr_results(ddetr_results, orig_frame_width, orig_frame_height, confidence_threshold, preprocess_width, preprocess_height, logger=None):
    """Parse DDETR results into list of DetectedObject instances

    Args:
        ddetr_results: Dictionary containing labels, scores, boxes from Triton
        orig_frame_width: Original frame width (e.g., 1920)
        orig_frame_height: Original frame height (e.g., 1080)
        confidence_threshold: Minimum confidence threshold for detections
        logger: Logger instance for logging messages
    """
    detected_objects = []

    # Extract outputs
    labels = ddetr_results.get("ensemble_labels")
    scores = ddetr_results.get("ensemble_scores")
    boxes = ddetr_results.get("ensemble_boxes")

    if labels is None or scores is None or boxes is None:
        logger.warning("Missing DDETR output tensors\n")
        return detected_objects

    # Handle batch dimension (usually batch_size=1)
    if len(labels.shape) > 1:
        labels = labels[0]
    if len(scores.shape) > 1:
        scores = scores[0]
    if len(boxes.shape) > 2:
        boxes = boxes[0]

    # Critical: Triton preprocessing resizes frames to configured dimensions for inference
    # Bounding boxes are in this coordinate space, must scale to original frame dimensions
    scale_x = orig_frame_width / preprocess_width
    scale_y = orig_frame_height / preprocess_height

    # Iterate through detections
    for label, score, box in zip(labels, scores, boxes):
        # Filter by confidence
        if score < confidence_threshold:
            continue

        # Get class name
        class_id = int(label)
        if 0 <= class_id < len(COCO_CLASSES):
            class_name = COCO_CLASSES[class_id]
        else:
            class_name = f"class_{class_id}"

        # Skip N/A classes
        if class_name == "N/A":
            continue

        # Scale box coordinates from 960x544 to actual frame size
        scaled_box = [
            float(box[0]) * scale_x,  # x
            float(box[1]) * scale_y,  # y
            float(box[2]) * scale_x,  # width
            float(box[3]) * scale_y   # height
        ]

        detected_obj = DetectedObject(
            bbox=scaled_box,
            confidence=float(score),
            class_id=class_id,
            class_name=class_name
        )
        detected_objects.append(detected_obj)

    return detected_objects



# ============================================================================
# PRIMARY INFERENCE PROBE
# ============================================================================

def pgie_src_pad_buffer_probe(pad, info, u_data):
    """
    Probe function to extract DDETR Triton inference results and add them to DeepStream metadata
    This runs after the inference server and adds object metadata to the frame

    Args:
        u_data: Tuple of (config, logger)
    """
    config, logger = u_data

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logger.error("Unable to get GstBuffer\n")
        return Gst.PadProbeReturn.OK

    # Get batch metadata
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    # Loop through all frames in batch
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Get frame dimensions from config
        orig_frame_width = config.tiled_output_width
        orig_frame_height = config.tiled_output_height

        # Search for tensor output metadata
        l_user = frame_meta.frame_user_meta_list
        detected_objects = []

        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            # Check if this is tensor output metadata from inference
            if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                # Mark frame as having inference results
                frame_meta.bInferDone = True

                # Get the tensor metadata
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                # Extract DDETR model results
                try:
                    ddetr_results = get_ddetr_model_output(tensor_meta, logger)
                    detected_objects = parse_ddetr_results(
                        ddetr_results,
                        orig_frame_width,
                        orig_frame_height,
                        confidence_threshold=config.confidence_threshold,
                        preprocess_width=Constants.PREPROCESS_WIDTH,
                        preprocess_height=Constants.PREPROCESS_HEIGHT,
                        logger=logger
                    )


                except Exception as e:
                    logger.error(f"Error parsing DDETR results: {e}\n")
                    logger.error(f"{traceback.format_exc()}\n")

            # Move to next user meta
            try:
                l_user = l_user.next
            except StopIteration:
                break

        # Add detected objects to DeepStream metadata
        if len(detected_objects) > 0:
            # Critical: Acquire lock before modifying batch metadata (thread-safe access)
            pyds.nvds_acquire_meta_lock(batch_meta)
            for detected_obj in detected_objects:
                add_object_data_to_ds_meta(
                    batch_meta,
                    frame_meta,
                    detected_obj,
                    Constants.PGIE_UNIQUE_ID,
                    orig_frame_width,
                    orig_frame_height
                )
            # Always release lock after metadata modifications
            pyds.nvds_release_meta_lock(batch_meta)

        # Move to next frame in batch
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ============================================================================
# FPS TRACKING PROBE
# ============================================================================

def fps_tracker_probe(pad, info, u_data):
    """Probe to track FPS for each stream

    Args:
        u_data: Tuple of (config, logger)
    """
    from ds_utils import StreamFPSTracker
    global fps_trackers, last_fps_print_time

    config, logger = u_data

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    current_time = time.time()

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        stream_id = frame_meta.pad_index

        # Initialize FPS tracker for this stream if it doesn't exist
        if stream_id not in fps_trackers:
            fps_trackers[stream_id] = StreamFPSTracker(window_seconds=1.0)

        # Add frame to tracker
        fps_trackers[stream_id].add_frame()


        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # Print FPS based on config interval
    if current_time - last_fps_print_time >= config.fps_print_interval:
        fps_lines = ["\n=== FPS Report ==="]
        for stream_id in sorted(fps_trackers.keys()):
            tracker = fps_trackers[stream_id]
            if tracker.has_enough_data():
                fps = tracker.get_fps()
                fps_lines.append(f"Stream {stream_id}: {fps:.2f} FPS")
        fps_lines.append("==================")
        logger.info("\n".join(fps_lines))
        last_fps_print_time = current_time


    return Gst.PadProbeReturn.OK
