import numpy as np
import ctypes
import pyds
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from ds_pipeline import (
    get_batch_meta, iter_frames, iter_output_tensors,
    iter_objects, add_obj_meta,
    add_osd_text, set_obj_label,
)
from config import Config

LAYER_DTYPES = {
    "ensemble_labels": ctypes.c_int64,
    "ensemble_scores": ctypes.c_float,
    "ensemble_boxes": ctypes.c_float,
}

_labels = None


def _load_labels(path):
    global _labels
    if _labels is None:
        with open(path, 'r') as f:
            _labels = [line.strip() for line in f if line.strip()]
    return _labels


def _parse_ensemble_output(tensor_meta, frame_width, frame_height,
                           preprocess_width, preprocess_height, conf_threshold):
    layers = {}
    for i in range(tensor_meta.num_output_layers):
        layer_info = pyds.get_nvds_LayerInfo(tensor_meta, i)
        name = layer_info.layerName
        dtype = LAYER_DTYPES.get(name)
        if dtype is None:
            continue
        shape = tuple(np.trim_zeros(layer_info.inferDims.d, 'b'))
        if not shape:
            continue
        ptr = ctypes.cast(pyds.get_ptr(layer_info.buffer), ctypes.POINTER(dtype))
        layers[name] = np.ctypeslib.as_array(ptr, shape=shape)

    labels = layers.get("ensemble_labels")
    scores = layers.get("ensemble_scores")
    boxes = layers.get("ensemble_boxes")

    if labels is None or scores is None or boxes is None:
        return []

    scale_x = frame_width / preprocess_width
    scale_y = frame_height / preprocess_height

    mask = scores >= conf_threshold
    labels = labels[mask]
    scores = scores[mask]
    boxes = boxes[mask]

    detections = []
    for idx in range(len(scores)):
        x, y, w, h = boxes[idx]
        detections.append({
            'class_id': int(labels[idx]),
            'confidence': float(scores[idx]),
            'left': float(x) * scale_x,
            'top': float(y) * scale_y,
            'width': float(w) * scale_x,
            'height': float(h) * scale_y,
        })
    return detections


def pgie_src_probe(pad, info, u_data: Config):
    config = u_data
    batch_meta, _ = get_batch_meta(info)
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    for frame in iter_frames(batch_meta):
        for tensor_meta in iter_output_tensors(frame):
            detections = _parse_ensemble_output(
                tensor_meta,
                config.streammux_width, config.streammux_height,
                config.preprocess_width, config.preprocess_height,
                config.conf_threshold,
            )
            for det in detections:
                add_obj_meta(
                    batch_meta, frame,
                    det['left'], det['top'], det['width'], det['height'],
                    class_id=det['class_id'],
                    confidence=det['confidence'],
                    unique_id=tensor_meta.unique_id,
                )
            frame.bInferDone = True

    return Gst.PadProbeReturn.OK


def osd_probe(pad, info, u_data: Config):
    config = u_data
    labels = _load_labels(config.labels_file)
    batch_meta, _ = get_batch_meta(info)
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    for frame in iter_frames(batch_meta):
        for obj in iter_objects(frame):
            if obj.class_id < len(labels):
                set_obj_label(obj, f"{labels[obj.class_id]} {obj.confidence:.2f} ID:{obj.object_id}")
        text = f"Frame={frame.frame_num} Objects={frame.num_obj_meta}"
        add_osd_text(batch_meta, frame, text)
        print(text)

    return Gst.PadProbeReturn.OK
