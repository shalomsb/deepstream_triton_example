import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import sys

import os
import platform
import yaml
import traceback

def create_gst_element(factory, plugin_name, logger):
    """Create a GStreamer element and handle errors."""
    p = Gst.ElementFactory.make(factory, plugin_name)
    if not p:
        err_msg = f"Unable to create element with factory name: {factory}"
        logger.error(err_msg)
        raise Exception(err_msg)
    return p

def is_platform_aarch64():
    """Check if platform is aarch64 (NVIDIA Jetson) using uname."""
    return platform.uname()[4] == 'aarch64'

def try_make_directory(dir_path, exist_ok=True):
    """Helper function to try make directory safely."""
    try:
        os.makedirs(dir_path, exist_ok=exist_ok)
        return True
    except OSError:
        return False
    

def read_yaml_file(file_path):
    try:
         # Read config file
        with open(file_path, 'r') as f:
            config_file = yaml.safe_load(f)
        return config_file
    except Exception as e:
        print(f"unable to open config file {file_path}")
        traceback.print_exc()
        raise e

def read_labels_file(file_path):
    """
    Read labels from a text file with one label per line.
    Strips whitespace and newlines to ensure correct class retrieval.

    Args:
        file_path: Path to labels.txt file

    Returns:
        List of label strings with proper alignment (no \\n or trailing spaces)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read all lines, strip whitespace/newlines, and filter empty lines
            labels = [line.strip() for line in f.readlines()]
            # Ensure no empty strings (in case of trailing newlines)
            labels = [label if label else "N/A" for label in labels]
        return labels
    except Exception as e:
        print(f"Unable to open labels file {file_path}")
        traceback.print_exc()
        raise e
    
def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True
