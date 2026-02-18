from utils import create_gst_element, is_platform_aarch64
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import sys

def cb_newpad(decodebin, decoder_src_pad, data):
    """Callback for when decodebin creates a new pad."""
    source_bin, logger = data
    logger.info("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin_pad = source_bin
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not audio.
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin_pad.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    """Callback for when decodebin adds a child element."""
    _, logger = user_data
    logger.info(f"Decodebin child added: {name}\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    # Only set CUDA unified memory on discrete GPUs (not Jetson/aarch64)
    if not is_platform_aarch64():
        # Use CUDA unified memory in the pipeline so frames can be easily accessed on CPU in Python.
        # 0: NVBUF_MEM_CUDA_DEVICE, 1: NVBUF_MEM_CUDA_PINNED, 2: NVBUF_MEM_CUDA_UNIFIED
        # Don't use direct macro here like NVBUF_MEM_CUDA_UNIFIED since nvv4l2decoder uses a
        # different enum internally
        Object.set_property("cudadec-memtype", 2)
        logger.info("Setting decoder to use CUDA unified memory")

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') is not None:
            Object.set_property("drop-on-latency", True)


def create_usbcam_source_bin(name, device, logger):
    """Create a source bin for USB camera input."""
    bin_name = f"{name}-usbcam-bin"
    new_usbcam_source_bin = Gst.Bin.new(bin_name)
    source = create_gst_element("v4l2src", f"usbcam-source-{name}", logger)
    caps_v4l2src = create_gst_element("capsfilter", f"v4l2src_caps_{name}", logger)
    vidconvsrc = create_gst_element("videoconvert", f"convertor_src_{name}", logger)
    nvvidconvsrc = create_gst_element("nvvideoconvert", f"nvconvertor_src_{name}", logger)
    caps_vidconvsrc = create_gst_element("capsfilter", f"nvmm_caps_{name}", logger)

    # Add elements into the bin
    new_usbcam_source_bin.add(source)
    new_usbcam_source_bin.add(caps_v4l2src)
    new_usbcam_source_bin.add(vidconvsrc)
    new_usbcam_source_bin.add(nvvidconvsrc)
    new_usbcam_source_bin.add(caps_vidconvsrc)

    # Linking the elements in the bin
    source.link(caps_v4l2src)
    caps_v4l2src.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(caps_vidconvsrc)

    # Set the properties of the elements in the bin
    source.set_property('device', device)
    caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))

    # Create src pad for the bin using the last element added to the bin
    srcpad = caps_vidconvsrc.get_static_pad("src")
    # Wrap the pad with GhostPad for getting a proxy pad of the entire bin
    ghost_src = Gst.GhostPad.new("src", srcpad)
    new_usbcam_source_bin.add_pad(ghost_src)

    return new_usbcam_source_bin


def cb_newpad_nvurisrcbin(decodebin, decoder_src_pad, data):
    """Callback for when nvurisrcbin creates a new pad."""
    source_bin, logger, uri, file_loop = data
    logger.info("nvurisrcbin pad-added callback\n")

    # Get the source bin ghost pad and set its target to the new pad
    bin_ghost_pad = source_bin.get_static_pad("src")
    if not bin_ghost_pad.set_target(decoder_src_pad):
        logger.error("Failed to link nvurisrcbin src pad to source bin ghost pad\n")
    
        # Enable file-loop for file:// URIs if configured
    if uri.startswith("file://") and file_loop:
        decodebin.set_property("file-loop", True)
        logger.info(f"File loop enabled for: {uri}")
    elif uri.startswith("file://") and not file_loop:
        decodebin.set_property("file-loop", False)
        logger.info(f"File loop disabled for: {uri}")


def create_nvurisrcbin_bin(name, uri, logger, file_loop=True):
    """Create a source bin using nvurisrcbin

    Args:
        name: Source name
        uri: Source URI (file:// or rtsp://)
        logger: Logger instance
        file_loop: Enable looping for file:// sources (default: True)
    """
    logger.info("Creating source bin with nvurisrcbin")

    bin_name = f"source-bin-{name}"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        logger.error("Unable to create source bin\n")
        return None

    # Use nvurisrcbin - better performance for DeepStream
    uri_src_bin = create_gst_element("nvurisrcbin", f"uri-src-bin-{name}", logger)
    if not uri_src_bin:
        logger.error("Unable to create nvurisrcbin\n")
        return None

    uri_src_bin.set_property("uri", uri)

    # Connect to pad-added signal - nvurisrcbin creates pad dynamically
    uri_src_bin.connect("pad-added", cb_newpad_nvurisrcbin, (nbin, logger, uri, file_loop))

    Gst.Bin.add(nbin, uri_src_bin)

    # Create ghost pad with no target (will be set in callback)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        logger.error("Failed to add ghost pad in source bin\n")
        return None

    return nbin


def create_uridecode_bin(name, uri, logger):
    """Create a source bin for URI-based input (files, RTSP streams, etc.)."""
    logger.info("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the pipeline
    bin_name = f"source-bin-{name}"
    logger.info(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        logger.error(" Unable to create source bin \n")
        return None

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = create_gst_element("uridecodebin", f"uri-decode-bin-{name}", logger)
    if not uri_decode_bin:
        logger.error(" Unable to create uri decode bin \n")
        return None

    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)

    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has been created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, (nbin, logger))
    uri_decode_bin.connect("child-added", decodebin_child_added, (nbin, logger))

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        logger.error(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def create_source_bin(index, uri, logger, file_loop=True):
    """Create a source bin for reading from a URI or USB camera.

    Args:
        index: Source index
        uri: Source URI or device path
        logger: Logger instance
        file_loop: Enable looping for file:// sources (default: True)
    """
    name = f"input_{index}"

    # Check if URI is a USB camera device
    if uri.startswith("/dev/video"):
        return create_usbcam_source_bin(name, uri, logger)
    else:
        return create_nvurisrcbin_bin(name, uri, logger, file_loop)
