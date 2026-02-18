import os
from constants import Constants
from utils import read_yaml_file

class Config:
    def __init__(self, config_file, project_directory, program_start_time):
        self.project_directory = project_directory
        self.program_start_time = program_start_time
        self.config_directory = os.path.join(project_directory, Constants.CONFIG_DIRECTORY)
        self.config_file = os.path.join(self.config_directory, config_file)
        config_data = read_yaml_file(self.config_file)

        self.streammux_input_width = config_data["streammux"]["width"]
        self.streammux_input_height = config_data["streammux"]["height"]
        self.MUXER_BATCH_TIMEOUT_USEC = config_data["streammux"]["MUXER_BATCH_TIMEOUT_USEC"]

        self.pgie_config_file = os.path.join(self.project_directory, Constants.CONFIG_DIRECTORY, config_data["pgie"]["config_file"])
        self.pgie_interval = config_data["pgie"]["interval"]
        self.confidence_threshold = config_data["pgie"]["confidence_threshold"]

        # Select tracker configuration based on use_dcf_tracker setting
        use_dcf_tracker = config_data["tracker"]["use_dcf_tracker"]
        tracker_file = "tracker_config_dcf.txt" if use_dcf_tracker else "tracker_config_iou.txt"
        self.tracker_config_file = os.path.join(self.project_directory, Constants.CONFIG_DIRECTORY, tracker_file)
        self.use_dcf_tracker = use_dcf_tracker

        self.tiled_output_width = config_data["display"]["tiled_output_width"]
        self.tiled_output_height = config_data["display"]["tiled_output_height"]
        self.fps_print_interval = config_data["display"]["fps_print_interval"]

        self.source = config_data["source"]
        self.file_loop = config_data["file_source"]["loop"]