from datetime import datetime, timezone
import os

import deepstream
from config import Config
from constants import Constants
from logger import CustomLogger


def main():
    # Store application top-level arguments
    program_start_time = datetime.now(timezone.utc)
    current_workdir = os.path.dirname(os.path.abspath(__file__))

    logger = CustomLogger("deepstream_app", log_to_terminal=True, log_to_file=False)

    # Initialize App Config
    app_config = Config(config_file=Constants.DEFAULT_CONFIG_FILE,
                                    project_directory=current_workdir,
                                    program_start_time=program_start_time)

    # Run the DeepStream application
    deepstream.main(config=app_config, logger=logger)

    logger.info("Main App Exiting")

if __name__ == "__main__":
    main()
