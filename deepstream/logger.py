import logging
import os

class CustomLogger(logging.Logger):
    """
    This custom logger class extends the standard `logging.Logger` class with options
    to configure logging to the terminal and/or a log file. It provides a flexible
    way to control logging behavior and format.
    """

    def __init__(self, name, output_file_directory=None, level=logging.DEBUG, log_to_terminal=True, log_to_file=False):
        super().__init__(name, level)
        self.log_to_terminal = log_to_terminal
        self.log_to_file = log_to_file
        self.output_file_path = os.path.join(output_file_directory, f"{self.name}.log") if output_file_directory else None
        self._setup_handlers()

    def _setup_handlers(self):
        """
        Configure logger handlers for terminal and file output.
        """

        # Create a console handler to log to the screen.
        if self.log_to_terminal:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%d_%m_%Y-%H_%M_%S")
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(self.level)
            self.addHandler(console_handler)
        # Create a file handler to log to a file.
        if self.log_to_file and self.output_file_path:
            file_handler = logging.FileHandler(self.output_file_path)
            file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%d_%m_%Y-%H_%M_%S")
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(self.level)
            self.addHandler(file_handler)
