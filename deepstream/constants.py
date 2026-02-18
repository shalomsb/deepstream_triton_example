class Constants:
    CONFIG_DIRECTORY = "configs"
    DEFAULT_CONFIG_FILE = "default_config.yaml"

    # DeepStream inference constants (must match config files)
    PGIE_UNIQUE_ID = 1

    # Triton preprocessing model dimensions (must match preprocessing model output)
    PREPROCESS_WIDTH = 960
    PREPROCESS_HEIGHT = 544