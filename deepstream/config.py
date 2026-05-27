from ds_pipeline import AppConfig


class Config(AppConfig):
    def __init__(self, yaml_filename="configs/config.yaml"):
        super().__init__(__file__, yaml_filename=yaml_filename)

        self.source = self.data["source"]
        self.file_loop = self.get("file_source", "loop", default=True)

        self.pgie_config = self.resolve("pgie", "config_file")
        self.tracker_config = self.resolve("tracker", "config_file")

        sm = self.data.get("streammux", {})
        self.streammux_width = sm.get("width", 1920)
        self.streammux_height = sm.get("height", 1080)
        self.streammux_batch_size = sm.get("batch_size", 1)

        pgie = self.data.get("pgie", {})
        self.preprocess_width = pgie.get("preprocess_width", 640)
        self.preprocess_height = pgie.get("preprocess_height", 640)
        self.conf_threshold = pgie.get("conf_threshold", 0.25)
        self.labels_file = pgie.get("labels_file", "/deepstream/labels.txt")
