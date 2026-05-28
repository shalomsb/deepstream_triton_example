from pathlib import Path

from ds_pipeline import AppConfig, parse_yaml


MAX_SOURCES = 8


def _auto_grid(n):
    """Return (rows, cols) tile grid for n sources, biased toward landscape."""
    if n <= 1:
        return (1, 1)
    if n <= 2:
        return (1, 2)
    if n <= 4:
        return (2, 2)
    if n <= 6:
        return (2, 3)
    return (2, 4)


class Config(AppConfig):
    def __init__(self, yaml_filename="configs/config.yaml"):
        super().__init__(__file__, yaml_filename=yaml_filename)

        self.sources = self.data["sources"]
        if not isinstance(self.sources, list) or not (1 <= len(self.sources) <= MAX_SOURCES):
            raise ValueError(
                f"'sources' must be a list of 1..{MAX_SOURCES} entries, got {self.sources!r}")

        n = len(self.sources)
        self.file_loop = self.get("file_source", "loop", default=True)

        self.tracker_config = self.resolve("tracker", "config_file")

        sm = self.data.get("streammux", {})
        self.streammux_width = sm.get("width", 1920)
        self.streammux_height = sm.get("height", 1080)
        self.streammux_batch_size = sm.get("batch_size", n)

        rows, cols = _auto_grid(n)
        tl = self.data.get("tiler", {})
        self.tiler_rows = tl.get("rows", rows)
        self.tiler_cols = tl.get("cols", cols)
        self.tiler_width = tl.get("width", self.streammux_width)
        self.tiler_height = tl.get("height", self.streammux_height)

        # pgie.config_file points to a YAML that gathers all per-model knobs
        # (which nvinferserver .txt to use, network dims, conf threshold,
        # whether the preprocess letterboxes, labels file).
        pgie_cfg_path = self.resolve("pgie", "config_file")
        pgie = parse_yaml(pgie_cfg_path)
        pgie_dir = Path(pgie_cfg_path).resolve().parent
        self.pgie_config = str((pgie_dir / pgie["infer_config"]).resolve())
        self.network_width = pgie.get("network_width", 640)
        self.network_height = pgie.get("network_height", 640)
        self.maintain_aspect_ratio = bool(pgie.get("maintain_aspect_ratio", False))
        self.conf_threshold = pgie.get("conf_threshold", 0.25)
        self.labels_file = pgie.get("labels_file", "/deepstream/labels.txt")
        # Back-compat aliases — pgie_src_probe still reads these names.
        self.preprocess_width = self.network_width
        self.preprocess_height = self.network_height
