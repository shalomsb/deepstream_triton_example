import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args['model_config'])

        out_labels_cfg = pb_utils.get_output_config_by_name(self.model_config, "output_labels")
        out_scores_cfg = pb_utils.get_output_config_by_name(self.model_config, "output_scores")
        out_boxes_cfg = pb_utils.get_output_config_by_name(self.model_config, "output_boxes")

        self.labels_dtype = pb_utils.triton_string_to_numpy(out_labels_cfg['data_type'])
        self.scores_dtype = pb_utils.triton_string_to_numpy(out_scores_cfg['data_type'])
        self.boxes_dtype = pb_utils.triton_string_to_numpy(out_boxes_cfg['data_type'])

    def execute(self, requests):
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "raw_output").as_numpy()

            # [1, 300, 6] -> [300, 6]
            raw = raw[0]

            # (x1, y1, x2, y2, conf, cls)
            labels = raw[:, 5].astype(self.labels_dtype)
            scores = raw[:, 4].astype(self.scores_dtype)
            x1 = raw[:, 0]
            y1 = raw[:, 1]
            x2 = raw[:, 2]
            y2 = raw[:, 3]
            boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1).astype(self.boxes_dtype)

            # Preserve batch dim for ensemble compatibility
            out_labels = pb_utils.Tensor("output_labels", labels[np.newaxis])
            out_scores = pb_utils.Tensor("output_scores", scores[np.newaxis])
            out_boxes = pb_utils.Tensor("output_boxes", boxes[np.newaxis])

            responses.append(pb_utils.InferenceResponse(output_tensors=[out_labels, out_scores, out_boxes]))

        return responses

    def finalize(self):
        self.logger.log_info('yolo26_postprocess cleaning up...')
