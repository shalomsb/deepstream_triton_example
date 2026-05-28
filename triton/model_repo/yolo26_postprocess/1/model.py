import json

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args['model_config'])
        self.device = torch.device(f"cuda:{args['model_instance_device_id']}")

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "raw_output")

            # raw: [B, 300, 6] on cuda, fp32 -- (x1, y1, x2, y2, conf, cls)
            raw = from_dlpack(in_tensor.to_dlpack())

            labels = raw[:, :, 5].to(torch.int64).contiguous()
            scores = raw[:, :, 4].contiguous()
            x1 = raw[:, :, 0]
            y1 = raw[:, :, 1]
            x2 = raw[:, :, 2]
            y2 = raw[:, :, 3]
            boxes = torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1).contiguous()

            out_labels = pb_utils.Tensor.from_dlpack("output_labels", to_dlpack(labels))
            out_scores = pb_utils.Tensor.from_dlpack("output_scores", to_dlpack(scores))
            out_boxes = pb_utils.Tensor.from_dlpack("output_boxes", to_dlpack(boxes))

            responses.append(pb_utils.InferenceResponse(
                output_tensors=[out_labels, out_scores, out_boxes]))

        return responses

    def finalize(self):
        self.logger.log_info('yolo26_postprocess cleaning up...')
