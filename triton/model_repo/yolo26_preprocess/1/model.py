import json

import torch
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args['model_config'])

        self.device = torch.device(f"cuda:{args['model_instance_device_id']}")
        self.target_h = 640
        self.target_w = 640

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "input_images")

            # Zero-copy: Triton GPU buffer -> PyTorch CUDA tensor.
            # Input: UINT8 [B, 3, H, W] on cuda.
            img = from_dlpack(in_tensor.to_dlpack())

            # Resize + normalize on GPU.
            img = img.float().div_(255.0)
            out = F.interpolate(img, size=(self.target_h, self.target_w),
                                mode='bilinear', align_corners=False)
            out = out.contiguous()

            # Zero-copy: PyTorch CUDA tensor -> Triton GPU buffer.
            out_tensor = pb_utils.Tensor.from_dlpack("preprocessed_image", to_dlpack(out))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        self.logger.log_info('yolo26_preprocess cleaning up...')
