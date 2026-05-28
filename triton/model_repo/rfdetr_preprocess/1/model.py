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
        self.net_h = 704
        self.net_w = 704

        # ImageNet normalization (broadcast on [3,1,1]).
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "input_images")
            img = from_dlpack(in_tensor.to_dlpack())          # [B, 3, H, W] uint8 cuda

            B, C, H, W = img.shape
            img = img.float().div_(255.0)

            # Letterbox: maintain aspect ratio + symmetric pad to 704x704.
            scale = min(self.net_w / W, self.net_h / H)
            new_w = int(round(W * scale))
            new_h = int(round(H * scale))
            resized = F.interpolate(img, size=(new_h, new_w),
                                    mode='bilinear', align_corners=False)

            pad_total_w = self.net_w - new_w
            pad_total_h = self.net_h - new_h
            pad_left = pad_total_w // 2
            pad_right = pad_total_w - pad_left
            pad_top = pad_total_h // 2
            pad_bottom = pad_total_h - pad_top
            padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

            normalized = (padded - self.mean) / self.std
            out = normalized.contiguous()

            out_tensor = pb_utils.Tensor.from_dlpack("preprocessed_image", to_dlpack(out))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        self.logger.log_info('rfdetr_preprocess cleaning up...')
