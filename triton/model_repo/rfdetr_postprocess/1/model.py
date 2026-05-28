import json

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

import triton_python_backend_utils as pb_utils


# Must match the rfdetr_large input spatial dims (set in its config.pbtxt).
NET_H = 704
NET_W = 704
NUM_SELECT = 300


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args['model_config'])
        self.device = torch.device(f"cuda:{args['model_instance_device_id']}")

    def execute(self, requests):
        responses = []
        for request in requests:
            boxes_in = from_dlpack(
                pb_utils.get_input_tensor_by_name(request, "dets").to_dlpack())
            logits = from_dlpack(
                pb_utils.get_input_tensor_by_name(request, "labels").to_dlpack())

            # boxes_in: [B, N, 4] cxcywh normalized [0,1]
            # logits:   [B, N, C] raw -- sigmoid per class, not softmax
            B, N, C = logits.shape

            prob = torch.sigmoid(logits)                                # [B, N, C]
            topk_scores, topk_idx = torch.topk(prob.view(B, -1),
                                               NUM_SELECT, dim=1)        # [B, K]
            query_idx = topk_idx // C
            labels = (topk_idx % C).to(torch.int64)

            # gather selected boxes per batch
            gather_idx = query_idx.unsqueeze(-1).expand(-1, -1, 4)
            selected = torch.gather(boxes_in, 1, gather_idx)             # [B, K, 4] cxcywh

            cx = selected[..., 0]
            cy = selected[..., 1]
            bw = selected[..., 2]
            bh = selected[..., 3]
            left = (cx - bw / 2.0) * NET_W
            top = (cy - bh / 2.0) * NET_H
            width = bw * NET_W
            height = bh * NET_H
            boxes_out = torch.stack([left, top, width, height], dim=-1).contiguous()

            out_labels = pb_utils.Tensor.from_dlpack(
                "ensemble_labels", to_dlpack(labels.contiguous()))
            out_scores = pb_utils.Tensor.from_dlpack(
                "ensemble_scores", to_dlpack(topk_scores.contiguous()))
            out_boxes = pb_utils.Tensor.from_dlpack(
                "ensemble_boxes", to_dlpack(boxes_out))

            responses.append(pb_utils.InferenceResponse(
                output_tensors=[out_labels, out_scores, out_boxes]))

        return responses

    def finalize(self):
        self.logger.log_info('rfdetr_postprocess cleaning up...')
