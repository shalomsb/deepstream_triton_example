import json
import numpy as np
import cv2
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args['model_config'])

        out_cfg = pb_utils.get_output_config_by_name(self.model_config, "preprocessed_image")
        self.output_dtype = pb_utils.triton_string_to_numpy(out_cfg['data_type'])

        self.target_h = 640
        self.target_w = 640

    def execute(self, requests):
        responses = []
        for request in requests:
            # Input: [1, 3, H, W] (batch dim from ensemble)
            img = pb_utils.get_input_tensor_by_name(request, "input_images").as_numpy()
            batch_size = img.shape[0]

            results = []
            for b in range(batch_size):
                frame = img[b]  # [3, H, W]
                # CHW -> HWC for cv2.resize
                frame_hwc = frame.transpose(1, 2, 0)
                resized = cv2.resize(frame_hwc, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
                # Normalize to [0, 1] and back to CHW
                out = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
                results.append(out)

            # Preserve batch dim: [B, 3, 640, 640]
            result = np.stack(results).astype(self.output_dtype)

            out_tensor = pb_utils.Tensor("preprocessed_image", result)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        self.logger.log_info('yolo26_preprocess cleaning up...')
