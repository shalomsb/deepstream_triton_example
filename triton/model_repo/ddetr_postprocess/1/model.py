import json
import numpy as np
import triton_python_backend_utils as pb_utils


def sigmoid(x):
    # e_x = np.exp(x - np.max(x))
    # return e_x / e_x.sum(axis=0)
    return 1 / (1 + np.exp(-x))

def box_cxcywh_to_xyxy(x):
    """Convert box from cxcywh to xyxy."""
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=-1)


def post_process(pred_logits, pred_boxes, target_sizes, num_select=100):
    """Perform the post-processing. Scale back the boxes to the original size.

    Args:
        pred_logits (np.ndarray): (B x NQ x 4) logit values from TRT engine.
        pred_boxes (np.ndarray): (B x NQ x 4) bbox values from TRT engine.
        target_sizes (np.ndarray): (B x 4) [w, h, w, h] containing original image dimension.
        num_select (int): Top-K proposals to choose from.

    Returns:
        labels (np.ndarray): (B x NS) class label of top num_select predictions.
        scores (np.ndarray): (B x NS) class probability of top num_select predictions.
        boxes (np.ndarray):  (B x NS x 4) scaled back bounding boxes of top num_select predictions.
    """
    # Sigmoid
    prob = sigmoid(pred_logits).reshape((pred_logits.shape[0], -1))

    # Get topk scores
    topk_indices = np.argsort(prob, axis=1)[:, ::-1][:, :num_select]

    scores = [per_batch_prob[ind] for per_batch_prob, ind in zip(prob, topk_indices)]
    scores = np.array(scores)

    # Get corresponding boxes
    topk_boxes = topk_indices // pred_logits.shape[2]
    # Get corresponding labels
    labels = topk_indices % pred_logits.shape[2]

    # Convert to x1, y1, x2, y2 format
    boxes = box_cxcywh_to_xyxy(pred_boxes)

    # Take corresponding topk boxes
    boxes = np.take_along_axis(boxes, np.repeat(np.expand_dims(topk_boxes, -1), 4, axis=-1), axis=1)

    # Scale back the bounding boxes to the original image size
    target_sizes = np.array(target_sizes)
    boxes = boxes * target_sizes[:, None, :]

    # Clamp bounding box coordinates
    for i, target_size in enumerate(target_sizes):
        w, h = target_size[0], target_size[1]
        boxes[i, :, 0::2] = np.clip(boxes[i, :, 0::2], 0.0, w)
        boxes[i, :, 1::2] = np.clip(boxes[i, :, 1::2], 0.0, h)
    boxes[:, :, 2:] -= boxes[:, :, :2]

    return labels, scores, boxes


class TritonPythonModel:
    def initialize(self, args):
        # Initialise Triton logger
        self.logger = pb_utils.Logger
        
        self.input_name1 = "res_logits"
        self.input_name2 = "res_boxes"
        
        self.output_name1 = "output_labels"
        self.output_name2 = "output_scores"
        self.output_name3 = "output_boxes"
        

        self.model_config = model_config = json.loads(args['model_config'])
        output1_config = pb_utils.get_output_config_by_name(model_config, self.output_name1)
        output2_config = pb_utils.get_output_config_by_name(model_config, self.output_name2)
        output3_config = pb_utils.get_output_config_by_name(model_config, self.output_name3)
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config['data_type'])
        self.output2_dtype = pb_utils.triton_string_to_numpy(output2_config['data_type'])
        self.output3_dtype = pb_utils.triton_string_to_numpy(output3_config['data_type'])


    def execute(self, requests):
        responses = []

        for request in requests:
            # self.logger.log_info(f"Got input image Inside Triton Python Model!")
            pred_logits = pb_utils.get_input_tensor_by_name(request, self.input_name1).as_numpy()
            pred_boxes = pb_utils.get_input_tensor_by_name(request, self.input_name2).as_numpy()
            
            # Scale boxes to preprocessed image size (960 width x 544 height)
            # NOT the original client size (1920x1080)
            target_sizes = [(960, 544, 960, 544)]
            labels_result, scores_result, boxes_result = post_process(pred_logits, pred_boxes, target_sizes)

            output1 = pb_utils.Tensor(self.output_name1, labels_result.astype(self.output1_dtype))
            output2 = pb_utils.Tensor(self.output_name2, scores_result.astype(self.output2_dtype))
            output3 = pb_utils.Tensor(self.output_name3, boxes_result.astype(self.output3_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output1, output2, output3])
            responses.append(inference_response)
            
        return responses

    def finalize(self):
        self.logger.log_info('Triton Python Model cleaning up...')