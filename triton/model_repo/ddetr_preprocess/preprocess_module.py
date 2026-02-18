import torch
import torch.nn as nn
import torch.nn.functional as F


class TAO_DDETRPreprocessModule(nn.Module):
    """
    TensorRT-optimized preprocessing module for TAO Deformable DETR
    """
    
    def __init__(self, target_height=544, target_width=960):
        super().__init__()
        self.target_height = target_height
        self.target_width = target_width
        
        # ImageNet normalization parameters for DDETR
        # Register as 1D tensors for flexible reshaping during forward pass
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]))
        
    def forward(self, input_images):                
        # Convert to float and normalize to [0, 1]
        processed_images = input_images.float() / 255.0
        
        # Resize all images to target size (matching original SuperClass implementation)
        resized_images = F.interpolate(
            processed_images,
            size=(self.target_height, self.target_width),
            mode='bilinear',  # Note: Original uses NEAREST for final seg_map resize, but bilinear for input preprocessing
            align_corners=False
        )  # [num_windows, 3, target_size, target_size]
        
        # Apply ImageNet normalization with proper broadcasting
        # Reshape normalization parameters for 4D tensor: [1, 3, 1, 1]
        mean_4d = self.mean.view(1, 3, 1, 1)
        std_4d = self.std.view(1, 3, 1, 1)
        normalized_windows = (resized_images - mean_4d) / std_4d

        return normalized_windows  # Return as float32
