import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) for DeepScan 2.0
    
    Purpose:
    Applies 'Forensic Focus' by highlighting suspicious regions in the feature map.
    Instead of treating all pixels equally, it learns which areas (e.g., eyes, mouth)
    contain the most relevant information for forgery detection.
    
    Mechanism:
    1. Channel Pooling: MaxPool and AvgPool along the channel axis.
    2. Concatenation: Stacks the two pools (2 channels).
    3. Convolution: 7x7 Conv to crush to 1 channel.
    4. Activation: Sigmoid to generate an attention map (0-1).
    5. Output: Input Features * Attention Map.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        
        # 1. Channel Pooling
        avg_out = torch.mean(x, dim=1, keepdim=True) # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, H, W)
        
        # 2. Concat
        x_cat = torch.cat([avg_out, max_out], dim=1) # (B, 2, H, W)
        
        # 3. Conv + Sigmoid
        attention_map = self.conv1(x_cat) # (B, 1, H, W)
        attention_map = self.sigmoid(attention_map)
        
        # 4. Refine Features
        return x * attention_map
