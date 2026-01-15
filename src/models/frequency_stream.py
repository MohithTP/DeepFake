import torch
import torch.nn as nn
import timm
from src.utils.dct_ops import DCTTransform

class FrequencyStream(nn.Module):
    """
    Frequency Stream using ResNet50 backbone.
    Takes RGB, converts to DCT, then extracts features.
    """
    def __init__(self, pretrained=True):
        super(FrequencyStream, self).__init__()
        self.dct = DCTTransform()
        
        # ResNet50 takes 3 channel input by default. 
        # DCT output is same logic shape (3 channels if we apply per-channel).
        self.backbone = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
        self.feature_dim = 2048
        
    def forward(self, x):
        """
        x: (B*9, 3, 256, 256) RGB patches
        """
        # 1. Convert to Freq Domain
        x_dct = self.dct(x) # (B*9, 3, 256, 256)
        
        # 2. Extract Features
        features = self.backbone(x_dct) # (B*9, 2048)
        return features
