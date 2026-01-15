import torch
import torch.nn as nn
import timm

class SpatialStream(nn.Module):
    """
    Spatial Stream using XceptionNet (or Xception-like) backbone.
    Extracts features from RGB patches.
    """
    def __init__(self, pretrained=True):
        super(SpatialStream, self).__init__()
        # Using timm's xception or legacy_xception
        # If xception is not available in timm factory easily, we can use efficientnet or resnet.
        # But report says Xception.
        # Check timm models: 'xception', 'legacy_xception'.
        # We will use 'xception' if available, otherwise 'tf_xception_65' or similar.
        # To be safe and robust, let's use 'xception' from timm.
        
        # We remove the classifier head (num_classes=0) to get features.
        # Standard Xception output is 2048 dims.
        try:
            self.backbone = timm.create_model('legacy_xception', pretrained=pretrained, num_classes=0)
            self.feature_dim = 2048
        except:
            # Fallback to ResNet50 if Xception not found in this specific timm version
            print("Warning: Xception not found, falling back to ResNet50 for Spatial Stream")
            self.backbone = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
            self.feature_dim = 2048

    def forward(self, x):
        """
        x: (B*9, 3, 256, 256) - Process patches in batch
        """
        features = self.backbone(x) # (B*9, 2048)
        return features
