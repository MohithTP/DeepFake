from src.models.attention import SpatialAttention
import torch.nn.functional as F

class SpatialStream(nn.Module):
    """
    Spatial Stream using XceptionNet backbone + Spatial Attention Module (SAM).
    Extracts features from RGB patches and highlights forensic artifacts.
    """
    def __init__(self, pretrained=True):
        super(SpatialStream, self).__init__()
        
        # 1. Backbone (Xception)
        # We need spatial features, not pooled vector.
        try:
            self.backbone = timm.create_model('legacy_xception', pretrained=pretrained, features_only=True)
            # legacy_xception features_only=True returns a list. 
            # Or we just use standard and call forward_features check.
            # detailed check: timm models usually support forward_features.
            self.backbone = timm.create_model('legacy_xception', pretrained=pretrained, num_classes=0, global_pool='')
            self.feature_dim = 2048
        except:
            print("Warning: Xception not found, falling back to ResNet50 for Spatial Stream")
            self.backbone = timm.create_model('resnet50', pretrained=pretrained, num_classes=0, global_pool='')
            self.feature_dim = 2048
            
        # 2. Forensic Focus (SAM)
        self.sam = SpatialAttention(kernel_size=7)
        
        # 3. Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        x: (B*9, 3, 256, 256)
        """
        # 1. Extract Spatial Features (B, 2048, H, W)
        features = self.backbone(x) 
        
        # 2. Apply Attention
        refined_features = self.sam(features)
        
        # 3. Global Pooling (B, 2048)
        pooled = self.gap(refined_features).flatten(1)
        
        return pooled
