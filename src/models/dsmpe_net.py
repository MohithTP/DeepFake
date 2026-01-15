import torch
import torch.nn as nn
from src.models.patch_extractor import PatchGenerator
from src.models.spatial_stream import SpatialStream
from src.models.frequency_stream import FrequencyStream

class DSMPE_Net(nn.Module):
    """
    Dual-Stream Multi-Patch Ensemble Network.
    
    Structure:
    1. Patch Generator (9 patches)
    2. Spatial Stream (Xception) -> 2048 dim
    3. Frequency Stream (ResNet) -> 2048 dim
    4. Concat -> 4096 dim per patch
    5. Patch Classifiers (Auxiliary Loss)
    6. Ensemble Meta-Classifier (Global Prediction)
    """
    def __init__(self, num_patches=9, pretrained=True):
        super(DSMPE_Net, self).__init__()
        self.num_patches = num_patches
        
        # 1. Generator
        self.patch_gen = PatchGenerator()
        
        # 2. Streams
        self.spatial = SpatialStream(pretrained=pretrained)
        self.freq = FrequencyStream(pretrained=pretrained)
        
        # Concat dim
        self.fusion_dim = self.spatial.feature_dim + self.freq.feature_dim
        
        # 3. Patch-level Classifier (for Multi-Level Supervision)
        # Predicts Real/Fake for EACH patch
        self.patch_classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1) # Sigmoid applied in Loss usually, but here raw logits
        )
        
        # 4. Ensemble Meta-Classifier
        # Takes all patch features and aggregates them.
        # Simple approach: Concat all patch features? 9 * 4096 = 36864 dims (Large!)
        # Better approach: Attention or MLP aggregation.
        # Let's use a simple MLP aggregation as per typical ensemble logic.
        
        self.meta_input_dim = self.fusion_dim * num_patches 
        
        self.meta_classifier = nn.Sequential(
            nn.Linear(self.meta_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Global Logit
        )

    def forward(self, x):
        """
        x: (B, 3, 1024, 1024)
        Returns:
            global_logit: (B, 1)
            patch_logits: (B, 9)
        """
        B = x.shape[0]
        
        # 1. Patches
        patches = self.patch_gen(x) # (B, 9, 3, 256, 256)
        
        # Flatten for batch processing
        # (B*9, 3, 256, 256)
        patches_flat = patches.reshape(-1, 3, 256, 256)
        
        # 2. Streams
        s_feat = self.spatial(patches_flat) # (B*9, 2048)
        f_feat = self.freq(patches_flat)    # (B*9, 2048)
        
        # Fusion
        fused = torch.cat([s_feat, f_feat], dim=1) # (B*9, 4096)
        
        # 3. Patch Supervision
        patch_logits = self.patch_classifier(fused) # (B*9, 1)
        patch_logits = patch_logits.reshape(B, self.num_patches) # (B, 9)
        
        # 4. Meta Classification
        # Reshape to (B, 9*4096)
        meta_input = fused.reshape(B, -1)
        global_logit = self.meta_classifier(meta_input) # (B, 1)
        
        return global_logit, patch_logits

if __name__ == '__main__':
    # Test
    try:
        model = DSMPE_Net()
        dummy = torch.randn(2, 3, 1024, 1024)
        g, p = model(dummy)
        print(f"Global: {g.shape}, Patch: {p.shape}")
    except ImportError:
        print("Deps missing")
