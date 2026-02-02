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
        self.fusion_dim = self.spatial.feature_dim + self.freq.feature_dim # 4096
        
        # 3. Patch-level Classifier (Auxiliary)
        self.patch_classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        # 4. ViT Aggregator (DeepScan 2.0)
        # Replaces the flat MLP with a Transformer Encoder
        # Input: Sequence of (9 patches + 1 CLS token)
        # Dim: fusion_dim (4096)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.fusion_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.fusion_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.fusion_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1) # Global Logit from CLS token
        )

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Patches
        patches = self.patch_gen(x) # (B, 9, 3, 256, 256)
        patches_flat = patches.reshape(-1, 3, 256, 256)
        
        # 2. Streams
        s_feat = self.spatial(patches_flat) # (B*9, 2048)
        f_feat = self.freq(patches_flat)    # (B*9, 2048)
        
        # Fusion
        fused = torch.cat([s_feat, f_feat], dim=1) # (B*9, 4096)
        
        # 3. Patch Supervision
        patch_logits = self.patch_classifier(fused)
        patch_logits = patch_logits.reshape(B, self.num_patches) # (B, 9)
        
        # 4. ViT Aggregation
        # Reshape to Sequence: (B, 9, 4096)
        sequence = fused.reshape(B, self.num_patches, -1)
        
        # Add CLS Token
        cls_tokens = self.cls_token.repeat(B, 1, 1) # (B, 1, 4096)
        sequence = torch.cat([cls_tokens, sequence], dim=1) # (B, 10, 4096)
        
        # Add Positional Embedding
        sequence += self.pos_embedding[:, :(self.num_patches + 1)]
        
        # Transformer Pass
        transformer_out = self.transformer(sequence) # (B, 10, 4096)
        
        # Extract CLS token output (Index 0)
        cls_out = transformer_out[:, 0] # (B, 4096)
        
        # Final Classification
        global_logit = self.head(cls_out) # (B, 1)
        
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
