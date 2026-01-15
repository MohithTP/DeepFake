import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchGenerator(nn.Module):
    """
    Splits a high-resolution image (1024x1024) into overlapping patches.
    Proposed configuration: 9 patches of 256x256.
    """
    def __init__(self, high_res_size=1024, patch_size=256, num_patches_per_axis=3):
        super(PatchGenerator, self).__init__()
        self.high_res_size = high_res_size
        self.patch_size = patch_size
        self.num_patches_per_axis = num_patches_per_axis
        
        # Calculate stride with overlap
        # We want 3 patches across 1024 pixels.
        # If we just tiled, 256*3 = 768 < 1024. So we must space them out to cover the image,
        # OR we center crop the patches from specific regions.
        #
        # For a standard 3x3 grid covering 1024x1024 with 256x256 patches, there will be GAPS if we don't scale.
        # However, the user request says "segmenting them into nine overlapping grids".
        # If we strictly use 256x256 patches on a 1024x1024 image, we need substantial stride.
        #
        # Let's define the centers for the 3x3 grid.
        # 1024 / 3 is approx 341.
        # We can place centers at roughly [170, 512, 854]
        
        step = high_res_size // num_patches_per_axis
        self.centers = [step//2, step + step//2, 2*step + step//2] # Approx centers
        
        # Actually, simpler approach: Use Unfold (sliding window)
        # But Unfold is rigid. 
        # Let's implement a dynamic crop based on the grid structure to ensure we cover the face.
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W).
        Returns:
            patches: Tensor of shape (B, 9, C, H_p, W_p)
        """
        B, C, H, W = x.shape
        
        # 1. Update params based on actual input size (Dynamic handling)
        # If input is smaller than 3*patch_size, we need overlap.
        # If input is too small (e.g. < 256), this will fail. We assume H, W >= 256.
        
        # Logic: 3 crops along height, 3 along width.
        # If H == 256, all 3 crops are the same? Or just return duplicates?
        # Ideally, we calculate top/left coordinates dynamically.
        
        h_step = 0
        w_step = 0
        
        if self.num_patches_per_axis > 1:
            h_step = (H - self.patch_size) // (self.num_patches_per_axis - 1)
            w_step = (W - self.patch_size) // (self.num_patches_per_axis - 1)
            
        # Safety for small images
        h_step = max(0, h_step)
        w_step = max(0, w_step)
        
        patches = []
        
        for i in range(self.num_patches_per_axis):
            for j in range(self.num_patches_per_axis):
                top = i * h_step
                left = j * w_step
                
                patch = x[:, :, top:top+self.patch_size, left:left+self.patch_size]
                patches.append(patch)

                
        # Stack into (B, 9, C, H_p, W_p)
        patches = torch.stack(patches, dim=1)
        return patches

if __name__ == '__main__':
    # Test Sanity
    dummy_img = torch.randn(2, 3, 1024, 1024)
    pg = PatchGenerator()
    out = pg(dummy_img)
    print(f"Input shape: {dummy_img.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 9, 3, 256, 256)
    print("PatchGenerator Test Passed")
