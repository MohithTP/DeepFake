import torch
import torch.nn as nn
import numpy as np

# We used to need to import torch_dct, but it might not be available in the environment.
# Let's implement a simple DCT using standard matrix multiplication for stability and fewer dependencies.

def get_dct_filter(tile_size, in_channels, device='cpu'):
    # DCT-II basis functions
    # Returns a Conv2d weight of shape (C*tile_size*tile_size, 1, tile_size, tile_size)
    # Actually, we usually want (Channel_Out, Channel_In, K, K)
    
    # Basis: B_{u,v} = alpha(u)alpha(v) * cos(...) * cos(...)
    # But usually for deepfake detection, we want to extract the frequency coefficients 
    # and treat them as an image channel or input.
    pass

# Simpler approach: Use the torch-dct library logic or fft if available.
# Since we added torch-dct to requirements, let's assume we can use a functional wrapper.
# If unavailable, we fallback to scipy.fftpack logic ported to torch.

class DCTTransform(nn.Module):
    def __init__(self):
        super(DCTTransform, self).__init__()
        
    def forward(self, x):
        """
        Apply 2D DCT to the input tensor.
        Input: (B, C, H, W)
        Output: (B, C, H, W) - Log scaled DCT coefficients
        """
        # We can implement DCT via FFT: DCT-2 is related to DFT of symmetric extension.
        # fast implementation:
        return self.dct_2d(x)

    def dct_2d(self, x):
        # Taking DCT of last 2 dimensions
        # X_dct = DCT(x)
        # Using torch.fft.rfft2 might be an alternative, but let's try to do actual DCT.
        
        # A simple robust way is using fixed weights for fixed size (256x256).
        # But that's huge memory.
        
        # Method: DCT(X) = W * X * W.T where W is the DCT matrix.
        # This works if H=W.
        N = x.shape[-1]
        
        # Create DCT matrix on the fly or buffer it.
        # For simplicity in this specialized agent, let's assuming we generate it for the batch.
        
        dct_mat = self._get_dct_matrix(N, x.device) # (N, N)
        
        # Apply to last dimension (W)
        # x is (..., H, W). mat is (W, W). x @ mat.T -> DCT on rows
        t1 = torch.matmul(dct_mat, x.transpose(-1, -2)) # DCT on columns
        out = torch.matmul(dct_mat, t1.transpose(-1, -2).transpose(-1, -2)) # DCT on rows?
        
        # Check math:
        # DCT(A) = M A M'
        # x shape (B, C, N, N)
        # M shape (N, N)
        # M @ x -> (B, C, N, N) acting on columns? No, matmul acts on last 2 dims.
        # If A is (N, N), M@A is matrix mult. 
        # We want M * A * M.T
        
        dct_mat = dct_mat.unsqueeze(0).unsqueeze(0) # (1, 1, N, N)
        
        # M * A
        step1 = torch.matmul(dct_mat, x) 
        # (M * A) * M.T
        step2 = torch.matmul(step1, dct_mat.transpose(-1, -2))
        
        # Log scale for stability
        return torch.log(torch.abs(step2) + 1e-12)

    def _get_dct_matrix(self, N, device):
        # Return standard DCT-II matrix-
        k = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(0) # (1, N)
        n = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(1) # (N, 1)
        
        norm = torch.ones(N, 1, device=device)
        norm[0, 0] = 1 / np.sqrt(2)
        norm = norm * np.sqrt(2 / N)
        
        weights = torch.cos(np.pi / (2 * N) * k * (2 * n + 1))
        # weights: rows are k (freq), cols are n (time/space)
        # Wait, standard def: C_kn = ... cos(...)
        # We want rows to be basis vectors.
        # Usually DCT matrix D s.t. Y = DX
        # D_ij = c_i * cos(...)
        
        D = norm * torch.cos(np.pi * n * (2 * k + 1) / (2 * N))
        # We need D to be orthogonal.
        # This looks correct for orthonormal DCT.
        return D

if __name__ == '__main__':
    dct = DCTTransform()
    x = torch.randn(1, 3, 256, 256)
    out = dct(x)
    print(f"DCT Output shape: {out.shape}")
