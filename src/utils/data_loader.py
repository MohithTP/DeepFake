import os
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

class DeepfakeDataset(Dataset):

    def __init__(self, root_dir, phase='train', transform=None, limit=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.real_paths = sorted(list((self.root_dir / 'real').glob('*.jpg'))) + \
                          sorted(list((self.root_dir / 'real').glob('*.png')))
        self.fake_paths = sorted(list((self.root_dir / 'fake').glob('*.jpg'))) + \
                          sorted(list((self.root_dir / 'fake').glob('*.png')))
        
        if limit:
            self.real_paths = self.real_paths[:limit]
            self.fake_paths = self.fake_paths[:limit]
            
        self.image_paths = self.real_paths + self.fake_paths
        # Labels: 0 for Real, 1 for Fake
        self.labels = [0]*len(self.real_paths) + [1]*len(self.fake_paths)
        
        print(f"[{phase.upper()}] Loaded {len(self.real_paths)} Real, {len(self.fake_paths)} Fake images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        label = self.labels[idx]
        
        # Read Image
        # Setup for DSMPE-Net: 1024x1024 BGR -> RGB
        img = cv2.imread(img_path)
        if img is None:
            # Fallback for broken images
            img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if necessary
        if img.shape[0] != 1024 or img.shape[1] != 1024:
            img = cv2.resize(img, (1024, 1024))
            
        # OpenCV stores as H, W, C --> Torch expects C, H, W
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img / 255.0 # Normalize 0-1 (converges faster during training)
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.float32)
