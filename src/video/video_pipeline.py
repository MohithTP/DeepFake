import torch
import cv2
import argparse
import numpy as np
from src.video.frame_selector import SmartFrameSelector
from src.models.dsmpe_net import DSMPE_Net

class VideoPipeline:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Pipe on {self.device}...")
        
        # 1. Load Model
        self.model = DSMPE_Net(pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("Model Loaded.")
        
        # 2. Init Components
        self.selector = SmartFrameSelector()
        
    def process_video(self, video_path):
        print(f"Processing: {video_path}")
        
        # 1. Select Frames
        frames = self.selector.select_frames(video_path, max_frames=20)
        if not frames:
            print("No valid frames found (Too blurry / duplicates?)")
            return 0.5 # Uncertain
        
        print(f"Selected {len(frames)} frames for analysis.")
        
        # 2. Preprocess Batch
        batch = []
        for frame in frames:
            # Resize & Normalize
            img = cv2.resize(frame, (1024, 1024)) # Model expects 1024
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            batch.append(img)
            
        # (B, H, W, C) -> (B, C, H, W)
        batch_tensor = torch.tensor(np.array(batch)).permute(0, 3, 1, 2).float().to(self.device)
        
        # 3. Inference
        with torch.no_grad():
            global_logits, _ = self.model(batch_tensor)
            probs = torch.sigmoid(global_logits).cpu().numpy().flatten()
            
        # 4. Aggregation Strategy
        # Average Probability? Max? Voting?
        # Voting is usually robust. average is safer.
        avg_prob = np.mean(probs)
        
        # Detailed Report
        print(f"Frame Scores: {[f'{p:.2f}' for p in probs]}")
        
        return avg_prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    pipeline = VideoPipeline(args.model)
    score = pipeline.process_video(args.video)
    
    print("\n" + "="*30)
    print(f"FINAL VIDEO SCORE: {score:.4f}")
    if score > 0.5:
        print("🚨 VERDICT: FAKE VIDEO 🚨")
    else:
        print("✅ VERDICT: REAL VIDEO")
    print("="*30)
