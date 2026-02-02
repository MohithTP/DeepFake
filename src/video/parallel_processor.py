import torch
import torch.multiprocessing as mp
import cv2
import time
from src.video.frame_selector import SmartFrameSelector
from src.models.dsmpe_net import DSMPE_Net
from src.models.patch_extractor import PatchGenerator
import torchvision.utils as vutils
import os
import numpy as np

import traceback

def frame_producer(video_path, queue, selector_config):
    """
    Worker process: Decodes video and selects best frames using dHash and Sharpness.
    """
    try:
        # print("DEBUG: Producer Process Started")
        selector = SmartFrameSelector(**selector_config)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {video_path}")
            queue.put(None)
            return

        count = 0
        selected_count = 0
        last_hash = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every 5th frame
            if count % 5 == 0:
                curr_hash = selector.get_dhash(frame)
                is_dup = False
                if last_hash is not None:
                    dist = np.count_nonzero(curr_hash != last_hash)
                    if dist < selector.diff_threshold:
                        is_dup = True
                
                if not is_dup:
                    # 3. Face Detection (New)
                    # If this returns None, it means no face found (and we should skip if strict).
                    # But extract_face() handles the logic: returns frame if mtcnn is missing.
                    # If mtcnn is present, returns None if no face.
                    
                    processed_frame = selector.extract_face(frame)
                    
                    if processed_frame is None:
                        # Skip frame if face detection is active but no face found
                        continue
                        
                    # 4. Blur Check (on the face crop)
                    if not selector.is_blurry(processed_frame):
                        # Pre-preprocess: Model expects 1024x1024 RGB
                        img = cv2.resize(processed_frame, (1024, 1024))
                        # processed_frame comes from selector.extract_face.
                        # If MTCNN is used inside, it usually returns RGB.
                        # If selector returns RGB, we should NOT convert BGR2RGB again.
                        
                        # Let's assume selector returns BGR (OpenCV standard) for safety, 
                        # OR if it returns RGB, we need to know.
                        
                        # Fix: Check frame_selector.py first.
                        # For now, let's just ensure we divide by 255.0
                        
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Potentially double-swap?
                        
                        # DEBUG: Save one frame to check color/crop
                        if selected_count == 0:
                            debug_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite("debug_input_frame.png", debug_save)
                            
                        img = img.astype(np.float32) / 255.0
                        queue.put(img)
                        selected_count += 1
                        last_hash = curr_hash
            count += 1
        
        cap.release()
        # print(f"DEBUG: Producer finishing. Sent {selected_count} frames.")
    except Exception as e:
        print(f"❌ Producer Process Error: {e}")
        traceback.print_exc()
    finally:
        queue.put(None) # EOF Signal

class ParallelProcessor:
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def run_inference(self, video_path, batch_size=4, selector_config={}, visualize=False):
        """
        Orchestrates parallel decoding (CPU) and inference (GPU).
        """
        ctx = mp.get_context('spawn')
        queue = ctx.Queue(maxsize=30)
        
        # 1. Start Producer process
        p = ctx.Process(target=frame_producer, args=(video_path, queue, selector_config))
        p.start()
        
        # 2. Load Model in main process
        print(f"Loading Model on {self.device}...")
        model = DSMPE_Net(pretrained=False).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        
        scores = []
        batch = []
        
        print(f"Starting Parallel Analysis (Batch Size: {batch_size})...")
        start_time = time.time()
        
        try:
            while True:
                frame_data = queue.get()
                if frame_data is None:
                    break
                
                batch.append(frame_data)
                
                if visualize and len(scores) == 0 and len(batch) == 1:
                    try:
                        print("   -> Generating debug visualization for first frame...")
                        pg = PatchGenerator() 
                        # batch is list of (1024, 1024, 3) arrays
                        # Convert to (1, 3, 1024, 1024)
                        dummy_batch = torch.from_numpy(np.array(batch)).permute(0, 3, 1, 2)
                        patches = pg(dummy_batch) # (1, 9, 3, 256, 256)
                        patches_flat = patches.reshape(-1, 3, 256, 256)
                        vutils.save_image(patches_flat, "debug_patches.png", nrow=3, padding=2, normalize=True)
                        print("   -> Saved 'debug_patches.png' (Shows the 9 patches model analyzes)")
                    except Exception as e:
                        print(f"   [!] Visualization failed: {e}")
                
                
                    input_tensor = torch.from_numpy(np.array(batch)).permute(0, 3, 1, 2).to(self.device)
                    # DEBUG: Check Input
                    if len(scores) == 0:
                         print(f"   [DEBUG] Input Tensor Stats: Min={input_tensor.min():.4f}, Max={input_tensor.max():.4f}, Mean={input_tensor.mean():.4f}")
                         
                    with torch.no_grad():
                        logits, _ = model(input_tensor)
                        
                        # DEBUG: Check Logits
                        if len(scores) == 0:
                            print(f"   [DEBUG] Raw Logits: {logits.cpu().numpy().flatten()}")
                            
                        probs = torch.sigmoid(logits).cpu().numpy().flatten()
                        scores.extend(probs)
                    batch = []
                    
            # Process remaining frames
            if batch:
                input_tensor = torch.from_numpy(np.array(batch)).permute(0, 3, 1, 2).to(self.device)
                with torch.no_grad():
                    logits, _ = model(input_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    scores.extend(probs)
        except Exception as e:
            print(f"Inference Error: {e}")
        finally:
            p.join()
        
        elapsed = time.time() - start_time
        
        # --- DEEPSCAN 2.0: SMART CONSENSUS LOGIC ---
        # Instead of simple mean, we look for "Strong Evidence".
        # Deepfakes often have "flicker" where some frames look real, but some look obviously fake.
        
        scores = np.array(scores)
        if len(scores) == 0:
            return 0.0
            
        # 1. High Confidence Detection Filter
        # How many frames are we "Very Sure" (> 0.8) are fake?
        strong_fake_count = np.sum(scores > 0.8)
        strong_fake_ratio = strong_fake_count / len(scores)
        
        # 2. Consecutive Spike Check (Temporal Consistency)
        # Deepfakes don't usually turn on/off for 1 frame. They persist.
        # Check for 3 consecutive frames > 0.7
        consecutive_fake = 0
        max_consecutive = 0
        current_streak = 0
        for s in scores:
            if s > 0.7:
                current_streak += 1
            else:
                max_consecutive = max(max_consecutive, current_streak)
                current_streak = 0
        max_consecutive = max(max_consecutive, current_streak)
        
        # 3. Power Mean (Emphasize outlier high scores)
        # If we have 100 frames, and 5 are 0.99 and 95 are 0.01:
        # Arithmetic Mean = 0.059 (VERDICT: REAL) -> WRONG
        # Power Mean (p=2) will be higher.
        # But let's use a "Top-K" logic.
        
        top_k_score = np.mean(np.sort(scores)[-max(1, int(len(scores)*0.1)):]) # Top 10%
        
        print(f"Stats: HighConf Frames: {strong_fake_count}/{len(scores)} | Max Streak: {max_consecutive} | Top 10% Means: {top_k_score:.4f}")
        
        # --- FINAL VERDICT ---
        # Rule A: If > 20% of frames are specific Strong Fakes
        if strong_fake_ratio > 0.2:
            final_score = 0.95 # Confirmed Fake
        # Rule B: If we found a sustained glitch (Streak > 5 frames)
        elif max_consecutive >= 5:
            final_score = 0.85 # Probable Fake
        # Rule C: If the Top 10% most suspicious frames average > 0.85
        elif top_k_score > 0.85:
            final_score = 0.80 # Suspicious
        else:
            final_score = np.mean(scores) # Fallback to average (Likely Real)
        
        print(f"Analysis Complete in {elapsed:.2f}s.")
        print(f"Processed {len(scores)} high-quality frames.")
        return final_score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, required=True, help='Path to .pth model weights')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--blur_thresh', type=float, default=5.0, help='Blur threshold (lower = more sensitive)')
    parser.add_argument('--visualize', action='store_true', help='Save the patches seen by the model to debug_patches.png')
    parser.add_argument('--no_face', action='store_true', help='Disable Face Detection (Use full frame)')
    args = parser.parse_args()
    
    if args.visualize:
        print("\n[INFO] Visualization Mode Enabled: Saving patch view to 'debug_patches.png'")
        
    # Configure Selector
    sel_config = {
        'blur_threshold': args.blur_thresh,
        'use_face_det': not args.no_face,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if not args.no_face:
        print("[INFO] Face Detection Enabled (MTCNN)")

    processor = ParallelProcessor(args.model)
    score = processor.run_inference(args.video, args.batch_size, selector_config=sel_config, visualize=args.visualize)
    
    print("\n" + "="*40)
    print(f"PRO-PIPELINE VIDEO SCORE: {score:.4f}")
    if score > 0.5:
        print("[!] VERDICT: FAKE VIDEO [!]")
    else:
        print("[OK] VERDICT: REAL VIDEO")
    print("="*40)
