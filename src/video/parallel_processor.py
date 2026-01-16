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
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                
                
                if len(batch) == batch_size:
                    input_tensor = torch.from_numpy(np.array(batch)).permute(0, 3, 1, 2).to(self.device)
                    with torch.no_grad():
                        logits, _ = model(input_tensor)
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
        final_score = np.mean(scores) if scores else 0.5
        
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
