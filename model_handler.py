import os
import sys
import time
from flask import current_app

try:
    import eventlet
except ImportError:
    eventlet = None

try:
    import torch
    import cv2
    import numpy as np
except ImportError:
    torch = None
    cv2 = None
    np = None
    print("WARNING: ML libraries (torch/cv2/numpy) not found. Deepfake detection will be disabled.")

# Add deepfake_model to path to import specific modules
DEEPFAKE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'deepfake_model')
if DEEPFAKE_MODEL_PATH not in sys.path:
    sys.path.append(DEEPFAKE_MODEL_PATH)

try:
    if torch:
        from src.models.dsmpe_net import DSMPE_Net
        from src.models.patch_extractor import PatchGenerator
        from src.video.frame_selector import SmartFrameSelector
    else:
        raise ImportError("Torch not loaded")
except ImportError as e:
    print(f"Error importing model modules: {e}")
    print("Ensure 'deepfake_model' repo is cloned and dependencies are installed.")

class DeepFakeDetector:
    def __init__(self, weights_path):
        self.weights_path = weights_path
        if torch:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu' # Fallback
        self.model = None
        self.load_model()

    def load_model(self):
        if not torch:
            print("Torch not available. Model cannot be loaded.")
            self.model = None
            return

        print(f"Loading model on {self.device}...")
        if not os.path.exists(self.weights_path):
            print(f"WARNING: Weights file {self.weights_path} not found! Running in MOCK mode.")
            return

        try:
            self.model = DSMPE_Net(pretrained=False).to(self.device)
            state_dict = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def check_media(self, filepath, progress_callback=None):
        """
        Main entry point.
        Returns: (is_fake: bool, score: float, patch_scores: list, meta: dict)
        """
        # MOCK MODE check
        if self.model is None:
            if progress_callback: progress_callback("Model not loaded. Running Mock Analysis...")
            print("Running in MOCK mode due to missing weights/model.")
            # Mock logic: if filename has 'fake', it's fake.
            score = 0.95 if 'fake' in os.path.basename(filepath).lower() else 0.05
            # Mock patch scores (3x3 = 9 values)
            import random
            mock_patches = [score + random.uniform(-0.1, 0.1) for _ in range(9)]
            mock_patches = [max(0.0, min(1.0, p)) for p in mock_patches] # Clip
            return score > 0.5, score, mock_patches, {'type': 'mock', 'info': 'Mock Analysis'}

        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            if progress_callback: progress_callback("Analyzing image...")
            return self._check_image(filepath)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return self._check_video(filepath, progress_callback)
        else:
            print(f"Unsupported file type: {ext}")
            return False, 0.0, [], {'error': f"Unsupported file type: {ext}"}

    def _check_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image {image_path}")
            
            # Preprocess as per inference.py
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (1024, 1024))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device) # (1, 3, 1024, 1024)

            with torch.no_grad():
                global_logit, patch_logits = self.model(img_tensor)
                prob = torch.sigmoid(global_logit).item()

                # global_logit, patch_logits = self.model(img_tensor)
                
                # # Combine global and patch logits
                # # patch_logits shape: (1, 9) usually
                # all_logits = torch.cat([global_logit.view(-1), patch_logits.view(-1)])
                # all_probs = torch.sigmoid(all_logits)
                
                # # Top-K Average Consensus (K=5)
                # k = 5
                # top_k_probs, _ = torch.topk(all_probs, min(k, len(all_probs)))
                # prob = torch.mean(top_k_probs).item()

                patch_probs = torch.sigmoid(patch_logits).squeeze().cpu().numpy().tolist()
            return prob > 0.5, prob, patch_probs, {'type': 'image'}
        except Exception as e:
            print(f"Image inference error: {e}")
            return False, 0.0, [], {'error': str(e)}

    def _check_video(self, video_path, progress_callback=None):
        """
        Full Production Pipeline Logic (adapted from src/video/parallel_processor.py).
        1. Deduplication (dHash)
        2. Face Extraction (MTCNN)
        3. Quality Check (Blur)
        4. Inference
        """
        try:
            if progress_callback: progress_callback("Initializing Video Analysis Pipeline...")
            # Configure Selector
            selector = SmartFrameSelector(
                blur_threshold=5.0,
                diff_threshold=5,
                use_face_det=True if self.device.type == 'cuda' or self.device.type == 'cpu' else False, # Enable if possible
                device=str(self.device)
            )
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video {video_path}")

            scores = []
            frames_processed = 0
            frames_sampled = 0
            last_hash = None
            
            # Limit to maintain responsiveness, but higher than demo mode
            MAX_FRAMES = 30
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Global sample rate (process every 5th frame)
                if frames_sampled % 5 != 0:
                    frames_sampled += 1
                    continue
                
                if progress_callback and frames_sampled % 20 == 0:
                     progress_callback(f"Scanning frame stream... ({frames_sampled} frames checked)")
                     if eventlet: eventlet.sleep(0.01) # Yield to event loop explicit
                     else: time.sleep(0.01)
                
                # 1. Deduplication
                try:
                    curr_hash = selector.get_dhash(frame)
                    if last_hash is not None:
                        dist = np.count_nonzero(curr_hash != last_hash)
                        if dist < selector.diff_threshold:
                            frames_sampled += 1
                            continue # Duplicate
                    last_hash = curr_hash
                except Exception:
                    pass # Hash fail safe

                # 2. Face Extraction
                # extract_face handles MTCNN if available, else returns full frame
                processed_frame = selector.extract_face(frame)
                
                if processed_frame is None:
                    frames_sampled += 1
                    continue # No face found
                
                # 3. Blur Check
                if not selector.is_blurry(processed_frame):
                    # Preprocess for Model
                    img = cv2.resize(processed_frame, (1024, 1024))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        logits, _ = self.model(img_tensor)
                        prob = torch.sigmoid(logits).item()
                        scores.append(prob)
                        frames_processed += 1
                        
                        if progress_callback:
                             progress_callback(f"Analyzing Face Batch #{frames_processed}/{MAX_FRAMES}...")
                             if eventlet: eventlet.sleep(0.01) # Yield to event loop explicit
                             else: time.sleep(0.01)

                    if frames_processed >= MAX_FRAMES:
                        break
                
                frames_sampled += 1
            cap.release()
            
            if progress_callback: progress_callback("Finalizing forensic timeline...")
            
            if not scores:
                print("No valid frames extracted from video.")
                return False, 0.0, [], {'frames_processed': 0, 'frames_sampled': frames_sampled, 'type': 'video', 'error': 'No valid frames extracted (No face/Too blurry)'}, {}
            
            avg_score = np.mean(scores)
            print(f"Video Score: {avg_score:.4f} (Frames: {len(scores)}/{frames_sampled})")
            return avg_score > 0.5, avg_score, [], {'frames_processed': len(scores), 'frames_sampled': frames_sampled, 'type': 'video'}, {}

        except Exception as e:
            print(f"Video inference error: {e}")
            return False, 0.0, [], {'frames_processed': 0, 'frames_sampled': 0, 'type': 'video', 'error': str(e)}, {'error': str(e)}
