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
        from src.models.dsmpe_net import DSMPE_Net # type: ignore
        from src.models.patch_extractor import PatchGenerator # type: ignore
        from src.video.frame_selector import SmartFrameSelector # type: ignore
    else:
        raise ImportError("Torch not loaded")
    
    # Import the Agentic Router
    from src.agent.dispatcher import DispatcherAgent # type: ignore
    
    # Import Text Tamper Model
    from src.models.text_tamper import TextTamperDetector
except ImportError as e:
    print(f"Error importing model modules: {e}")
    print("Ensure 'deepfake_model' repo is cloned and dependencies are installed.")

class DeepFakeDetector:
    def __init__(self, weights_path, text_tamper_weights_path='xception_ela_doctamper_latest.pth'):
        self.weights_path = weights_path
        self.text_tamper_weights_path = text_tamper_weights_path
        if torch:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu' # Fallback
        self.model = None
        self.text_tamper_detector = None
        self.load_model()
        
        # Initialize the Intelligent Switchboard (Dispatcher)
        self.dispatcher = DispatcherAgent()

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
            
        # Initialize Text Tamper Detector
        try:
            self.text_tamper_detector = TextTamperDetector(weights_path=self.text_tamper_weights_path, device=self.device)
        except Exception as e:
            print(f"Failed to initialize Text Tamper Detector: {e}")
            self.text_tamper_detector = None

    def check_media(self, filepath, progress_callback=None):
        """
        Main entry point using Agentic Routing (Intelligent Switchboard).
        Returns: (is_fake: bool, score: float, patch_scores: list, meta: dict)
        """
        if progress_callback: progress_callback("Agentic Router Analyzing Input...")
        
        # Dispatch the request to the Intelligence Switchboard
        routing_info = self.dispatcher.dispatch(filepath)
        route = routing_info.get("route", "REJECT")
        reason = routing_info.get("reason", "Unknown routing decision.")
        
        if progress_callback: progress_callback(f"Router Decision: {route} ({reason[:50]}...)")
        print(f"[Agentic Router] Decision: {route} | Reason: {reason}")

        # MOCK MODE check
        if self.model is None and route != "REJECT":
            if progress_callback: progress_callback("Model not loaded. Running Mock Analysis...")
            print("Running in MOCK mode due to missing weights/model.")
            score = 0.95 if 'fake' in os.path.basename(filepath).lower() else 0.05
            import random
            mock_patches = [score + random.uniform(-0.1, 0.1) for _ in range(9)]
            mock_patches = [max(0.0, min(1.0, p)) for p in mock_patches] # Clip
            return score > 0.5, score, mock_patches, {'type': 'mock', 'info': 'Mock Analysis', 'agent_reason': reason}

        if route == "REJECT":
            return False, 0.0, [], {'status': 'rejected', 'reason': reason, 'type': 'agent_reject'}
        
        elif route == "FACE_PIPELINE":
            if progress_callback: progress_callback("Processing Face Forensics...")
            return self._check_image(filepath, progress_callback)
            
        elif route == "VIDEO_PIPELINE":
            return self._check_video(filepath, progress_callback)
            
        elif route == "TEXT_TAMPER":
            if progress_callback: progress_callback("Processing Textual Image Analysis...")
            if self.text_tamper_detector:
                is_fake, prob = self.text_tamper_detector.predict(filepath)
                return is_fake, prob, [], {'type': 'text_tamper', 'info': 'Processed by TextTamper model'}
            else:
                return False, 0.0, [], {'type': 'text_tamper', 'error': 'TextTamper detector not initialized'}
            
        else:
            print(f"Unknown route requested by Agent: {route}")
            return False, 0.0, [], {'error': f"Unknown route: {route}"}

    def _check_image(self, image_path, progress_callback=None):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image {image_path}")
            
            # --- AGENTIC REFLECTION (Feedback Loop) ---
            # We wrap the core inference to allow for reflection if confidence is low.
            
            def perform_inference(image_data):
                img_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (1024, 1024))
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device) 
                
                with torch.no_grad():
                    global_logit, patch_logits = self._run_inference_with_timing(img_tensor)
                    prob = torch.sigmoid(global_logit).item()
                    patch_probs = torch.sigmoid(patch_logits).squeeze().cpu().numpy().tolist()
                return prob, patch_probs

            prob, patch_probs = perform_inference(img)
            
            # Reflection Check: Low confidence (near 0.5)
            # If the score is between 0.4 and 0.6, the model is uncertain.
            if 0.4 < prob < 0.6:
                if progress_callback: progress_callback("Low Confidence Detected. Agent Reflecting...")
                print(f"[Reflection] Uncertain score {prob:.4f}. Attempting recovery via multi-scale analysis...")
                
                # RECOVERY ACTION: Try a centered crop or slight rotation 
                # This is a sample 'reflection' strategy.
                h, w = img.shape[:2]
                side = min(h, w)
                center_crop = img[(h-side)//2 : (h+side)//2, (w-side)//2 : (w+side)//2]
                
                new_prob, new_patch_probs = perform_inference(center_crop)
                print(f"[Reflection] Second pass score: {new_prob:.4f}")
                
                # If the second pass is more decisive (further from 0.5), we prefer it.
                if abs(new_prob - 0.5) > abs(prob - 0.5):
                    print("[Reflection] Adopting more confident second-pass score.")
                    prob, patch_probs = new_prob, new_patch_probs
                    reflection_status = "Reflected-Adopted"
                else:
                    reflection_status = "Reflected-Stayed"
            else:
                reflection_status = "Original"

            return prob > 0.5, prob, patch_probs, {'type': 'image', 'reflection': reflection_status}
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
                        logits, _ = self._run_inference_with_timing(img_tensor)
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
                return False, 0.0, [], {'frames_processed': 0, 'frames_sampled': frames_sampled, 'type': 'video', 'error': 'No valid frames extracted (No face/Too blurry)'}
            
            avg_score = np.mean(scores)
            print(f"Video Score: {avg_score:.4f} (Frames: {len(scores)}/{frames_sampled})")
            return avg_score > 0.5, avg_score, [], {'frames_processed': len(scores), 'frames_sampled': frames_sampled, 'type': 'video'}

        except Exception as e:
            print(f"Video inference error: {e}")
            return False, 0.0, [], {'frames_processed': 0, 'frames_sampled': 0, 'type': 'video', 'error': str(e)}

    def _run_inference_with_timing(self, img_tensor):
        """
        Manually run model forward pass to profile timing of specific stages.
        Replicates DSMPE_Net forward logic.
        """
        import time
        import logging

        if img_tensor.is_cuda:
            torch.cuda.synchronize()
        t_start = time.time()
        
        B = img_tensor.shape[0]
        model = self.model
        
        # --- Stage 1: Patch Extraction, Features, Patch Classifiers ---
        with torch.no_grad():
            # 1. Patches
            patches = model.patch_gen(img_tensor) # (B, 9, 3, 256, 256)
            patches_flat = patches.reshape(-1, 3, 256, 256)
            
            # 2. Streams
            s_feat = model.spatial(patches_flat)
            f_feat = model.freq(patches_flat)
            fused = torch.cat([s_feat, f_feat], dim=1) # (B*9, 4096)
            
            # 3. Patch Supervision
            patch_logits = model.patch_classifier(fused)
            patch_logits = patch_logits.reshape(B, model.num_patches)
            
            if img_tensor.is_cuda:
                torch.cuda.synchronize()
            t_mid = time.time()
            
            # --- Stage 2: Global Meta-Classifier ---
            # 4. Meta Classification
            meta_input = fused.reshape(B, -1)
            global_logit = model.meta_classifier(meta_input)
            
            if img_tensor.is_cuda:
                torch.cuda.synchronize()
            t_end = time.time()
            
        time_patch = t_mid - t_start
        time_global = t_end - t_mid
        time_total = t_end - t_start
        
        logging.info(f"TIMING [Inference]: Patch Analysis: {time_patch:.4f}s | Global Decision: {time_global:.4f}s | Total: {time_total:.4f}s")
        
        return global_logit, patch_logits
