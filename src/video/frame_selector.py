import cv2
import numpy as np
import torch

class SmartFrameSelector:
    """
    Selects the most representative and high-quality frames from a video.
    
    Filters:
    1. Deduplication (using dHash)
    2. Sharpness (using Laplacian Variance)
    3. Face Presence (Future integration with Facenet)
    """
try:
    from facenet_pytorch import MTCNN
except ImportError:
    print("Warning: facenet-pytorch not installed. Face detection disabled.")
    MTCNN = None

class SmartFrameSelector:
    """
    Selects the most representative and high-quality frames from a video.
    
    Filters:
    1. Deduplication (using dHash)
    2. Face Detection (MTCNN) - Highest Quality Face Crop
    3. Sharpness (using Laplacian Variance)
    """
    def __init__(self, hash_size=8, blur_threshold=5.0, diff_threshold=5, use_face_det=True, device='cpu'):
        self.hash_size = hash_size
        self.blur_threshold = blur_threshold
        self.diff_threshold = diff_threshold
        self.device = device
        
        self.mtcnn = None
        if use_face_det and MTCNN is not None:
            # Keep margin=0 or small to avoid background noise for now
            self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=device, margin=20)
        
    def get_dhash(self, image):
        """Compute Difference Hash."""
        resized = cv2.resize(image, (self.hash_size + 1, self.hash_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Compute difference between adjacent pixels
        diff = gray[:, 1:] > gray[:, :-1]
        return diff.flatten()
        
    def extract_face(self, frame):
        """
        Detects and returns the largest face crop. 
        Returns None if no face is found (or returns original frame if configured).
        """
        if self.mtcnn is None:
            return frame # Fallback to full frame
            
        try:
            # MTCNN expects RGB (PIL or Tensor), but allows numpy if it works.
            # Best to convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect
            boxes, _ = self.mtcnn.detect(frame_rgb)
            
            if boxes is not None and len(boxes) > 0:
                # Select largest box (already sorted if select_largest=True usually, but lets match logic)
                # Box format: [x1, y1, x2, y2]
                box = boxes[0]
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Clip to frame dims
                h, w, _ = frame.shape
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size == 0: 
                    return None
                    
                return face_crop
            else:
                return None # No face found
        except Exception as e:
            print(f"Face Det Error: {e}")
            return None

    def is_blurry(self, image):
        """Check if image is blurry using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_threshold

    def select_frames(self, video_path, max_frames=20):
        """
        Processes video and returns a list of high-quality, unique frames.
        """
        cap = cv2.VideoCapture(video_path)
        selected_frames = []
        last_hash = None
        
        count = 0
        while cap.isOpened() and len(selected_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. Deduplication check (every frame or sample?)
            # Sampling every 5th frame for speed
            if count % 5 != 0:
                count += 1
                continue
            
            curr_hash = self.get_dhash(frame)
            if last_hash is not None:
                dist = np.count_nonzero(curr_hash != last_hash)
                if dist < self.diff_threshold:
                    count += 1
                    continue # Skip duplicate/near-duplicate
            
            # 2. Blur check
            if self.is_blurry(frame):
                count += 1
                continue
            
            # 3. Success!
            selected_frames.append(frame)
            last_hash = curr_hash
            count += 1
            
        cap.release()
        return selected_frames

if __name__ == "__main__":
    # Test logic with a dummy or existing video
    selector = SmartFrameSelector()
    print("SmartFrameSelector initialized.")
