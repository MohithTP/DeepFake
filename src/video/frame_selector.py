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
    def __init__(self, hash_size=8, blur_threshold=5.0, diff_threshold=5):
        self.hash_size = hash_size
        self.blur_threshold = blur_threshold
        self.diff_threshold = diff_threshold
        
    def get_dhash(self, image):
        """Compute Difference Hash."""
        resized = cv2.resize(image, (self.hash_size + 1, self.hash_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Compute difference between adjacent pixels
        diff = gray[:, 1:] > gray[:, :-1]
        return diff.flatten()

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
