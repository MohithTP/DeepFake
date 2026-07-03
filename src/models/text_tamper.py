import os
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageChops, ImageEnhance

try:
    import timm
except ImportError:
    timm = None

def get_ela_tensor(img, quality=90):
    """
    Converts a PIL RGB Image -> ELA Image -> PyTorch Tensor
    This reveals compression artifacts (pasted text looks brighter/noisier).
    """
    buffer = io.BytesIO()
    img.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    
    img_compressed = Image.open(buffer)
    ela_img = ImageChops.difference(img, img_compressed)
    
    extrema = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
    return transforms.functional.to_tensor(ela_img)

class TextTamperDetector:
    def __init__(self, weights_path, device='cpu'):
        self.device = device
        self.weights_path = weights_path
        self.model = None
        self.load_model()

    def load_model(self):
        if not timm:
            print("timm not available. Text Tamper Model cannot be loaded.")
            return

        if not os.path.exists(self.weights_path):
            print(f"WARNING: Text Tamper Weights {self.weights_path} not found! Running in MOCK mode.")
            return

        try:
            print(f"Loading Text Tamper model on {self.device}...")
            # Initialize Xception for binary classification (1 class output with BCEWithLogitsLoss)
            self.model = timm.create_model('xception', pretrained=False)
            num_ftrs = self.model.get_classifier().in_features
            self.model.fc = nn.Linear(num_ftrs, 1)
            
            state_dict = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("Text Tamper Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load Text Tamper model: {e}")
            self.model = None

    def predict(self, image_path):
        """
        Returns (is_fake: bool, prob: float)
        """
        if self.model is None:
            # Mock Mode
            print("TextTamper Running in MOCK mode due to missing weights/model.")
            score = 0.95 if 'fake' in os.path.basename(image_path).lower() else 0.05
            return score > 0.5, score

        try:
            image = Image.open(image_path).convert('RGB')
            # Resize
            image = transforms.Resize((256, 256))(image)
            
            # Get ELA
            image_tensor = get_ela_tensor(image)
            
            # Normalize
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            image_tensor = normalize(image_tensor)
            
            # Inference
            input_tensor = image_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(input_tensor)
                prob = torch.sigmoid(logits).item()
                
            return prob > 0.5, prob
        except Exception as e:
            print(f"Text Tamper inference error: {e}")
            raise e
