import torch
import cv2
import numpy as np
import argparse
from src.models.dsmpe_net import DSMPE_Net
from src.models.patch_extractor import PatchGenerator

def run_inference(args): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference running on: {device}")
    
    # 1. Load Model
    model = DSMPE_Net(pretrained=False).to(device)
    if args.weight_path:
        print(f"Loading weights from {args.weight_path}...")
        try:
            model.load_state_dict(torch.load(args.weight_path, map_location=device))
        except Exception as e:
            print(f"Error loading weights: {e}. Running with random weights (Sanity Check only).")
    model.eval()
    
    # 2. Preprocess Image
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Could not read image at {args.image_path}")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device) # (1, 3, 1024, 1024)
    
    # 3. Predict
    with torch.no_grad():
        global_logit, patch_logits = model(img_tensor)
        
        prob = torch.sigmoid(global_logit).item()
        patch_probs = torch.sigmoid(patch_logits).squeeze().cpu().numpy()
        
    # 4. Results
    verdict = "FAKE" if prob > 0.5 else "REAL"
    print("\n" + "="*30)
    print(f"RESULT: {verdict}")
    print(f"Confidence: {prob if prob > 0.5 else 1-prob:.2%}")
    print("="*30)
    
    print("\nLocal Patch Analysis (Explainability):")
    # 3x3 Grid
    for i in range(3):
        row = patch_probs[i*3 : (i+1)*3]
        row_str = " | ".join([f"{p:.2f}" for p in row])
        print(f"Row {i+1}: {row_str}")
    
    print("\n(Values close to 1.0 indicate highly suspicious patches)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to test image')
    parser.add_argument('--weight_path', type=str, default=None, help='Path to .pth weights')
    parser.add_argument('--visualize', action='store_true', help='Save the patches seen by the model to image.png')
    args = parser.parse_args()
    
    run_inference(args)
