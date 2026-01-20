import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.dsmpe_net import DSMPE_Net
from src.utils.data_loader import DeepfakeDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
import time

def evaluate_model(args):
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Evaluation running on: {device}")
    
    # 2. Load Model
    print(f"📦 Loading Model Weights: {args.weights}")
    model = DSMPE_Net(pretrained=False).to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print(f"❌ Error: Weight file not found at {args.weights}")
        return
        
    model.eval()
    
    # 3. Load Data
    print(f"📊 Loading Validation Data from: {args.data_dir}")
    # Note: data_loader.py might expect 'train'/'test' phase logic. 
    # If folder is 'valid', we can treat it as 'test' (no shuffle, no augment)
    val_dataset = DeepfakeDataset(root_dir=args.data_dir, phase='test') 
    
    if len(val_dataset) == 0:
        print("❌ Dataset Empty! Check path structure (should have 'real' and 'fake' subfolders).")
        return

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"🚀 Starting Evaluation on {len(val_dataset)} images...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            global_logit, patch_logits = model(images)
            
            # Use global logit for final classification
            probs = torch.sigmoid(global_logit).squeeze()
            
            # Handle batch size 1 vs N
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)
                
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if i % 10 == 0:
                print(f"   Processed batch {i}/{len(val_loader)}")

    duration = time.time() - start_time
    print(f"✅ Inference finished in {duration:.2f}s")

    # 4. Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5 # Fail-safe if only one class present
        print("⚠️ Warning: Could not calc AUC (maybe only one class in dataset?)")

    print("\n" + "="*40)
    print("       🔍 FINAL VALIDATION RESULTS")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}  (Trustworthiness)")
    print(f"Recall:    {rec:.4f}  (Detection Ability)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print("="*40)

    # 5. Confusion Matrix Visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix (Acc: {acc*100:.1f}%)')
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✅ Saved confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="dataset/valid", help='Path to validation dataset')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth)')
    parser.add_argument('--batch_size', type=int, default=16)
    
    args = parser.parse_args()
    evaluate_model(args)
