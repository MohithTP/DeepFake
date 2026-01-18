import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.dsmpe_net import DSMPE_Net
from src.utils.data_loader import DeepfakeDataset
import argparse
from pathlib import Path

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset
    train_dataset = DeepfakeDataset(root_dir=args.data_dir, phase='train', limit=args.limit)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Model
    model = DSMPE_Net(pretrained=True).to(device)
    
    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Resume from checkpoint if provided
    start_epoch = 0
    if args.resume_path and Path(args.resume_path).exists():
        print(f"Resuming from checkpoint: {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(checkpoint)

        # Extract epoch number from filename if it follows dsmpe_net_epoch_N.pth
        if "epoch_" in args.resume_path:
            try:
                start_epoch = int(args.resume_path.split("epoch_")[-1].split(".")[0])
                print(f"Resuming from Epoch {start_epoch}")
            except:
                pass

    # 5. Multi-Level Loss
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting training...")
    model.train()
    
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device) # Labels: (B)
            
            optimizer.zero_grad()
            
            # Forward
            global_logit, patch_logits = model(images)
            # global_logit: (B, 1), patch_logits: (B, 9)
            
            # Loss Calculation
            # Global Loss
            global_loss = criterion(global_logit.squeeze(), labels)
            
            # Patch Loss
            # We assume if image is FAKE, some patches are FAKE. 
            # If image is REAL, ALL patches are REAL.
            # This is "Weakly Supervised" usually.
            # But DSMPE proposal implies training "individual patch classifiers".
            # Simple approach: Assign global label to all patches.
            # (Real=0 -> All patches 0. Fake=1 -> All patches 1? Not necessarily true, but good baseline)
            # Better: MIL (Multiple Instance Learning).
            
            labels_expanded = labels.unsqueeze(1).repeat(1, 9) # (B, 9)
            patch_loss = criterion(patch_logits, labels_expanded)
            
            loss = global_loss + 0.5 * patch_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy (Global)
            preds = torch.sigmoid(global_logit).squeeze() > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 2 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch {batch_idx}: Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} Acc: {100.*correct/total:.2f}%")
        
    # Save
    torch.save(model.state_dict(), "dsmpe_net_final.pth")
    print("Model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='d:/DeepFake/data/dummy', help='Path to dataset root')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--limit', type=int, default=None, help='Limit dataset size for debugging')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to .pth checkpoint to resume from')
    args = parser.parse_args()
    
    train(args)
