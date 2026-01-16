import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from collections import OrderedDict
from src.models.dsmpe_net import DSMPE_Net
from src.utils.data_loader import DeepfakeDataset
import argparse

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class DeepfakeClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.to(device)
        self.model.train()
        
        # Train for 1 LOCAL EPOCH
        total_loss = 0
        for images, labels in self.trainloader:
            images, labels = images.to(device), labels.to(device)
            self.optimizer.zero_grad()
            global_logit, patch_logits = self.model(images)
            
            # Multi-level loss
            global_loss = self.criterion(global_logit.squeeze(), labels)
            labels_expanded = labels.unsqueeze(1).repeat(1, 9)
            patch_loss = self.criterion(patch_logits, labels_expanded)
            loss = global_loss + 0.5 * patch_loss
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return get_parameters(self.model), len(self.trainloader.dataset), {"loss": total_loss / len(self.trainloader)}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.to(device)
        self.model.eval()
        
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = self.model(images)
                loss += self.criterion(outputs.squeeze(), labels).item()
                preds = torch.sigmoid(outputs).squeeze() > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0
        return loss / len(self.valloader), total, {"accuracy": accuracy}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_address', type=str, default="127.0.0.1:8080")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    # Load data
    trainset = DeepfakeDataset(root_dir=args.data_dir, phase='train')
    valset = DeepfakeDataset(root_dir=args.data_dir, phase='test') # Using test as val for MVP
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size)

    # Load model
    model = DSMPE_Net(pretrained=True).to(device)

    # Start client
    print(f"Connecting to server at {args.server_address}...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=DeepfakeClient(model, trainloader, valloader)
    )

if __name__ == "__main__":
    main()
