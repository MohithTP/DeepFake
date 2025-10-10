import pandas as pd
import numpy as np
import torch #type: ignore
import torch.nn as nn #type: ignore
from torch.utils.data import Dataset, DataLoader #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import accuracy_score, classification_report #type: ignore
from gensim.models.fasttext import load_facebook_model #type: ignore

# Load FastText model
ft_model = load_facebook_model("archive1\cc.en.300.bin")  # Or use load_facebook_model for .bin files

# Preprocessing: Convert text to mean FastText embeddings
def get_fasttext_embedding(text, model, dim=300):
    words = text.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(dim)

# Create a dataset
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, model):
        self.texts = [get_fasttext_embedding(text, model) for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Define LSTM-based model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=128, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use last hidden state
        return out

# Prepare data
df = pd.read_csv("WELFake_Dataset.csv")  # Make sure it has 'text' and 'label' columns
df['text'] = df['text'].fillna('').astype(str)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

train_dataset = FakeNewsDataset(X_train, y_train.values, ft_model)
test_dataset = FakeNewsDataset(X_test, y_test.values, ft_model)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize and train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluate
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc:.4f}")
print(classification_report(all_labels, all_preds))
