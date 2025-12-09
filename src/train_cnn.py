import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from data_loader import load_data
from models import LyaCNN

path0 = 'data/raw/EX0_spectra.hdf5'
path1 = 'data/raw/EX1_spectra.hdf5'

print("Loading data...")
X, y = load_data(path0, path1)

#X = np.exp(-X)
X = np.log1p(X)  # Alternative normalization for stability
print("Data loaded.")

# Split data into training and testing sets
train_idx, test_idx = train_test_split(np.arange(X.shape[0]//2), test_size=0.2, random_state=42)
X_train = np.concatenate([X[train_idx], X[train_idx + X.shape[0]//2]])
y_train = np.concatenate([y[train_idx], y[train_idx + X.shape[0]//2]])
X_test = np.concatenate([X[test_idx], X[test_idx + X.shape[0]//2]])
y_test = np.concatenate([y[test_idx], y[test_idx + X.shape[0]//2]])
print(f"Training Shape: {X_train.shape}")
print(f"Testing Shape: {X_test.shape}")

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("DataLoaders created.")

# Setup model, loss function, and optimizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #Use Mac GPU
print(f"Using device: {device}")

model = LyaCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
criterion = nn.BCEWithLogitsLoss() #binary cross-entropy with logits

# Training loop
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() 

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    model.eval()
    with torch.no_grad():
        # Get pedictions for the test set
        test_outputs = model(X_test_tensor.to(device))
        #Convert logits to binary predictions 0 or 1 (sigmoid threshold at 0.5)
        test_preds = (torch.sigmoid(test_outputs) > 0.5).float().cpu().numpy()

        test_accuracy = accuracy_score(y_test, test_preds)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# # Evaluation
# model.eval()
# with torch.no_grad():
#     # Get pedictions for the test set
#     test_outputs = model(X_test_tensor.to(device))
#     #Convert logits to binary predictions 0 or 1 (sigmoid threshold at 0.5)
#     test_preds = (torch.sigmoid(test_outputs) > 0.5).float().cpu().numpy()

#     test_accuracy = accuracy_score(y_test, test_preds)
#     print(f"Test Accuracy: {test_accuracy * 100:.2f}%")