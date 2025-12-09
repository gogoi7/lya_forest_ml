import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

from data_loader import load_data
from models import LyaCNN

# 1. Load Data
path0 = 'data/raw/EX0_spectra.hdf5'
path1 = 'data/raw/EX1_spectra.hdf5'

print("Loading data...")
X, y = load_data(path0, path1)

# --- DEBUG STEP 1: Check the Input Statistics ---
print(f"Raw Tau Stats -> Min: {np.min(X):.4f}, Max: {np.max(X):.4f}, Mean: {np.mean(X):.4f}")

# Apply Physics Normalization
X = np.exp(-X)

print(f"Flux Stats -> Min: {np.min(X):.4f}, Max: {np.max(X):.4f}, Mean: {np.mean(X):.4f}")
# ------------------------------------------------

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Convert to Tensor
tensor_x = torch.Tensor(X_train[:32]).unsqueeze(1) # Take ONLY first 32 samples
tensor_y = torch.Tensor(y_train[:32]).unsqueeze(1) # Take ONLY first 32 labels

# 2. Setup "Overfit" Experiment
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = LyaCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # Standard LR

tensor_x = tensor_x.to(device)
tensor_y = tensor_y.to(device)

print("\n--- Starting 'Overfit One Batch' Test ---")
# Loop 100 times on the SAME batch
for i in range(100):
    optimizer.zero_grad()
    outputs = model(tensor_x)
    loss = criterion(outputs, tensor_y)
    loss.backward()
    optimizer.step()
    
    if (i+1) % 10 == 0:
        # Check accuracy
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == tensor_y).float().mean()
        print(f"Iter {i+1}: Loss = {loss.item():.4f}, Accuracy = {acc*100:.0f}%")