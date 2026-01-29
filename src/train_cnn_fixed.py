import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

from data_loader import load_data
from models import LyaResNet

# Settings
path0 = 'data/raw/EX0_spectra.hdf5'
path1 = 'data/raw/EX1_spectra.hdf5'
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("=" * 60)
print("IMPROVED CNN TRAINING WITH PROPER NORMALIZATION")
print("=" * 60)

# Load data
print("\n[1/5] Loading data...")
X_raw, y_raw = load_data(path0, path1)
print(f"Raw data shape: {X_raw.shape}")
print(f"Raw tau stats - Min: {np.min(X_raw):.4f}, Max: {np.max(X_raw):.4f}, Mean: {np.mean(X_raw):.4f}, Std: {np.std(X_raw):.4f}")

# Apply structural split (same as RF baseline to prevent data leakage)
print("\n[2/5] Applying structural train/test split...")
LOS_p_sim = X_raw.shape[0] // 2
uniq_idx = np.arange(LOS_p_sim)

train_idx_base, test_idx_base = train_test_split(
    uniq_idx, test_size=0.2, random_state=42
)

# Split data
X_train_raw = np.concatenate([X_raw[train_idx_base], X_raw[train_idx_base + LOS_p_sim]])
y_train = np.concatenate([y_raw[train_idx_base], y_raw[train_idx_base + LOS_p_sim]])

X_test_raw = np.concatenate([X_raw[test_idx_base], X_raw[test_idx_base + LOS_p_sim]])
y_test = np.concatenate([y_raw[test_idx_base], y_raw[test_idx_base + LOS_p_sim]])

print(f"Train size: {X_train_raw.shape[0]} | Test size: {X_test_raw.shape[0]}")

# Normalization strategy: Try multiple approaches
print("\n[3/5] Normalizing data...")
print("Strategy: Standardization (mean=0, std=1) per sample")
print("This is crucial for neural networks!")

# Option 1: Standardize each spectrum independently (recommended for spectral data)
# This normalizes each sample to have mean=0, std=1
X_train = np.zeros_like(X_train_raw, dtype=np.float32)
X_test = np.zeros_like(X_test_raw, dtype=np.float32)

for i in range(X_train_raw.shape[0]):
    mean = np.mean(X_train_raw[i])
    std = np.std(X_train_raw[i])
    if std > 1e-8:  # Avoid division by zero
        X_train[i] = (X_train_raw[i] - mean) / std
    else:
        X_train[i] = X_train_raw[i] - mean

for i in range(X_test_raw.shape[0]):
    mean = np.mean(X_test_raw[i])
    std = np.std(X_test_raw[i])
    if std > 1e-8:
        X_test[i] = (X_test_raw[i] - mean) / std
    else:
        X_test[i] = X_test_raw[i] - mean

print(f"Normalized train stats - Mean: {np.mean(X_train):.4f}, Std: {np.std(X_train):.4f}")
print(f"Normalized test stats - Mean: {np.mean(X_test):.4f}, Std: {np.std(X_test):.4f}")

# Convert to tensors
print("\n[4/5] Preparing DataLoaders...")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Setup model
print("\n[5/5] Initializing model...")
print(f"Using device: {DEVICE}")

model = LyaResNet().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
criterion = nn.BCEWithLogitsLoss()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

# Training loop
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

best_acc = 0.0
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
    
    avg_train_loss = running_loss / num_batches
    scheduler.step(avg_train_loss)
    
    # Evaluation phase
    model.eval()
    with torch.no_grad():
        # Evaluate on test set
        test_outputs = model(X_test_tensor.to(DEVICE))
        test_preds = (torch.sigmoid(test_outputs) > 0.5).float().cpu().numpy().flatten()
        test_accuracy = accuracy_score(y_test, test_preds)
        
        # Also check training accuracy for monitoring
        train_outputs = model(X_train_tensor.to(DEVICE))
        train_preds = (torch.sigmoid(train_outputs) > 0.5).float().cpu().numpy().flatten()
        train_accuracy = accuracy_score(y_train, train_preds)
    
    # Track best model
    if test_accuracy > best_acc:
        best_acc = test_accuracy
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'best_model.pth')
    
    # Print progress
    current_lr = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_accuracy*100:5.2f}% | "
              f"Test Acc: {test_accuracy*100:5.2f}% | "
              f"LR: {current_lr:.2e}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Best test accuracy: {best_acc*100:.2f}% (epoch {best_epoch})")

# Final evaluation with detailed metrics
print("\n" + "=" * 60)
print("DETAILED EVALUATION")
print("=" * 60)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor.to(DEVICE))
    test_preds = (torch.sigmoid(test_outputs) > 0.5).float().cpu().numpy().flatten()

print("\nClassification Report:")
print(classification_report(y_test, test_preds, target_names=['EX0', 'EX1']))
