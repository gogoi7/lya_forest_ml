import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data


# --- 1. THE HIGH-RES MODEL (Robust Version) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class SpectralResNet(nn.Module):
    def __init__(self, input_length, num_classes=2):
        super(SpectralResNet, self).__init__()

        self.in_channels = 16
        # Initial Layer: No Stride, preserve resolution
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)

        # Residual Layers
        self.layer1 = self._make_layer(16, 2)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)

        # --- AUTOMATIC SHAPE CALCULATION ---
        # Instead of guessing with division, we push fake data through to see the size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            dummy_out = self._forward_features(dummy_input)
            self.flattened_dim = dummy_out.view(1, -1).size(1)

        print(f"Auto-calculated Flattened Dimension: {self.flattened_dim}")
        self.fc = nn.Linear(self.flattened_dim, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _forward_features(self, x):
        """Helper to run just the conv parts"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# --- 2. THE TEST LOGIC ---
if __name__ == "__main__":
    PATH0 = "data/raw/EX0_spectra.hdf5"
    PATH1 = "data/raw/EX1_spectra.hdf5"
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"--- RUNNING SANITY CHECK ON {DEVICE} ---")

    X_raw, y_raw = load_data(PATH0, PATH1)

    # --- FIX: SHUFFLE DATA ---
    # We must shuffle to get a mix of Class 0 and Class 1 in the batch
    indices = np.arange(len(X_raw))
    np.random.shuffle(indices)

    BATCH_SIZE = 64
    batch_idx = indices[:BATCH_SIZE]  # Take random 64 indices

    X_batch = np.exp(-X_raw[batch_idx])
    y_batch = y_raw[batch_idx]

    # Verify we have both classes
    classes, counts = np.unique(y_batch, return_counts=True)
    print(f"Batch Class Distribution: {dict(zip(classes, counts))}")

    # Prepare Tensors
    inputs = torch.tensor(X_batch, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    labels = torch.tensor(y_batch, dtype=torch.long).to(DEVICE)

    # Initialize
    model = SpectralResNet(input_length=X_batch.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Overfit Loop
    print("\nAttempting to overfit on single batch...")
    for epoch in range(101):  # Increased to 100 just in case
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc = 100 * correct / BATCH_SIZE

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f} | Acc = {acc:.2f}%")

        if acc == 100.0:
            print(f"\nSUCCESS: Reached 100% accuracy at Epoch {epoch}!")
            break
