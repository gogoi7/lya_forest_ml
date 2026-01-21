import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from data_loader import load_data


# --- 1. MODEL DEFINITION (Robust High-Res ResNet) ---
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
        # High-Res Input Layer (No Stride, No MaxPool)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, 2)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)

        # Auto-Calculate Flatten Dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            dummy_out = self._forward_features(dummy_input)
            self.flattened_dim = dummy_out.view(1, -1).size(1)

        self.fc = nn.Linear(self.flattened_dim, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# --- 2. EXECUTION LOGIC ---
if __name__ == "__main__":
    # Settings
    PATH0 = "data/raw/EX0_spectra.hdf5"
    PATH1 = "data/raw/EX1_spectra.hdf5"
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005  # Slightly lower LR for stability
    EPOCHS = 20
    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )  # Use Mac GPU if available

    print(f"Loading data on {DEVICE}...")
    X_raw, y_raw = load_data(PATH0, PATH1)

    # Preprocessing
    X_flux = np.exp(-X_raw)

    # --- STRUCTURAL SPLIT (PREVENTS LEAKAGE) ---
    print("Applying structural train/test split...")
    LOS_p_sim = X_flux.shape[0] // 2
    uniq_idx = np.arange(LOS_p_sim)

    # Split indices
    train_idx_base, test_idx_base = train_test_split(
        uniq_idx, test_size=0.2, random_state=42
    )

    # Reconstruct full arrays
    X_train = np.concatenate(
        [X_flux[train_idx_base], X_flux[train_idx_base + LOS_p_sim]]
    )
    y_train = np.concatenate([y_raw[train_idx_base], y_raw[train_idx_base + LOS_p_sim]])

    X_test = np.concatenate([X_flux[test_idx_base], X_flux[test_idx_base + LOS_p_sim]])
    y_test = np.concatenate([y_raw[test_idx_base], y_raw[test_idx_base + LOS_p_sim]])

    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # Prepare DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model
    model = SpectralResNet(input_length=X_train.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    print("Starting training...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation Step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Acc: {val_acc:.2f}%"
        )

        # SAVE BEST MODEL
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> New best model saved! ({val_acc:.2f}%)")

    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.2f}%")
