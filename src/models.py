import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    A standard Residual Block for 1D data.
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> (+ Input) -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ResBlock, self).__init__()

        padding = (kernel_size - 1) // 2  # Auto-calculate padding to keep length same

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # If we change dimensions (stride > 1), we need to resize the 'shortcut' too
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class LyaResNet(nn.Module):
    def __init__(self):
        super(LyaResNet, self).__init__()

        # Initial processing (Stem)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual Layers
        # We stack blocks to go deeper without losing the signal
        self.layer1 = ResBlock(32, 32, kernel_size=5)
        self.layer2 = ResBlock(32, 64, kernel_size=5, stride=2)  # Stride 2 downsamples
        self.layer3 = ResBlock(64, 128, kernel_size=5, stride=2)
        self.layer4 = ResBlock(128, 256, kernel_size=5, stride=2)

        # Classifier (Global Average Pooling + Linear)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Force output to length 1
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        out = self.stem(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)  # Flatten (Batch, 256)
        out = self.fc(out)

        return out


if __name__ == "__main__":
    model = LyaResNet()
    dummy = torch.randn(8, 1, 2499)
    print(f"Output shape: {model(dummy).shape}")  # Should be [8, 1]

# import torch
# import torch.nn as nn

# class LyaCNN(nn.Module):
#     def __init__(self):
#         super(LyaCNN, self).__init__()

#         # --- Feature Extractor (The "Eye") ---
#         # We keep the kernels large (11) to see absorption line shapes
#         self.conv1 = nn.Conv1d(1, 16, kernel_size=11, padding=5)
#         self.bn1 = nn.BatchNorm1d(16)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(2)

#         self.conv2 = nn.Conv1d(16, 32, kernel_size=11, padding=5)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool1d(2)

#         # --- The Global Aggregator (The "Brain") ---
#         # NO MORE FLATTENING CALCULATION!
#         # Global Average Pooling turns (Batch, 32, Length) -> (Batch, 32)
#         # So the input to the linear layer is just the number of channels.
#         self.fc1 = nn.Linear(32, 64)

#         self.relu3 = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(64, 1)

#     def forward(self, x):
#         # Block 1
#         x = self.pool1(self.relu1(self.bn1(self.conv1(x))))

#         # Block 2
#         x = self.pool2(self.relu2(self.bn2(self.conv2(x))))

#         # --- GLOBAL AVERAGE POOLING ---
#         # x is currently shape (Batch, 32, Length)
#         # We average across the last dimension (the spectrum length)
#         x = x.mean(dim=2)
#         # x is now shape (Batch, 32)

#         # Classifier
#         x = self.fc1(x)
#         x = self.relu3(x)
#         x = self.dropout(x)
#         x = self.fc2(x)

#         return x

# if __name__ == "__main__":
#     model = LyaCNN()
#     # Test with random length to prove it handles anything!
#     dummy_input = torch.randn(8, 1, 2499)
#     print(f"Output: {model(dummy_input).shape}")
# import torch
# import torch.nn as nn

# class LyaCNN(nn.Module):
#     def __init__(self):
#         super(LyaCNN, self).__init__()

#         # --- BLOCK 1 ---
#         # Increased kernel size to 11 to see "absorption lines" better
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=11, padding=5)
#         self.bn1 = nn.BatchNorm1d(16) # Batch Norm to stabilize training
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(kernel_size=4) # Aggressive pooling to reduce dimensionality

#         # --- BLOCK 2 ---
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, padding=5)
#         self.bn2 = nn.BatchNorm1d(32) # Batch Norm
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool1d(kernel_size=4)

#         # --- CLASSIFIER ---
#         # Calculation:
#         # Input: 2499
#         # Pool1 (/4) -> ~624
#         # Pool2 (/4) -> ~156
#         self.flattened_size = 32 * 156

#         self.fc1 = nn.Linear(self.flattened_size, 64)
#         self.relu3 = nn.ReLU()
#         self.dropout = nn.Dropout(0.1) #prevent overfitting
#         self.fc2 = nn.Linear(64, 1)

#     def forward(self, x):
#         # Block 1
#         x = self.conv1(x)
#         x = self.bn1(x) # Batch Norm applied BEFORE activation
#         x = self.relu1(x)
#         x = self.pool1(x)

#         # Block 2
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)

#         # Flatten & Classify
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu3(x)
#         x = self.dropout(x)
#         x = self.fc2(x)

#         return x

# if __name__ == "__main__":
#     # Sanity Check
#     model = LyaCNN()
#     dummy_input = torch.randn(8, 1, 2499)
#     output = model(dummy_input)
#     print("Output shape:", output.shape)

# import torch
# import torch.nn as nn

# class LyaCNN(nn.Module):
#     def __init__(self):
#         super(LyaCNN, self).__init__()

#         # Block 1: Detect small-scale features (sharp absorption lines?)
#         # In_channels = 1 (optical depth), Out_channels = 16, Kernel=5
#         #Input length: (Batch, 1, 2499)
#         # After Conv1d: (Batch, 16, 2499)
#         # After MaxPool1d: (Batch, 16, 1249)
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(kernel_size=2) # Downsample by factor of 2

#         # Block 2: Capture larger patterns
#         #Input length: (Batch, 16, 1249)
#         # After Conv1d: (Batch, 32, 1249) increasing channels (depth) to capture more complex features
#         # After MaxPool1d: (Batch, 32, 624)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool1d(kernel_size=2)

#         # Fully connected layers (classifier)
#         # Input length = 2499 // 4 = 624 (after two poolings)
#         # So input features = 32 * 624 = 19968
#         self.flattened_size = 32 * 624
#         self.fc1 = nn.Linear(self.flattened_size, 64)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(64, 1) # Binary classification (EX0 vs EX1)

#     def forward(self, x):
#         # forward pass through nn layers
#         #pass input through conv block 1
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#         #pass input through conv block 2
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
#         #flatten for fully connected layers
#         #reshape from (Batch, 32, 624) to (Batch, 32*624)
#         x = x.view(x.size(0), -1)
#         #pass through the linear layers
#         x = self.fc1(x)
#         x = self.relu3(x)
#         x = self.fc2(x)

#         return x

# if __name__ == "__main__":
#     # Test the model with dummy input
#     model = LyaCNN()
#     dummy_input = torch.randn(8, 1, 2499) # Batch size of 8
#     output = model(dummy_input)
#     print("Input shape:", dummy_input.shape) # Should be (8, 1, 2499)
#     print("Output shape:", output.shape) # Should be (8, 1)
