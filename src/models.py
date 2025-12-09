import torch
import torch.nn as nn

class LyaCNN(nn.Module):
    def __init__(self):
        super(LyaCNN, self).__init__()
        
        # --- BLOCK 1 ---
        # Increased kernel size to 11 to see "absorption lines" better
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(16) # Batch Norm to stabilize training
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4) # Aggressive pooling to reduce dimensionality
        
        # --- BLOCK 2 ---
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, padding=5)
        self.bn2 = nn.BatchNorm1d(32) # Batch Norm
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        
        # --- CLASSIFIER ---
        # Calculation: 
        # Input: 2499
        # Pool1 (/4) -> ~624
        # Pool2 (/4) -> ~156
        self.flattened_size = 32 * 156
        
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.1) #prevent overfitting
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x) # Batch Norm applied BEFORE activation
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten & Classify
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Sanity Check
    model = LyaCNN()
    dummy_input = torch.randn(8, 1, 2499)
    output = model(dummy_input)
    print("Output shape:", output.shape)

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