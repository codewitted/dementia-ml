import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMRI2DCNN(nn.Module):
    """
    A simple 2D CNN for single-channel MRI slices, outputting class logits.
    """
    def __init__(self, num_classes=2):
        super(SimpleMRI2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # (batch, 1, H, W) -> (batch, 16, H, W)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # assumes input image size is 128x128
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, H, W)
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 16, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 32, H/4, W/4)
        x = self.pool(F.relu(self.conv3(x)))  # (batch, 64, H/8, W/8)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)            # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Example usage:
# model = SimpleMRI2DCNN(num_classes=2)
