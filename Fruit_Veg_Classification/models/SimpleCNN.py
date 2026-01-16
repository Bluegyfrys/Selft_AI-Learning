import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a simple Convolutional Neural Network
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1) # 输入通道数为3（RGB），输出通道数为16 224 224
        self.conv2 = nn.Conv2d(16, 32,  kernel_size=3, padding=1)   # 112 112
        self.conv3 = nn.Conv2d(32, 64,  kernel_size=3, padding=1)   # 56 56
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   # 28 28
        self.fc1 = nn.Linear(128 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x