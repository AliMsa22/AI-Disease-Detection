import torch
import torch.nn as nn
import torch.nn.functional as F

class ChestXrayCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(ChestXrayCNN, self).__init__()

        # First convolutional layer (input: 3 channels, output: 32 channels)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling to reduce spatial dimensions

        # Second convolutional layer (input: 32 channels, output: 64 channels)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer (input: 64 channels, output: 128 channels)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Adjust size based on image size after pooling
        self.fc2 = nn.Linear(512, num_classes)  # Output layer (num_classes is 7 for your dataset)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten the output 
        x = x.view(x.size(0), -1)

        # Fully connected layers 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Apply softmax for multi-class classification
        x = F.log_softmax(x, dim=1)

        return x
