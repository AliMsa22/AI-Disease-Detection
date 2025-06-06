import torch.nn as nn

class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.flatten = nn.Flatten()
        self.fc_block = nn.Sequential(
            nn.Linear(128 * 30 * 30, 512), nn.ReLU(),
            nn.Linear(512, 100), nn.ReLU(),
            nn.Linear(100, 1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        return self.fc_block(x)