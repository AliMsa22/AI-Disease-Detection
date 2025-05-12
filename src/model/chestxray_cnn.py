import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicDropout(nn.Module):
    def __init__(self, initial_rate=0.2, final_rate=0.5, total_epochs=100):
        super(DynamicDropout, self).__init__()
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.total_epochs = total_epochs

    def forward(self, x, epoch):
        # Linearly interpolate the dropout rate based on the current epoch
        current_rate = self.initial_rate + (self.final_rate - self.initial_rate) * (epoch / self.total_epochs)
        return F.dropout(x, p=current_rate, training=self.training)

class ChestXrayCNN(nn.Module):
    def __init__(self, num_classes=7, initial_dropout=0.2, final_dropout=0.5, total_epochs=50):
        super(ChestXrayCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Dynamic dropout
        self.dynamic_dropout = DynamicDropout(initial_rate=initial_dropout, final_rate=final_dropout, total_epochs=total_epochs)

        self.fc1 = nn.Linear(512 * 16 * 16, 512)  # Adjust input size based on pooling
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, epoch):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.01))

        # Apply dynamic dropout
        x = self.dynamic_dropout(x, epoch)

        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

