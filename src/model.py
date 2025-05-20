import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 29 * 29, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 sınıf varsa: ['glioma', 'meningioma', 'pituitary', 'no_tumor']

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch, 16, 62, 62]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch, 32, 29, 29]
        x = x.view(-1, 32 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
