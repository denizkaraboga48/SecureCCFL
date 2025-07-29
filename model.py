# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)   # 20 parametre
        self.conv2 = nn.Conv2d(2, 2, kernel_size=3, padding=1)   # 38 parametre
        self.fc = nn.Linear(2, 10)  # 2*10 + 10 = 30

    def forward(self, x):
        x = F.relu(self.conv1(x))       # -> [batch, 2, 28, 28]
        x = F.max_pool2d(x, 2)          # -> [batch, 2, 14, 14]
        x = F.relu(self.conv2(x))       # -> [batch, 2, 14, 14]
        x = F.adaptive_avg_pool2d(x, 1) # -> [batch, 2, 1, 1]
        x = x.view(x.size(0), -1)       # -> [batch, 2]
        x = self.fc(x)                  # -> [batch, 10]
        return x

