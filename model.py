import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet34_Weights, ResNet18_Weights

class ResNetForMNIST(nn.Module):
    def __init__(self, num_classes: int = 10, arch: str = "resnet18",
                 freeze_backbone: bool = True, use_pretrained: bool = True):
        super().__init__()
        if use_pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1 if arch == "resnet34" else ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        if arch == "resnet34":
            self.backbone = resnet34(weights=weights)
        else:
            self.backbone = resnet18(weights=weights)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)
