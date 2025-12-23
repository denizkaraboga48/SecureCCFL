import torch
import torch.nn as nn
import torch.nn.functional as F
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
    

class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout_p: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def build_model(
    model_name: str,
    num_classes: int = 10,
    freeze_backbone: bool = True,
    use_pretrained: bool = True,
    dropout_p: float = 0.25,
):
    name = model_name.lower().strip()

    if name == "tinycnn":
        return TinyCNN(num_classes=num_classes, dropout_p=dropout_p)

    if name in ("resnet18", "resnet34"):
        return ResNetForMNIST(
            num_classes=num_classes,
            arch=name,
            freeze_backbone=freeze_backbone,
            use_pretrained=use_pretrained,
        )

    raise ValueError(f"Unknown model_name: {model_name}")

