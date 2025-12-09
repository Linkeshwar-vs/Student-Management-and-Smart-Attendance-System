# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class CDCNNConv2d(nn.Conv2d):
    """Central Difference Convolution Layer"""
    def __init__(self, *args, theta=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta = theta

    def forward(self, x):
        # Standard conv
        out_normal = super().forward(x)
        if self.theta == 0:
            return out_normal

        # Central difference term
        kernel_diff = self.weight.sum(dim=(2, 3), keepdim=True)
        out_diff = F.conv2d(x, kernel_diff, bias=None, stride=self.stride,
                            padding=0, groups=self.groups)

        return out_normal - self.theta * out_diff


class ResNet_CDCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base_model = resnet18(pretrained=True)

        # Replace first conv layer with CDCNN
        base_model.conv1 = CDCNNConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.features = nn.Sequential(*list(base_model.children())[:-1])  # remove fc
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
