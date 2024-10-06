import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101, resnet152


class self_net(nn.Module):
    def __init__(self, num_classes=2):
        super(self_net, self).__init__()
        self.num_classes = num_classes

        self.backbone = resnet50(pretrained=True)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)
