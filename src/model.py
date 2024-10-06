import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152


class self_net(nn.Module):
    def __init__(self, num_classes=2):
        super(self_net, self).__init__()
        self.num_classes = num_classes

        self.backbone = resnet152(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        attention_map = self.attention(x)
        x = x * attention_map
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
