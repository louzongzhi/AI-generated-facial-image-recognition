import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class self_net(nn.Module):
    def __init__(self, num_classes):
        super(self_net, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        self.densenet = models.densenet161(pretrained=True)

        self.resnet.fc = nn.Identity()
        self.densenet.classifier = nn.Identity()

        self.ca_resnet = ChannelAttention(num_channels=2048)
        self.ca_densenet = ChannelAttention(num_channels=2208)

        self.fusion_layer1 = nn.Sequential(
            nn.Conv2d(4352, 2048, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        self.ca_fusion1 = ChannelAttention(num_channels=2048)
        
        self.fusion_layer2 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.ca_fusion2 = ChannelAttention(num_channels=1024)

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        resnet_features = self.resnet(x)
        densenet_features = self.densenet(x)

        resnet_features = self.ca_resnet(resnet_features)
        densenet_features = self.ca_densenet(densenet_features)

        fused_features1 = torch.cat((resnet_features, densenet_features), dim=1)
        fused_features1 = self.fusion_layer1(fused_features1)
        fused_features1 = self.ca_fusion1(fused_features1)

        fused_features2 = self.fusion_layer2(fused_features1)
        fused_features2 = self.ca_fusion2(fused_features2)

        pooled_features = F.adaptive_avg_pool2d(fused_features2, (1, 1))

        flattened_features = torch.flatten(pooled_features, 1)
        out = self.classifier(flattened_features)
        return out
