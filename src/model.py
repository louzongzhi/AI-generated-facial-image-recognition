import torch
import torch.nn as nn
from torchvision.models import resnet152
from efficientnet_pytorch import EfficientNet
from torchvision.models import inception_v3
import torchvision.models as models


#---------------------------------------------------------------------------------------------------------------------#

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.relu(self.fc(x))


#---------------------------------------------------------------------------------------------------------------------#

class EfficientNetPath(nn.Module):
    def __init__(self):
        super(EfficientNetPath, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        self.model._fc = None  # 移除最后一个全连接层

    def forward(self, x):
        return self.model(x)


class ResNetPath(nn.Module):
    def __init__(self):
        super(ResNetPath, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.fc = None  # 移除最后一个全连接层

    def forward(self, x):
        return self.model(x)


class InceptionPath(nn.Module):
    def __init__(self):
        super(InceptionPath, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.model.AuxLogits.fc = None  # 移除辅助分类器
        self.model.fc = None  # 移除最后一个全连接层

    def forward(self, x):
        # Inception V3 needs to be resized before passing through the model
        if x.size()[2] != 299 or x.size()[3] != 299:
            x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return self.model(x)


#---------------------------------------------------------------------------------------------------------------------#

class PathSpecificLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PathSpecificLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


#---------------------------------------------------------------------------------------------------------------------#

class self_net(nn.Module):
    def __init__(self, num_classes=2):
        super(self_net, self).__init__()
        self.num_classes = num_classes

        self.efficientnet_path = EfficientNetPath()
        self.resnet_path = ResNetPath()
        self.inception_path = InceptionPath()
        self.path_specific_efficientnet = PathSpecificLayer(1280, 512)
        self.path_specific_resnet = PathSpecificLayer(2048, 512)
        self.path_specific_inception = PathSpecificLayer(2048, 512)

        self.early_fusion_conv = nn.Conv2d(512 * 3, 1024, kernel_size=1)

        self.attention = nn.Sequential(
            nn.Linear(1024 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax(dim=1)
        )

        self.late_fusion_fc = nn.Linear(1024 * 3, 512)

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        efficientnet_features = self.path_specific_efficientnet(self.efficientnet_path(x))
        resnet_features = self.path_specific_resnet(self.resnet_path(x))
        inception_features = self.path_specific_inception(self.inception_path(x))

        # 早期融合
        early_fused_features = torch.cat((efficientnet_features, resnet_features, inception_features), dim=1)
        early_fused_features = self.early_fusion_conv(early_fused_features)

        # 注意力机制
        feature_vector = F.adaptive_avg_pool2d(early_fused_features, (1, 1)).view(early_fused_features.size(0), -1)
        attention_weights = self.attention(feature_vector)
        attention_applied_features = torch.stack([efficientnet_features * attention_weights[:, 0:1],
                                                  resnet_features * attention_weights[:, 1:2],
                                                  inception_features * attention_weights[:, 2:3]], dim=1)
        attention_applied_features = attention_applied_features.sum(dim=1)

        # 晚期融合
        late_fused_features = F.adaptive_avg_pool2d(attention_applied_features, (1, 1)).view(attention_applied_features.size(0), -1)
        late_fused_features = self.late_fusion_fc(late_fused_features)

        # 分类
        out = self.classifier(late_fused_features)
        return out
