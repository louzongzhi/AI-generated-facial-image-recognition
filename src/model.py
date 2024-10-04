import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.vision.models import resnet101
import numpy as np
import pandas as pd

class ResNet101(nn.Layer):
    def __init__(self, num_classes=2):
        super(ResNet101, self).__init__()
        self.backbone = resnet101(pretrained=True)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

def build_model():
    model = ResNet101(num_classes=2)
    model = paddle.Model(model)
    return model

def train(model, data, epochs=2, batch_size=32, verbose=1):
    model.prepare(
        paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 5))
    )
    model.fit(data, epochs=2, batch_size=32, verbose=1)
    return model
