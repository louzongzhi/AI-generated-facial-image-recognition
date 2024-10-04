import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.vision.models import resnet101
import numpy as np
import pandas as pd


class self_net(nn.Layer):
    def __init__(self, num_classes=2):
        super(self_net, self).__init__()
        self.backbone = resnet101(pretrained=True)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

def build_model():
    model = self_net(num_classes=2)
    model = paddle.Model(model)
    return model

def train(model, data, epochs=100, batch_size=64, verbose=1):
    model.prepare(
        paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 5))
    )
    model.fit(data, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model
