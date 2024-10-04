import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.vision.models import resnet101
import numpy as np
import pandas as pd


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Layer):
    default_act = nn.Silu()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2D(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias_attr=False)
        self.bn = nn.BatchNorm2D(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Layer) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C3k2(nn.Layer):
    def __init__(self, c1, c2=False, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1) if c2 else nn.Identity()
        self.m = nn.Sequential(*[Conv(c_, c_, 3, 1) for _ in range(n)])
        #self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(paddle.concat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C2PSA(nn.Layer):
    def __init__(self, c1):
        super().__init__()
        pass

    def forward(self, x):
        return x


class Classify(nn.Layer):
    def __init__(self, c1, num_classes):
        super().__init__()
        self.cls = nn.Linear(c1, num_classes)

    def forward(self, x):
        x = paddle.mean(x, axis=(2, 3))
        return self.cls(x)


class self_net(nn.Layer):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            Conv(3, 64, 3, 2),
            Conv(64, 128, 3, 2),
            C3k2(128, 256, 2, False, 0.25),
            Conv(256, 256, 3, 2),
            C3k2(256, 512, 2, False, 0.25),
            Conv(512, 512, 3, 2),
            C3k2(512, 512, 2, True),
            Conv(512, 1024, 3, 2),
            C3k2(1024, 1024, 2, True),
            C2PSA(1024),
        )
        self.head = Classify(1024, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
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
