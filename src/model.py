import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

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


class Bottleneck(nn.Layer):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(nn.Layer):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.m(x)


class C3k2(nn.Layer):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        if c3k:
            self.m = nn.Sequential(*[C3k(c_, c_, 2, shortcut, g) for _ in range(n)])
        else:
            self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g) for _ in range(n)])

    def forward(self, x):
        y1, y2 = paddle.split(self.cv1(x), 2, axis=1)
        for m in self.m:
            y1 = m(y1)
        return self.cv2(paddle.concat([y2, y1], axis=1))


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        qkv = qkv.reshape([B, N + N, self.key_dim * 2 + self.head_dim]).transpose([0, 2, 1])
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], axis=1)
        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = F.softmax(attn, axis=-1)
        attn = paddle.matmul(attn, v)
        attn = attn.transpose([0, 2, 1]).reshape([B, H, W, self.head_dim * self.num_heads])
        attn = attn.transpose([0, 3, 1, 2])
        x = self.proj(attn) + self.pe(x)
        return x

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


class C2PSA(nn.Layer):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = make_divisible(c2 * 0.5, 8)
        self.conv = Conv(c1, c_, k, s, autopad(k, p, d), g, d, act)
        self.psa = Attention(c_)

    def forward(self, x):
        x = self.conv(x)
        x = self.psa(x)
        return x


class Classify(nn.Layer):
    def __init__(self, c1, num_classes):
        super().__init__()
        self.fc = nn.Linear(c1, num_classes)

    def forward(self, x):
        if len(x.shape) == 4:
            x = paddle.mean(x, axis=[2, 3])
        return self.fc(x)


class self_net(nn.Layer):
    def __init__(self, num_classes=2):
        super().__init__()
        c1, c2, c3, c4, c5 = 64, 128, 256, 512, 1024

        # backbone
        self.x_0 = Conv(3, c1, 3, 2)
        self.x_1 = Conv(c1, c2, 3, 2)
        self.x_2 = C3k2(c2, c3, 2, False, 0.25)
        self.x_3 = Conv(c3, c3, 3, 2)
        self.x_4 = C3k2(c3, c4, 2, False, 0.25)
        self.x_5 = Conv(c4, c4, 3, 2)
        self.x_6 = C3k2(c4, c4, 2, True)
        self.x_7 = Conv(c4, c5, 3, 2)
        self.x_8 = C3k2(c5, c5, 2, True)
        self.x_9 = C2PSA(c5, c5)

        # neck

        # head
        self.final = Classify(c5, num_classes)

    def forward(self, x):
        # backbone
        x = self.x_0(x)
        x = self.x_1(x)
        x = self.x_2(x)
        x = self.x_3(x)
        x = self.x_4(x)
        x = self.x_5(x)
        x = self.x_6(x)
        x = self.x_7(x)
        x = self.x_8(x)
        x = self.x_9(x)

        # neck
        pass

        # head
        x = self.final(x)

        return x

def build_model():
    model = self_net(num_classes=2)
    model = paddle.Model(model)
    return model

def train(model, data, epochs=100, batch_size=64, verbose=1):
    model.prepare(
        paddle.optimizer.Adam(
            learning_rate=0.001,
            parameters=model.parameters(),
        ),
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 5))
    )
    model.fit(
        data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    return model
