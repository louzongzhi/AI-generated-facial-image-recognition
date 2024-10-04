import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T


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
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        if c3k:
            self.m = nn.Sequential(*[C3k(c_, c_, 2, shortcut, g) for _ in range(n)])
        else:
            self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g) for _ in range(n)])

    def forward(self, x):
        y = paddle.split(self.cv1(x), 2, axis=1)
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(paddle.concat(y, axis=1))


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
        qkv = qkv.reshape([B, self.num_heads, -1, N])
        q, k, v = paddle.split(qkv, [self.key_dim, self.key_dim, self.head_dim], axis=2)

        attn = paddle.matmul(q.transpose([0, 1, 3, 2]), k) * self.scale
        attn = F.softmax(attn, axis=-1)
        x = paddle.matmul(v, attn.transpose([0, 1, 3, 2])).reshape([B, C, H, W]) + self.pe(x)
        x = self.proj(x)
        return x


class PSABlock(nn.Layer):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Layer):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2, "C2PSA requires c1 and c2 to be the same"
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2D(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2D(2 * self.c, c1, 1, 1)

        self.m = nn.Sequential(*[PSABlock(self.c) for _ in range(n)])

    def forward(self, x):
        a, b = paddle.split(self.cv1(x), [self.c, self.c], axis=1)
        b = self.m(b)
        return self.cv2(paddle.concat([a, b], axis=1))


class Classify(nn.Layer):
    def __init__(self, c1, num_classes):
        super().__init__()
        self.cls = nn.Linear(c1, num_classes)

    def forward(self, x):
        x = paddle.mean(x, axis=[2, 3])
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
            C2PSA(1024, 1024),
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
