import paddle
import paddle.vision.transforms as T
import paddle.vision.transforms.functional as F
import paddle.vision.datasets as datasets
from paddle.io import DataLoader
import numpy as np
import pandas as pd
from model import *

model = build_model()

transform = T.Compose(
    [
        T.Resize(224),
        T.Transpose(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Half()
    ]
)
train_dataset = paddle.vision.datasets.ImageFolder('data', transform=transform)
test_dataset = paddle.vision.datasets.ImageFolder('data', transform=transform)
train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = paddle.io.DataLoader(test_dataset, batch_size=32, shuffle=False)