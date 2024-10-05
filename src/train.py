import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.metric import Accuracy
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize, RandomHorizontalFlip, RandomRotation
from paddle.vision.datasets import ImageFolder
from paddle.io import DataLoader
from paddle.metric import Accuracy
from paddle.optimizer import Adam
from paddle.metric import Accuracy
from PIL import Image
from paddle.io import Dataset

from model import *

def preprocess(img):
    transform = Compose([
        Resize(size=(224, 224)),
        Normalize([0.5] * 3, [0.5] * 3, data_format='HWC'),
        RandomHorizontalFlip(0.5),
        RandomRotation(0.5),
        Transpose(),
    ])
    img = transform(img).astype("float32")
    return img

def load_dataset(root_path, label):
    data_list = []
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.endswith('png'):
                data_list.append([os.path.join(root, f), label])
    return data_list

class FaceDataset(Dataset):
    def __init__(self, data, is_val=False):
        super().__init__()
        split_index = int(len(data) * 0.8)
        self.samples = data[split_index:] if is_val else data[:split_index]

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = preprocess(img)
        label = np.array([label], dtype="int64")
        return img, label

    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    fake_path = 'data/1/'
    real_path = 'data/0/'

    fake_list = load_dataset(fake_path, label=1)
    real_list = load_dataset(real_path, label=0)

    data_list = fake_list + real_list

    train_set = FaceDataset(data_list, is_val=False)
    eval_set = FaceDataset(data_list, is_val=True)

    model = build_model()
    model.prepare(
        optimizer=paddle.optimizer.AdamW(learning_rate=1e-5, parameters=model.parameters()),
        loss=paddle.nn.CrossEntropyLoss(),
        metrics=paddle.metric.Accuracy()
    )

    model.fit(
        train_data=train_set,
        eval_data=eval_set,
        batch_size=64,
        epochs=100,
        log_freq=1,
    )
    
    os.makedirs('src', exist_ok=True)
    model.save('src/model')
