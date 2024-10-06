import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time
import random
import math
from tqdm import tqdm
from PIL import Image
from utils import weights_init_normal

from data import RealFakeDataLoader
from model import self_net


data_path = 'data'
cropSize = 256
batch_size = 32
num_threads = 4
validation_split = 0.2

data_loader = RealFakeDataLoader(data_path, cropSize, batch_size, num_threads, validation_split)
train_loader = data_loader.train_dataloader
val_loader = data_loader.val_dataloader

model = self_net().to(device="cuda" if torch.cuda.is_available() else "cpu")