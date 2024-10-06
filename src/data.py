import os
import random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from PIL import Image


class RealFakeDataLoader:
    def __init__(self, data_path, cropSize, batch_size, num_threads, validation_split=0.2, isTrain=True):
        self.data_path = data_path
        self.cropSize = cropSize
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.validation_split = validation_split
        self.isTrain = isTrain

        self.real_list = self.recursively_read(os.path.join(data_path, '0'))
        self.fake_list = self.recursively_read(os.path.join(data_path, '1'))
        self.total_list = self.real_list + self.fake_list
        random.shuffle(self.total_list)

        self.mean, self.std = self.calculate_mean_std()

        if self.isTrain:
            train_size = int((1 - self.validation_split) * len(self.total_list))
            val_size = len(self.total_list) - train_size
            self.train_dataset, self.val_dataset = random_split(self.total_list, [train_size, val_size])
        else:
            self.train_dataset = self.total_list

        self.train_dataloader = self.create_dataloader(self.train_dataset, isTrain=True)
        if self.val_dataset:
            self.val_dataloader = self.create_dataloader(self.val_dataset, isTrain=False)

    def recursively_read(self, rootdir, exts=["png", "jpg", "JPEG", "jpeg"]):
        out = []
        for r, _, f in os.walk(rootdir):
            for file in f:
                if file.split('.')[-1].lower() in exts:
                    out.append(os.path.join(r, file))
        return out

    def create_dataset(self):
        return RealFakeDataset(self.total_list, self.cropSize, self.isTrain, self.mean, self.std)

    def create_dataloader(self, dataset, isTrain):
        if isTrain:
            sampler = self.get_bal_sampler(dataset)
        else:
            sampler = None
        return DataLoader(
            RealFakeDataset(dataset, self.cropSize, isTrain, self.mean, self.std),
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.num_threads
        )

    def get_bal_sampler(self):
        targets = [0 if '0' in path else 1 for path in self.total_list]
        class_sample_count = torch.tensor([len(np.where(targets == t)[0]) for t in np.unique(targets)])
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return WeightedRandomSampler(samples_weight, len(samples_weight))

    def calculate_mean_std(self):
        if self.isTrain:
            transform = transforms.Compose([
                transforms.Resize(self.cropSize),
                transforms.ToTensor(),
            ])
            dataset = RealFakeDataset(self.total_list, self.cropSize, self.isTrain, transform=transform)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_threads)
        else:
            return torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])

        mean = 0.0
        std = 0.0
        nb_samples = 0
        for data, _ in dataloader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        return mean, std


class RealFakeDataset(Dataset):
    def __init__(self, file_list, cropSize, isTrain, mean=None, std=None, transform=None):
        self.file_list = file_list
        self.cropSize = cropSize
        self.isTrain = isTrain
        self.mean = mean
        self.std = std
        if transform is None:
            if self.isTrain:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(self.cropSize),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(self.cropSize),
                    transforms.CenterCrop(self.cropSize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        label = 0 if '0' in img_path else 1
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label
