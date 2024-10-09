from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def get_mean_and_std(train_data):
    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=False, num_workers=4,
        pin_memory=True)
    mean = 0.0
    std = 0.0
    for images, _ in train_loader:
        batch_mean = images.mean(dim=[0, 2, 3])
        batch_std = images.std(dim=[0, 2, 3])
        mean += batch_mean * images.size(0)
        std += batch_std * images.size(0)
    mean /= len(train_data)
    std /= len(train_data)
    return mean.tolist(), std.tolist()

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = ImageFolder(root=r'data/', transform=transform) # 替换为测试集再跑一次，分别记录
    mean, std = get_mean_and_std(train_dataset)
    print(f"Mean: {mean}")
    print(f"Std: {std}")
