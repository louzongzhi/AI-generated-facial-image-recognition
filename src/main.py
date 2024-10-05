import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import self_net

test_data = datasets.ImageFolder(root='./testdata', transform=transforms.ToTensor())

model = self_net()
model.load_state_dict(torch.load('./src/model.pth'))

model.eval()

with torch.no_grad():
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    for images, _ in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

with open('./cla_pre.csv', 'w', encoding='utf-8') as f:
    for i, label in enumerate(predicted):
        f.write(f'{test_data.imgs[i][0].stem},{label.item()}\n')
