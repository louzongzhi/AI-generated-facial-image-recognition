import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

from model import self_net

def main():
    parser = argparse.ArgumentParser(description='Train a simple neural network')
    parser.add_argument('--datadir', type=str, default='data', help='path to dataset')
    parser.add_argument('--checkpointdir', type=str, default='checkpoints', help='path to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=args.datadir, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    model = self_net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(train_loader):.4f}')

        if not os.path.exists(args.checkpointdir):
            os.makedirs(args.checkpointdir)
        torch.save(model.state_dict(), os.path.join(args.checkpointdir, f'checkpoint_epoch{epoch+1}.pth'))

if __name__ == '__main__':
    main()
