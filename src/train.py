import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import self_net
from data import RealFakeDataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
multiprocessing.set_start_method('spawn')


def plot_acc(test_acc_list):
    plt.plot(test_acc_list)
    plt.title('Test Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the test images: {100 * correct / total}%')

def train(model, train_loader, val_loader, num_epochs, device, writer, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
        
        val_loss, val_correct, val_total = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = val_correct / val_total
        writer.add_scalar('Validation/Loss', val_loss / len(val_loader), epoch)
        writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Validation Accuracy: {val_accuracy:.4f}')
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            save_model(model, os.path.join(save_path, 'best_model.pth'))
            print(f'Saving best model with accuracy: {best_acc:.4f}')
    
    return best_acc

def main():
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 100
    num_classes = 2
    cropSize = 224
    num_workers = 4
    isTrain = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = 'data'
    data_loader = RealFakeDataLoader(data_path, cropSize, batch_size, num_workers)
    train_loader = data_loader.train_dataloader
    val_loader = data_loader.val_dataloader

    net = self_net(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    writer = SummaryWriter()
    save_path = 'src'
    best_acc = train(net, train_loader, val_loader, num_epochs, device, writer, save_path)
    writer.close()

    plot_acc([best_acc])

    save_model(net, os.path.join(save_path, 'final_model.pth'))

    test(net, test_loader, device)


if __name__ == '__main__':
    main()