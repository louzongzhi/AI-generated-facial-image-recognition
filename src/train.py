import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import self_net
from data import RealFakeDataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import time

def train(model, train_loader, val_loader, num_epochs, device, writer, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0.0
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
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
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        train_loss_list.append(running_loss / len(train_loader))
        train_acc_list.append(correct / total)
        print('Epoch %d training Accuracy: %.3f' % (epoch + 1, correct / total))
        writer.add_scalar('Train/Loss', running_loss / len(train_loader), epoch)
        writer.add_scalar('Train/Accuracy', correct / total, epoch)
        writer.add_histogram('Train/Outputs', outputs, epoch)
        writer.add_histogram('Train/Labels', labels, epoch)
        writer.add_histogram('Train/Predicted', predicted, epoch)

    for i, data in enumerate(test_loader, 0):

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    test_acc_list.append(correct / total)
    print('Epoch %d testing Accuracy: %.3f' % (epoch + 1, correct / total))
    writer.add_scalar('Test/Loss', running_loss / len(test_loader), epoch)
    writer.add_scalar('Test/Accuracy', correct / total, epoch)
    writer.add_histogram('Test/Outputs', outputs, epoch)
    writer.add_histogram('Test/Labels', labels, epoch)
    writer.add_histogram('Test/Predicted', predicted, epoch)

    if correct / total > best_acc:
        best_acc = correct / total
        torch.save(net.state_dict(), 'best_model.pth')
        print('Saving..')
        print('Best Accuracy: %.3f' % best_acc)

    return test_acc_list


if __name__ == '__main__':
    # Hyper Parameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 100
    num_classes = 2
    num_channels = 3
    cropSize = 224
    num_workers = 4
    isTrain = True

    # Data
    data_path = 'data'
    data_loader = RealFakeDataLoader(data_path, cropSize, batch_size, num_threads, validation_split)
    train_loader, test_loader = data_loader.create_dataloader(isTrain=isTrain)

    # Model
    net = self_net(num_classes=num_classes)
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Training
    best_acc = 0.0
    test_acc_list = train(net, criterion, optimizer, train_loader, test_loader, num_epochs, best_acc)
    print(test_acc_list)
    # Plot
    plot_acc(test_acc_list)
    # Save
    save_model(net, 'src/model.pth')
    # Test
    test(net, test_loader)