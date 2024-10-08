import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import self_net

class DatasetLoader:
    def __init__(self, path, batch_size, validation_ratio=0.2):
        self.path = path
        self.batch_size = batch_size
        self.validation_ratio = validation_ratio
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_data(self):
        dataset = datasets.ImageFolder(root=self.path, transform=self.transform)
        dataset_size = len(dataset)
        train_size = int((1 - self.validation_ratio) * dataset_size)
        validation_size = dataset_size - train_size
        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

class Trainer:
    def __init__(self, net, train_loader, validation_loader, criterion, optimizer, device):
        self.net = net
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_accuracy = 0.0
        self.best_model_path = 'src/model.pth'

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.net.train()
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

            self.validate()

    def validate(self):
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.validation_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the validation images: {accuracy}%')
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(self.net.state_dict(), self.best_model_path)
            print(f'Saved new best model with accuracy: {self.best_accuracy}%')

def main():
    path_to_dataset = 'data'
    batch_size = 32
    num_epochs = 10

    dataset_loader = DatasetLoader(path_to_dataset, batch_size)
    dataset_loader.load_data()

    net = self_net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    trainer = Trainer(net, dataset_loader.train_loader, dataset_loader.validation_loader, criterion, optimizer, device)
    trainer.train(num_epochs)

if __name__ == '__main__':
    main()