import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import warnings
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.utils import accuracy, AverageMeter, ModelEma
from timm.models.efficientformer_v2 import efficientformerv2_l
from timm.loss import SoftTargetCrossEntropy
import os
import matplotlib.pyplot as plt
from timm.utils import accuracy, AverageMeter, ModelEma
from sklearn.metrics import classification_report
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.models.efficientformer_v2 import efficientformerv2_l
from torchvision import datasets

# 环境设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")

# 设置参数
BATCH_SIZE = 128
EPOCHS = 100
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE_DIR = './src/model'
DATA_ROOT = './dataset'

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomCrop(32, padding=4)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

mixup_fn = Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
    prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1,
    num_classes=2
)

# 加载训练和验证数据集
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_ROOT, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(root=os.path.join(DATA_ROOT, 'val'), transform=transform_test)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 初始化模型
model = efficientformerv2_l(num_classes=2).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion_train = SoftTargetCrossEntropy()
criterion_val = nn.CrossEntropyLoss()

# 使用EMA
model_ema = ModelEma(model, decay=0.9998)

# 设置随机种子
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 训练函数
def train(model, device, train_loader, optimizer, epoch, model_ema):
    model.train()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    total_num = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        samples, targets = mixup_fn(data, target)
        output = model(samples)
        optimizer.zero_grad()
        loss = criterion_train(output, targets)
        loss.backward()
        optimizer.step()
        if model_ema is not None:
            model_ema.update(model)
        loss_meter.update(loss.item(), target.size(0))
        acc1, _ = accuracy(output, target, topk=(1, 5))
        acc1_meter.update(acc1.item(), target.size(0))
        if (batch_idx + 1) % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * train_loader.batch_size}/{total_num} ({100. * batch_idx / len(train_loader):.0f}%)] \t Loss: {loss.item()}')
    ave_loss = loss_meter.avg
    acc = acc1_meter.avg
    print(f'epoch:{epoch} \t loss:{ave_loss:.2f} \t acc:{acc:.2f}')
    return ave_loss, acc

# 验证函数
@torch.no_grad()
def val(model, device, test_loader):
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    total_num = len(test_loader.dataset)
    val_list = []
    pred_list = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion_val(output, target)
        _, pred = torch.max(output.data, 1)
        val_list.extend(target.data.tolist())
        pred_list.extend(pred.data.tolist())
        loss_meter.update(loss.item(), target.size(0))
        acc1, _ = accuracy(output, target, topk=(1, 5))
        acc1_meter.update(acc1.item(), target.size(0))
    acc = acc1_meter.avg
    print(f'Val set: Average loss: {loss_meter.avg:.4f} \t Acc1:{acc:.3f}%')
    if acc > Best_ACC:
        torch.save(model, FILE_DIR + '/model.pth')
        Best_ACC = acc
    return val_list, pred_list, loss_meter.avg, acc

# 训练和验证
def main():
    global FILE_DIR
    os.makedirs(FILE_DIR, exist_ok=True)
    best_acc = 0.0
    epoch_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, DEVICE, train_loader, optimizer, epoch, model_ema)
        val_list, pred_list, val_loss, val_acc = val(model, DEVICE, test_loader)

        # 保存准确率
        epoch_list.append(epoch)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        # 保存最好模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), FILE_DIR + '/best.pth')

    # 绘制准确率曲线
    plt.plot(epoch_list, train_acc_list, 'r-', label='Train Acc')
    plt.plot(epoch_list, val_acc_list, 'b-', label='Val Acc')
    plt.legend(["Train Acc", "Val Acc"], loc="lower right")
    plt.title("Model Acc")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(FILE_DIR + '/acc.png')
    plt.close()

if __name__ == '__main__':
    main()