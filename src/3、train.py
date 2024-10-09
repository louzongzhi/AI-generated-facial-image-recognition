import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import warnings
from timm.utils import accuracy, AverageMeter, ModelEma
from sklearn.metrics import classification_report
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
# from models.efficientformer_v2 import efficientformerv2_s0
from timm.models.efficientformer_v2 import efficientformerv2_s0
from timm.models.efficientformer_v2 import efficientformerv2_l
from torchvision import datasets

torch.backends.cudnn.benchmark = False
os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')


#--------------------------------------------------------------------------------------------------#

def train(model, device, train_loader, optimizer, epoch,model_ema):
    model.train()

    loss_meter, acc1_meter, acc5_meter = AverageMeter(), AverageMeter(), AverageMeter()
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        samples, targets = mixup_fn(data, target)
        output = model(samples)
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = torch.nan_to_num(criterion_train(output, targets))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion_train(output, targets)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            loss.backward()
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss_meter.update(loss.item(), target.size(0))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch,
                (batch_idx + 1) * train_loader.batch_size,
                len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                loss.item(),
                lr
            )
        )

    ave_loss =loss_meter.avg
    acc = acc1_meter.avg
    print('epoch:{} \t loss:{:.2f} \t acc:{:.2f}'.format(epoch, ave_loss, acc))

    return ave_loss, acc


#--------------------------------------------------------------------------------------------------#

@torch.no_grad()
def val(model, device, test_loader):
    global Best_ACC
    model.eval()

    loss_meter, acc1_meter, acc5_meter = AverageMeter(), AverageMeter(), AverageMeter()
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))

    val_list = []
    pred_list = []
    for data, target in test_loader:
        for t in target:
            val_list.append(t.data.item())

        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        loss = criterion_val(output, target)
        _, pred = torch.max(output.data, 1)

        for p in pred:
            pred_list.append(p.data.item())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
    acc = acc1_meter.avg
    print(
        ' \n Val set: Average loss: {:.4f} \t Acc1:{:.3f}% \t Acc5:{:.3f}% \n '.format(
            loss_meter.avg,
            acc,
            acc5_meter.avg
        )
    )

    if acc > Best_ACC:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module, file_dir + '/' + 'model.pth')
        else:
            torch.save(model, file_dir + '/' + 'model.pth')
        Best_ACC = acc
    if isinstance(model, torch.nn.DataParallel):
        state = {

            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'Best_ACC': Best_ACC
        }
        if use_ema:
            state['state_dict_ema'] = model.module.state_dict()
        torch.save(state, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
    else:
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'Best_ACC': Best_ACC
        }
        if use_ema:
            state['state_dict_ema'] = model.state_dict()
        torch.save(state, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
    return val_list, pred_list, loss_meter.avg, acc


#--------------------------------------------------------------------------------------------------#

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#--------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    file_dir = 'src/model'
    if os.path.exists(file_dir):
        print('true')
        os.makedirs(file_dir, exist_ok=True)
    else:
        os.makedirs(file_dir)

    model_lr = 1e-4
    BATCH_SIZE = 16
    EPOCHS = 300
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = True
    use_dp = True
    classes = 2
    resume = None
    CLIP_GRAD = 5.0
    Best_ACC = 0
    use_ema = True
    model_ema_decay = 0.9998
    start_epoch = 1
    seed = 1
    seed_everything(seed)

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.44127703, 0.4712498, 0.43714803], std=[0.18507297, 0.18050247, 0.16784933]) # 要改mean和std
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.44127703, 0.4712498, 0.43714803], std=[0.18507297, 0.18050247, 0.16784933]) # 和上面的mean、std一个数值
    ])
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=0.1,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=classes
    )

    dataset_train = datasets.ImageFolder('dataset/train', transform=transform)
    dataset_test = datasets.ImageFolder("dataset/val", transform=transform_test)
    with open('class.txt', 'w') as file:
        file.write(str(dataset_train.class_to_idx))
    with open('class.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(dataset_train.class_to_idx))
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    criterion_train = SoftTargetCrossEntropy()
    criterion_val = torch.nn.CrossEntropyLoss()

    model_ft = efficientformerv2_l(pretrained=True, pretrained_strict=False)
    print(model_ft)
    # num_fr = model_ft.head.in_features
    # model_ft.head = nn.Linear(num_fr, classes)
    model_ft.set_distilled_training(False)
    model_ft.reset_classifier(classes, global_pool='avg')
    print(model_ft)

    if resume:
        model = torch.load(resume)
        print(model['state_dict'].keys())
        model_ft.load_state_dict(model['state_dict'])
        Best_ACC = model['Best_ACC']
        start_epoch = model['epoch'] + 1
    model_ft.to(DEVICE)

    optimizer = optim.AdamW(model_ft.parameters(), lr=model_lr)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-6)

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    if torch.cuda.device_count() > 1 and use_dp:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_ft = torch.nn.DataParallel(model_ft)
    if use_ema:
        model_ema = ModelEma(
            model_ft,
            decay=model_ema_decay,
            device=DEVICE,
            resume=resume
        )
    else:
        model_ema = None

    is_set_lr = False
    log_dir = {}
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []

    if resume and os.path.isfile(file_dir + "result.json"):
        with open(file_dir + 'result.json', 'r', encoding='utf-8') as file:
            logs = json.load(file)
            train_acc_list = logs['train_acc']
            train_loss_list = logs['train_loss']
            val_acc_list = logs['val_acc']
            val_loss_list = logs['val_loss']
            epoch_list = logs['epoch_list']

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_list.append(epoch)
        log_dir['epoch_list'] = epoch_list
        train_loss, train_acc = train(model_ft, DEVICE, train_loader, optimizer, epoch, model_ema)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        log_dir['train_acc'] = train_acc_list
        log_dir['train_loss'] = train_loss_list

        if use_ema:
            val_list, pred_list, val_loss, val_acc = val(model_ema.ema, DEVICE, test_loader)
        else:
            val_list, pred_list, val_loss, val_acc = val(model_ft, DEVICE, test_loader)

        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        log_dir['val_acc'] = val_acc_list
        log_dir['val_loss'] = val_loss_list
        log_dir['best_acc'] = Best_ACC

        with open(file_dir + '/result.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(log_dir))
        print(classification_report(val_list, pred_list, target_names=dataset_train.class_to_idx))

        if epoch < 600:
            cosine_schedule.step()
        else:
            if not is_set_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 1e-6
                    is_set_lr = True

        fig = plt.figure(1)
        plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
        plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
        plt.legend(["Train Loss", "Val Loss"], loc="upper right")
        plt.xlabel(u'epoch')
        plt.ylabel(u'loss')
        plt.title('Model Loss ')
        plt.savefig(file_dir + "/loss.png")
        plt.close(1)

        fig2 = plt.figure(2)
        plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
        plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
        plt.legend(["Train Acc", "Val Acc"], loc="lower right")
        plt.title("Model Acc")
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.savefig(file_dir + "/acc.png")
        plt.close(2)
