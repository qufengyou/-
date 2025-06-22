import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models  # 新增models
import matplotlib.pyplot as plt
from model_cnn import create_model
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def get_transforms(train=True):
    """增强版数据增强"""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
            transforms.RandomRotation(15),  # 随机旋转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def train_model(train_dir, num_epochs=50, batch_size=16, learning_rate=0.001, device='cuda'):
    # 数据加载
    full_dataset = ImageFolder(train_dir, transform=get_transforms(train=True))

    # 数据集分割
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 创建模型（使用预训练模型）
    # model = models.resnet18(pretrained=True)  # 使用预训练模型
    # model.fc = nn.Linear(model.fc.in_features, 10)  # 修改最后一层
    # model = model.to(device)
    model = create_model(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam([
    #      {'params': model.layer4.parameters(), 'lr': learning_rate / 10},  # 深层使用更小学习率
    #      {'params': model.fc.parameters(), 'lr': learning_rate}
    # ], weight_decay=1e-4)  # 添加权重衰减
    optimizer = optim.Adam(model.parameters(),lr = learning_rate, weight_decay=1e-4)

    # 学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # 训练记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # 早停参数
    best_val_acc = 0.0
    patience = 7
    no_improve = 0

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f'Train Epoch {epoch + 1}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Val Epoch {epoch + 1}', leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # 计算指标
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'best_model12.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'\nEarly stopping at epoch {epoch + 1}')
                break

        # 更新学习率
        scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印日志
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%')
        print('-' * 60)

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    return history


if __name__ == '__main__':
    # 固定随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # 训练参数
    train_dir = 'c:/abc/files1/python_files/1/code_base_cnn_more_stations/state-farm-distracted-driver-detection/imgs/train'  # 修改为实际路径
    num_epochs = 50
    batch_size = 32  # 适当增大batch_size
    learning_rate = 0.0005  # 更小的初始学习率
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 开始训练
    history = train_model(
        train_dir=train_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )