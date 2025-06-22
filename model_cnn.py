import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- 新增的注意力模块 -----------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 保持原有参数名称，调整压缩逻辑
        reduction = max(4, in_channels // reduction_ratio)  # 防止过小通道压缩
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduction),
            nn.ReLU(),
            nn.Linear(reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 保持原始前向逻辑结构
        b, c = x.size()[:2]
        avg = self.avg_pool(x).view(b, c)
        max_ = self.max_pool(x).view(b, c)
        scale = self.fc(avg) + self.fc(max_)  # 增强差异
        return x * scale.view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):  # 参数不变
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 保持原有实现结构
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))

# ----------------- 修改后的模型 -----------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            ChannelAttention(32),  # 新增通道注意力

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            SpatialAttention(),    # 新增空间注意力
            nn.Dropout(0.5)
        )

        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            ChannelAttention(64),  # 新增通道注意力
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            SpatialAttention(),    # 新增空间注意力（可选）
            nn.Dropout(0.5)
        )

        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            ChannelAttention(128),  # 新增通道注意力
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            SpatialAttention(),      # 新增空间注意力（可选）
            nn.Dropout(0.5)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            ChannelAttention(256),  # 新增通道注意力
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            SpatialAttention(),      # 新增空间注意力（可选）
            nn.Dropout(0.5)
        )

        # 自动计算全连接输入维度）
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            dummy_out = self.forward_features(dummy)
            fc_in = dummy_out.view(1, -1).shape[1]

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def create_model(device=None):
    model = SimpleCNN()
    if device:
        model = model.to(device)
    return model
