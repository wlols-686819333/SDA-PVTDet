import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAggregateModule(nn.Module):
    def __init__(self, in_channels, branch_channels=64):
        super().__init__()

        # 初始7x7卷积和BN
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )

        # 第一个分支: 1x1 -> 3x3
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1)
        )

        # 第二个分支: 1x1 -> 1x3 -> 3x1 -> dilated 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=2, dilation=2)
        )

        # 第三个分支: 1x1 -> 3x1 -> 1x3 -> dilated 3x3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=2, dilation=2)
        )

        # 最后的1x1卷积，将concat后的通道数调整回输入通道数
        self.final_conv = nn.Conv2d(branch_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):
        # 7x7卷积
        out = self.conv7x7(x)

        # 三个并行分支
        out1 = self.branch1(out)
        out2 = self.branch2(out)
        out3 = self.branch3(out)

        # 在通道维度concat
        out = torch.cat([out1, out2, out3], dim=1)

        # 通过1x1卷积调整通道数
        out = self.final_conv(out)

        # 残差连接
        out = out + x

        return out


class MLP(nn.Module):
    def __init__(self, in_features, mlp_ratio=4):
        super().__init__()
        hidden_features = int(in_features * mlp_ratio)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        # x: B, C, H, W -> B, H*W, C
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        # B, H*W, C -> B, C, H, W
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class GlobalAggregateModule(nn.Module):
    def __init__(self, in_channels, mlp_ratio=4):
        super().__init__()

        # DW-Conv + BN
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )

        # Shared MLP
        self.mlp = MLP(in_channels, mlp_ratio)

        # PW-Conv + BN
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        # 保存原始输入用于第一个残差连接
        identity1 = x

        # DW-Conv + BN
        out = self.dw_conv(x)

        # First MLP
        out = self.mlp(out)

        # First residual connection
        out = out + identity1

        # 保存PW-Conv的输入用于第二个残差连接
        identity2 = out

        # PW-Conv + BN
        out = self.pw_conv(out)

        # Second MLP (using the same MLP)
        out = self.mlp(out)

        # Second residual connection
        out = out + identity2

        return out


class CenterAggregateModule(nn.Module):
    def __init__(self, in_channels, branch_channels=64, mlp_ratio=4):
        super().__init__()

        # 两个并行分支，每个分支的输出通道数为in_channels
        self.global_branch = GlobalAggregateModule(in_channels, mlp_ratio)
        self.local_branch = LocalAggregateModule(in_channels, branch_channels)

        # 将两个分支的输出拼接后映射回原始通道数
        self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        global_out = self.global_branch(x)
        local_out = self.local_branch(x)
        out = torch.cat([global_out, local_out], dim=1)
        out = self.final_conv(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿着通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接平均值和最大值
        x = torch.cat([avg_out, max_out], dim=1)

        # 通过卷积和sigmoid得到空间注意力权重
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        # 先进行通道注意力
        x = x * self.channel_attention(x)
        # 再进行空间注意力
        x = x * self.spatial_attention(x)
        return x


class FeatureBackflowPN(nn.Module):
    def __init__(self, in_channels, branch_channels=64, mlp_ratio=4):
        super().__init__()

        # Center Aggregate Module for smallest feature map
        self.cam = CenterAggregateModule(in_channels, branch_channels, mlp_ratio)

        # PWConv and CBAM for each level
        self.pw_conv1 = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.pw_conv2 = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.pw_conv3 = nn.Conv2d(in_channels * 2, in_channels, 1)

        self.cbam1 = CBAM(in_channels)
        self.cbam2 = CBAM(in_channels)
        self.cbam3 = CBAM(in_channels)

        # Average Pooling layers
        self.avg_pool1 = nn.AvgPool2d(2, 2)
        self.avg_pool2 = nn.AvgPool2d(2, 2)
        self.avg_pool3 = nn.AvgPool2d(2, 2)

    def upsample(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, features):
        c2, c3, c4, c5 = features

        # Process smallest feature map with CAM
        p5 = self.cam(c5)
        p5 = self.cbam3(p5)

        # Bottom-up path with average pooling
        c4_pooled = self.avg_pool3(c4)
        c3_pooled = self.avg_pool2(c3)
        c2_pooled = self.avg_pool1(c2)

        # Process P4
        p4_up = self.upsample(p5, c4_pooled.shape[-2:])
        p4_up = self.cbam3(p4_up)
        p4_concat = torch.cat([c4_pooled, p4_up], dim=1)
        p4 = self.pw_conv3(p4_concat)

        # Process P3
        p3_up = self.upsample(p4, c3_pooled.shape[-2:])
        p3_up = self.cbam2(p3_up)
        p3_concat = torch.cat([c3_pooled, p3_up], dim=1)
        p3 = self.pw_conv2(p3_concat)

        # Process P2
        p2_up = self.upsample(p3, c2_pooled.shape[-2:])
        p2_up = self.cbam1(p2_up)
        p2_concat = torch.cat([c2_pooled, p2_up], dim=1)
        p2 = self.pw_conv1(p2_concat)

        return [p2, p3, p4, p5]