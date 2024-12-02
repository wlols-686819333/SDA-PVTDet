import math
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECA_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out

class SpatialCrossOverlapEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=64):
        super().__init__()
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.H, self.W = img_size[0] // stride, img_size[1] // stride
        # self.num_patches = self.H * self.W
        self.HDConv = nn.Conv2d(in_chans, embed_dim//2, kernel_size=(2*stride-1,stride), stride=stride, padding=(stride-1,0))
        self.WDConv = nn.Conv2d(in_chans, embed_dim//2, kernel_size=(stride,2*stride-1), stride=stride, padding=(0,stride-1))
        self.depthwise = nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=3, stride=1, padding=1, groups=embed_dim//2 )
        self.pointwise = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=1, stride=1)
        self.ECA = ECA_block(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim//2)


    def forward(self, x):

        _, _, H, W = x.shape
        x_h = self.HDConv(x)
        x_w = self.WDConv(x)
        x_h = self.norm2(x_h)
        x_w = self.norm2(x_w)
        x1 = torch.cat([x_h,x_w],dim=1)
        x2 = x_h + x_w
        x1 = self.ECA(x1)
        x2 = self.depthwise(x2)
        x2 = self.pointwise(x2)
        X = x1 + x2
        out = X.flatten(2).transpose(1, 2)
        out = self.norm(out)

        return out

so = SpatialCrossOverlapEmbed()

# x = torch.randn(1,3,224,224)
# print(x.shape)
# x = so(x)
# print(x.shape)