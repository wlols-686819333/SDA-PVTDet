import math
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GA_MSA(nn.Module):
    def __init__(self, dim, num_heads, activation=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Shared dilated convolution for query generation
        self.shared_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim)
        )

        # Key and Value linear projections
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # Channel attention layers
        self.channel_mixing = nn.Sequential(
            nn.Conv2d(3 * dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # Final projection
        self.proj = nn.Sequential(
            activation(),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

        self.silu = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape

        # Generate queries using dilated convolutions with shared weights
        # Rate = 1
        q1 = self.shared_conv(x)

        # Rate = 2
        q2 = F.conv2d(
            x,
            self.shared_conv[0].weight,
            self.shared_conv[0].bias,
            stride=1, padding=2, dilation=2
        )
        q2 = self.shared_conv[1](q2)

        # Rate = 3
        q3 = F.conv2d(
            x,
            self.shared_conv[0].weight,
            self.shared_conv[0].bias,
            stride=1, padding=3, dilation=3
        )
        q3 = self.shared_conv[1](q3)

        # Rate = 4
        q4 = F.conv2d(
            x,
            self.shared_conv[0].weight,
            self.shared_conv[0].bias,
            stride=1, padding=4, dilation=4
        )
        q4 = self.shared_conv[1](q4)

        # Combine multi-scale queries
        q = q1 + q2 + q3 + q4
        q = self.silu(q)  # B,C,H,W

        # Generate keys and values using linear projections
        k = x.flatten(2).transpose(1, 2)  # B,C,H,W -> B,HW,C
        k = self.to_k(k)  # B,HW,C
        k = k.transpose(1, 2).reshape(B, C, H, W)  # B,C,H,W

        v = x.flatten(2).transpose(1, 2)
        v = self.to_v(v)  # B,HW,C
        v = v.transpose(1, 2).reshape(B, C, H, W)  # B,C,H,W

        # Height-wise squeeze attention
        q_h = q.mean(-1, keepdim=True)  # B,C,H,1
        k_h = k.mean(-1, keepdim=True)  # B,C,H,1
        v_h = v.mean(-1, keepdim=True)  # B,C,H,1

        q_h = q_h.reshape(B, self.num_heads, self.head_dim, H, 1)
        k_h = k_h.reshape(B, self.num_heads, self.head_dim, H, 1)
        v_h = v_h.reshape(B, self.num_heads, self.head_dim, H, 1)

        attn_h = (q_h.squeeze(-1).transpose(-2, -1) @ k_h.squeeze(-1)) * self.scale
        attn_h = attn_h.softmax(dim=-1)
        out_h = (v_h.squeeze(-1) @ attn_h.transpose(-2, -1)).unsqueeze(-1)
        out_h = out_h.reshape(B, C, H, 1)

        # Width-wise squeeze attention
        q_w = q.mean(-2, keepdim=True)  # B,C,1,W
        k_w = k.mean(-2, keepdim=True)  # B,C,1,W
        v_w = v.mean(-2, keepdim=True)  # B,C,1,W

        q_w = q_w.reshape(B, self.num_heads, self.head_dim, 1, W)
        k_w = k_w.reshape(B, self.num_heads, self.head_dim, 1, W)
        v_w = v_w.reshape(B, self.num_heads, self.head_dim, 1, W)

        attn_w = (q_w.squeeze(-2).transpose(-2, -1) @ k_w.squeeze(-2)) * self.scale
        attn_w = attn_w.softmax(dim=-1)
        out_w = (v_w.squeeze(-2) @ attn_w.transpose(-2, -1)).unsqueeze(-2)
        out_w = out_w.reshape(B, C, 1, W)

        # Combine squeezed attention results
        out_squeeze = out_h + out_w

        # Channel Attention
        qkv_concat = torch.cat([
            q.reshape(B, C, H, W),
            k.reshape(B, C, H, W),
            v.reshape(B, C, H, W)
        ], dim=1)  # B, (3*C), H, W

        # Global average pooling
        channel_weights = F.adaptive_avg_pool2d(qkv_concat, 1)  # B, (3*C), 1, 1
        # Mix channels and get attention weights
        channel_weights = self.channel_mixing(channel_weights)  # B, C, 1, 1
        channel_weights = F.softmax(channel_weights, dim=1)  # B, C, 1, 1

        # Final output
        out = self.proj(out_squeeze)  # B, C, H, W
        out = out * channel_weights

        return out