import math
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape
        q = self.q(q).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def apply_center_diff(x, window_size):
    """Apply center difference: all pixels minus center pixel.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C)
        window_size (int): Size of the window

    Returns:
        torch.Tensor: Difference windows of shape (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    windows = window_partition(x, window_size)
    num_windows = windows.shape[0]

    center_values = windows[:, window_size // 2, window_size // 2, :].unsqueeze(1).unsqueeze(1)
    center_values = center_values.expand(-1, window_size, window_size, -1)

    diff_windows = windows - center_values
    diff_windows = diff_windows.view(num_windows, -1, C)

    return diff_windows


def apply_diagonal_diff(x, window_size):
    """Apply diagonal difference : corner pixels minus center pixel.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C)
        window_size (int): Size of the window

    Returns:
        torch.Tensor: Difference windows of shape (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    windows = window_partition(x, window_size)
    num_windows = windows.shape[0]

    center_values = windows[:, window_size // 2, window_size // 2, :].unsqueeze(1).unsqueeze(1)

    diagonal_mask = torch.zeros(window_size, window_size, device=x.device)
    diagonal_mask[0, 0] = diagonal_mask[0, -1] = diagonal_mask[-1, 0] = diagonal_mask[-1, -1] = 1.0
    diagonal_mask[window_size // 2, window_size // 2] = 1.0

    diff_windows = windows.clone()
    diff_windows_selected = windows * diagonal_mask.unsqueeze(0).unsqueeze(-1)
    diff_windows_selected = diff_windows_selected - center_values * diagonal_mask.unsqueeze(0).unsqueeze(-1)
    diff_windows = torch.where(diagonal_mask.unsqueeze(0).unsqueeze(-1) > 0,
                               diff_windows_selected,
                               windows)

    diff_windows = diff_windows.view(num_windows, -1, C)

    return diff_windows


def apply_cross_diff(x, window_size):
    """Apply cross difference: cross pixels minus center pixel.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C)
        window_size (int): Size of the window

    Returns:
        torch.Tensor: Difference windows of shape (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    windows = window_partition(x, window_size)
    num_windows = windows.shape[0]

    center_values = windows[:, window_size // 2, window_size // 2, :].unsqueeze(1).unsqueeze(1)

    cross_mask = torch.zeros(window_size, window_size, device=x.device)
    cross_mask[window_size // 2, :] = 1.0  # horizontal
    cross_mask[:, window_size // 2] = 1.0  # vertical

    diff_windows = windows.clone()
    diff_windows_selected = windows * cross_mask.unsqueeze(0).unsqueeze(-1)
    diff_windows_selected = diff_windows_selected - center_values * cross_mask.unsqueeze(0).unsqueeze(-1)
    diff_windows = torch.where(cross_mask.unsqueeze(0).unsqueeze(-1) > 0,
                               diff_windows_selected,
                               windows)

    diff_windows = diff_windows.view(num_windows, -1, C)

    return diff_windows


class CascadedDifferenceWindowAttention(nn.Module):
    def __init__(self, dim, window_size=3, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        # Calculate split dimensions to handle non-divisible channels
        self.split_dims = self._calculate_split_dims(dim)

        # Difference attention modules
        self.center_attn = WindowAttention(
            self.split_dims[0], (window_size, window_size),
            num_heads, qkv_bias, qk_scale, attn_drop, proj_drop
        )
        self.diagonal_attn = WindowAttention(
            self.split_dims[1], (window_size, window_size),
            num_heads, qkv_bias, qk_scale, attn_drop, proj_drop
        )
        self.cross_attn = WindowAttention(
            self.split_dims[2], (window_size, window_size),
            num_heads, qkv_bias, qk_scale, attn_drop, proj_drop
        )

        # Regular window attention modules with 7x7 window
        self.regular_attn1 = WindowAttention(
            self.split_dims[0], (7, 7),
            num_heads, qkv_bias, qk_scale, attn_drop, proj_drop
        )
        self.regular_attn2 = WindowAttention(
            self.split_dims[1], (7, 7),
            num_heads, qkv_bias, qk_scale, attn_drop, proj_drop
        )
        self.regular_attn3 = WindowAttention(
            self.split_dims[2], (7, 7),
            num_heads, qkv_bias, qk_scale, attn_drop, proj_drop
        )

        self.merge = nn.Linear(dim, dim)

    def _calculate_split_dims(self, dim):
        """Calculate split dimensions for non-divisible channel numbers"""
        base_split = dim // 3
        remainder = dim % 3

        splits = [base_split] * 3
        # Distribute remaining channels
        for i in range(remainder):
            splits[i] += 1

        return splits

    def forward(self, x):
        B, H, W, C = x.shape

        # Split channels into three parts using calculated dimensions
        x_splits = torch.split(x, self.split_dims, dim=-1)
        x1, x2, x3 = x_splits

        # Branch 1: Center difference
        diff1 = apply_center_diff(x1, self.window_size)
        diff_attn1 = self.center_attn(diff1, diff1, diff1)
        diff_attn1 = diff_attn1.view(-1, self.window_size, self.window_size, self.split_dims[0])
        diff_out1 = window_reverse(diff_attn1, self.window_size, H, W)

        # Regular attention for x1 with 7x7 window
        x1_windows = window_partition(x1, 7)
        x1_windows = x1_windows.view(-1, 49, self.split_dims[0])
        reg_attn1 = self.regular_attn1(x1_windows, x1_windows, x1_windows)
        reg_attn1 = reg_attn1.view(-1, 7, 7, self.split_dims[0])
        reg_out1 = window_reverse(reg_attn1, 7, H, W)

        # Combine difference and regular attention results for branch 1
        out1 = diff_out1 + reg_out1

        # Branch 2: Diagonal difference with cascade from branch 1
        x2 = x2 + out1
        diff2 = apply_diagonal_diff(x2, self.window_size)
        diff_attn2 = self.diagonal_attn(diff2, diff2, diff2)
        diff_attn2 = diff_attn2.view(-1, self.window_size, self.window_size, self.split_dims[1])
        diff_out2 = window_reverse(diff_attn2, self.window_size, H, W)

        # Regular attention for x2 with 7x7 window
        x2_windows = window_partition(x2, 7)
        x2_windows = x2_windows.view(-1, 49, self.split_dims[1])
        reg_attn2 = self.regular_attn2(x2_windows, x2_windows, x2_windows)
        reg_attn2 = reg_attn2.view(-1, 7, 7, self.split_dims[1])
        reg_out2 = window_reverse(reg_attn2, 7, H, W)

        # Combine difference and regular attention results for branch 2
        out2 = diff_out2 + reg_out2

        # Branch 3: Cross difference with cascade from branch 2
        x3 = x3 + out2
        diff3 = apply_cross_diff(x3, self.window_size)
        diff_attn3 = self.cross_attn(diff3, diff3, diff3)
        diff_attn3 = diff_attn3.view(-1, self.window_size, self.window_size, self.split_dims[2])
        diff_out3 = window_reverse(diff_attn3, self.window_size, H, W)

        # Regular attention for x3 with 7x7 window
        x3_windows = window_partition(x3, 7)
        x3_windows = x3_windows.view(-1, 49, self.split_dims[2])
        reg_attn3 = self.regular_attn3(x3_windows, x3_windows, x3_windows)
        reg_attn3 = reg_attn3.view(-1, 7, 7, self.split_dims[2])
        reg_out3 = window_reverse(reg_attn3, 7, H, W)

        # Combine difference and regular attention results for branch 3
        out3 = diff_out3 + reg_out3

        # Concatenate all outputs and apply final projection
        out = torch.cat([out1, out2, out3], dim=-1)
        out = self.merge(out)

        return out