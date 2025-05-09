import math
import functools
from typing import Callable

import torch
from torch import nn
from typing import Tuple
from einops import rearrange
import torch.nn.functional as F

def Normalize(in_channels):
    if in_channels<=32:
        num_groups=in_channels
    else:
        num_groups=32
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0., temporal_kernel=3):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(temporal_kernel, 3, 3), 
                               stride=1, padding=(temporal_kernel//2, 1, 1))

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(temporal_kernel, 3, 3), 
                               stride=1, padding=(temporal_kernel//2, 1, 1))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=(temporal_kernel, 3, 3), 
                                               stride=1, padding=(temporal_kernel//2, 1, 1))
            else:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class SpatialDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.norm = Normalize(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True, size=None):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.size=size

    def forward(self, x):
        if len(x.shape) != 5:
            raise RuntimeError("The input tensor has to be 5D")
        # x.shape: b c t h w
        b,c,t,h,w = x.shape
        x = x.permute(0,2,1,3,4)
        x = rearrange(x,'b t c h w -> (b t) c h w')
        if self.size is not None:
            x = torch.nn.functional.interpolate(x, size=(self.size), mode="nearest")
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        x = rearrange(x,'(b t) c h w -> b t c h w',b=b,t=t)
        x = x.permute(0,2,1,3,4) # b c t h w
        return x

class SpatioTemporalSelfAttention(nn.Module):
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, T, H, W = x.shape
        N = T * H * W
        # Reshape and permute the input tensor
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, C)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Compute q, k, v
        qkv = self.qkv(x).reshape(B, T*H*W, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention using torch.baddbmm for better performance
        attn = torch.empty(B, N, N, dtype=q.dtype, device=q.device)
        attn = torch.baddbmm(attn, q, k.transpose(-2, -1), 
                             beta=0, alpha=self.dim**-0.5)
        attn = attn.softmax(dim=-1)
        
        # Apply attention
        x = torch.bmm(attn, v)
        
        # Apply projection and dropout
        x = self.proj(x)
        x = self.dropout(x)
        
        # Reshape the output tensor to the original shape
        x = x.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        
        return x

class VideoEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 features: Tuple[int, ...] = (64, 128, 256, 512, 512),
                 temporal_kernel: int = 5,
                 num_blocks: Tuple[int, ...] = (4, 3, 2, 1, 1)):
        super().__init__()
        
        assert len(features) == len(num_blocks), "Length of features and num_blocks must match"
        
        self.stem = nn.Conv3d(in_channels, features[0], kernel_size=3, stride=1, padding=1)
        
        self.blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            for _ in range(num_blocks[i]):
                self.blocks.append(ResnetBlock3d(features[i], features[i], temporal_kernel=temporal_kernel))
            self.blocks.append(SpatialDownsample(features[i], features[i+1]))
        
        # Add final ResBlocks and attention
        self.final_blocks = nn.ModuleList([
            ResnetBlock3d(features[-1], features[-1], temporal_kernel=temporal_kernel),
            SpatioTemporalSelfAttention(features[-1]),
            ResnetBlock3d(features[-1], features[-1], temporal_kernel=temporal_kernel)
        ])
        # Final Conv3d layer
        self.final_conv = nn.Conv3d(features[-1], features[-1], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        
        for block in self.final_blocks:
            x = block(x)
        x = self.final_conv(x)
        return x

class VideoDecoder(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 features: Tuple[int, ...] = (512, 256, 128, 64, 64),
                 temporal_kernel: int = 5,
                 num_blocks: Tuple[int, ...] = (1, 2, 3, 4, 1)):
        super().__init__()
        
        assert len(features) == len(num_blocks), "Length of features and num_blocks must match"
        
        # Initial Conv3d and ResBlocks
        self.initial_conv = nn.Conv3d(features[0], features[0], kernel_size=3, stride=1, padding=1)
        self.initial_blocks = nn.ModuleList([
            ResnetBlock3d(features[0], features[0], temporal_kernel=temporal_kernel),
            SpatioTemporalSelfAttention(features[0]),
            ResnetBlock3d(features[0], features[0], temporal_kernel=temporal_kernel)
        ])
        
        self.blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            self.blocks.append(Upsample(features[i], with_conv=True))
            for idx in range(num_blocks[i]):
                if idx==0:
                    self.blocks.append(ResnetBlock3d(features[i], features[i+1], temporal_kernel=temporal_kernel,))
                else:
                    self.blocks.append(ResnetBlock3d(features[i+1], features[i+1], temporal_kernel=temporal_kernel))

        self.final_conv = nn.Conv3d(features[-1], in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        
        for block in self.initial_blocks:
            x = block(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x.clamp(0, 1)
