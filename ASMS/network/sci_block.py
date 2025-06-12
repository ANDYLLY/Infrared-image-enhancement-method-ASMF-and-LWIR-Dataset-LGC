import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.in_norm = nn.InstanceNorm2d(n, affine=True, track_running_stats=False)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.in_norm(x)
        
class MSD_Block(nn.Module):
    def __init__(self, layers=8, channels=16, outclass=1):
        super(MSD_Block, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding,
                      padding_mode='replicate'),
            AdaptiveNorm(channels),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding,
                      padding_mode='replicate'),
            AdaptiveNorm(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.outc = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=outclass, kernel_size=3, stride=1, padding=1,
                      padding_mode='replicate'),
            AdaptiveNorm(outclass),
            nn.Sigmoid()
        )

    def forward(self, x):
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.outc(fea)
        return fea