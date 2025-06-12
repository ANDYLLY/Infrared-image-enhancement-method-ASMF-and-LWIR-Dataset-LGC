import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Lowbound_Block(nn.Module):
    def __init__(self):
        super(Lowbound_Block, self).__init__()
    def forward(self, x, n_x):
        N, _, H, W = x.size()
        _, C_n, _, _ = n_x.size()
        x_expanded = x.expand(N, C_n, H, W)
        u_x = 1 - x_expanded
        t_x = u_x + x_expanded * n_x
        t_x = torch.clamp(t_x, max=1)
        return t_x