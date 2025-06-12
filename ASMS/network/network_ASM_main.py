import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(current_dir)
sys.path.append(current_dir)
from sci_block import MSD_Block
from lowbound_block import Lowbound_Block
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ASM_S_Net(nn.Module):
    def __init__(self,
                 num_layer=8,
                 num_channel=16,
                 backbone="default",
                 lowbound_enable=1,
                 inference=6
                 ):
        super(ASM_S_Net, self).__init__()
        if backbone == "default":
            self.backbone = MSD_Block(num_layer, num_channel, inference)
        else:
            self.backbone = MSD_Block(num_layer, num_channel, inference)
        if lowbound_enable == 1:
            self.t_clip = True
            self.t_clip_block = Lowbound_Block()
        else:
            self.t_clip = False
            self.t_clip_block = None

    def forward(self, x):
        features = self.backbone(x)
        n_x = features
        if self.t_clip:
            t_x = self.t_clip_block(x, n_x)
        else:
            t_x = n_x
        L_x = 1 - t_x
        J_x = (x - L_x) / (t_x + 1e-6)
        return J_x, t_x
