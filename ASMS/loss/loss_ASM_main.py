import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(current_dir)
sys.path.append(current_dir)

import Loss_ASM

class ASM_S_Loss(nn.Module):
    def __init__(self, W_T=100, W_Dark=20, W_G=1, enable_T=1):
        super(ASM_S_Loss, self).__init__()
        self.W_T = W_T
        self.W_Dark = W_Dark
        self.W_G = W_G
        self.enable_T = enable_T

        self.T_loss = Loss_ASM.T_Smooth()
        self.G_Loss = Loss_ASM.T_Global()
        self.TZ_loss = Loss_ASM.TZ_Smooth()

    def forward(self, x, y, t):
        B, C, H, W = t.size()
        Loss_T = torch.zeros(1).cuda()
        Loss_Darkness = torch.zeros(1).cuda()
        Loss_G = torch.zeros(1).cuda()
        
        for i in range(C):
            dark = (float(C) + 1) / (float(i) + 1)
            self.Dark_loss = Loss_ASM.T_Dark(darkness=dark)

            t_i = t[:, i, :, :].unsqueeze(1)
            y_i = y[:, i, :, :].unsqueeze(1)

            if self.enable_T:
                Loss_T += self.W_T * self.T_loss(t_i) + self.W_T * 5e-5 * self.TZ_loss(t_i)
            else:
                Loss_T += 0
            Loss_Darkness += self.W_Dark * self.Dark_loss(x, y_i)
            Loss_G += self.W_G * self.G_Loss(x, y_i) 
        Loss_Total = Loss_T + Loss_Darkness + Loss_G
        return Loss_Total, Loss_Darkness, Loss_T, Loss_G