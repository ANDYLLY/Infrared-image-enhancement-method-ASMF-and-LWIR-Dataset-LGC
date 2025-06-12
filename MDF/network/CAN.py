import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(current_dir)
sys.path.append(current_dir)

from guided_filter import FastGuidedFilter

def weights_init_identity(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight.data)
    elif classname.find('InstanceNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data,   0.0)

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()
        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))
        self.in_norm = nn.InstanceNorm2d(n, affine=True, track_running_stats=False)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.in_norm(x)


class W_Net(nn.Module):
    def __init__(self, layer=5, features=24):
        super(W_Net, self).__init__()
        self.norm = AdaptiveNorm(n=features)
        layers = [
            nn.Conv2d(1, features, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            self.norm,
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for l in range(1, layer):
            layers += [nn.Conv2d(features, features, kernel_size=3, stride=1, padding=2 ** l, dilation=2 ** l, bias=False),
                       self.norm,
                       nn.LeakyReLU(0.2, inplace=True)]
        layers += [
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            self.norm,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, 1, kernel_size=1, stride=1, padding=0, dilation=1)
        ]
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init_identity)

    def forward(self, x):
        y = self.net(x)
        return y


class FW_Net(nn.Module):
    def __init__(self, num=3, layer=5, features=24):
        super(FW_Net, self).__init__()
        self.block = W_Net(layer, features)
        self.img_num = num
        print("num:", num)

    def forward(self, x):
        processed_imgs = []
        for i in range(self.img_num):
            img = x[:, i:i + 1, :, :]
            processed_img = self.block(img)
            processed_imgs.append(processed_img)
        y = torch.cat(processed_imgs, dim=1)
        return y

class Sigmoid_Trans(nn.Module):
    def __init__(self):
        super(Sigmoid_Trans, self).__init__()

    def forward(self, x):
        a = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        c = torch.clamp(0.25*a, min=0.001)
        b = torch.clamp(0.5+0.5*a, max=1)
        up_c = 0.5
        down_c = 0.5
        d = c-(c*up_c)
        e = b+(1-b)*down_c
        k1 = d / c
        k2 = (a - d) / (a - c)
        k3 = (e - a) / (b - a)
        k4 = (1 - e) / (1 - b)
        b1 = 0
        b2 = (a*d-a*c)/(a-c)
        b3 = (b*a-e*a)/(b-a)
        b4 = (e-b)/(1-b)
        y1 = k1 * x + b1
        y2 = k2 * x + b2
        y3 = k3 * x + b3
        y4 = k4 * x + b4
        j1 = x <= c
        j2 = (x <= a) & (x > c)
        j3 = (x <= b) & (x > a)
        j4 = (x > b)
        y = y1 * j1 + y2 * j2 + y3 * j3 + y4 * j4
        return y

class MDF_Net(nn.Module):
    def __init__(self, img_num=3, layer=5, features=24, eps=1e-4, radius=1, sig=True):
        super(MDF_Net, self).__init__()
        if sig != 0:
            self.block = FW_Net(num=2*img_num, layer=layer, features=features)
        else:
            self.block = FW_Net(num=img_num, layer=layer, features=features)
        self.gf = FastGuidedFilter(radius, eps)
        self.sigmoid = Sigmoid_Trans()
        self.sig = sig

    def forward(self, x_lr, x_hr):
        EPS = 1e-8
        if self.sig != 0:
            x_lr_sigmoid = self.sigmoid(x_lr)
            x_hr_sigmoid = self.sigmoid(x_hr)
            x_lr_mixed = torch.cat((x_lr_sigmoid, x_lr), dim=1)
            x_hr_mixed = torch.cat((x_hr_sigmoid, x_hr), dim=1)
        else:
            x_lr_mixed = x_lr
            x_hr_mixed = x_hr
        w_lr = self.block(x_lr_mixed)
        w_hr = self.gf(x_lr_mixed, w_lr, x_hr_mixed)
        w_hr = torch.abs(w_hr)
        w_hr = (w_hr + EPS) / torch.sum((w_hr + EPS), dim=1, keepdim=True)
        o_hr = torch.sum(w_hr * x_hr_mixed, dim=1, keepdim=True).clamp(0, 1)
        return o_hr, w_hr
