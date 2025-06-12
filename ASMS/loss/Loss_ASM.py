import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional
import numpy as np

class TZ_Smooth(nn.Module):
    def __init__(self):
        super(TZ_Smooth, self).__init__()

        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)

        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)

    def forward(self, x):
        return  self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))

class T_Smooth(nn.Module):
    def __init__(self):
        super(T_Smooth, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size

class T_Dark(nn.Module):
    def __init__(self, darkness):
        super(T_Dark, self).__init__()
        self.darkness = darkness

    def forward(self, o, e):
        darklord = torch.mean(o, dim=(1, 2, 3)) / self.darkness
        dis = torch.mean(e - o, dim=(1, 2, 3))
        dis = torch.abs(dis + darklord)
        return torch.mean(dis)

class T_Global(nn.Module):
    def __init__(self):
        super(T_Global, self).__init__()

    def forward(self, o, e):
        dis = torch.var(e - o, dim=(1, 2, 3))
        return torch.mean(dis)



