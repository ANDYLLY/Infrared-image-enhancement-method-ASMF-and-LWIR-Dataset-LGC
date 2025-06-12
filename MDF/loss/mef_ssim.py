import torch
import torch.nn.functional as F
import torch.nn as nn
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / (gauss.sum())

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, window_size/6.).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=False, gray=0.5):

    K, C, H, W = list(Ys.size())
    muY_seq = F.conv2d(Ys, window, padding=ws // 2, groups=C).view(K, C, H, W)
    muY_sq_seq = muY_seq * muY_seq
    # DX = EX2 - (EX)2
    sigmaY_sq_seq = F.conv2d(Ys * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) - muY_sq_seq
    relative_contrast = sigmaY_sq_seq / torch.clamp(muY_sq_seq, min=0.2)

    # C(x) = max(c)
    sigmaY_sq, patch_index = torch.max(relative_contrast, dim=1)

    # C(y)
    window_x = create_window(ws, 1).cuda()
    muX = F.conv2d(X, window_x, padding=ws // 2).view(K, 1, H, W)
    muX_sq = muX * muX
    sigmaX_sq = F.conv2d(X * X, window_x, padding=ws // 2).view(K, 1, H, W) - muX_sq

    # E(XY)
    sigmaXY = F.conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) - muX.expand_as(muY_seq) * muY_seq

    # the first term of mef-ssim
    cs_seq = (2 * sigmaXY + C2) / (sigmaX_sq + sigmaY_sq_seq + C2)
    # select according to c, a question: where is s?
    cs_map = torch.gather(cs_seq.view(K, C, -1), 1, patch_index.view(K, 1, -1)).view(K, H, W)

    # the second term of mef-ssim
    is_lum = False
    if is_lum:
        lY = torch.mean(muY_seq.view(K, C, -1), dim=2)
        lL = torch.exp(-((muY_seq - gray) ** 2) / denom_l)

        lG = torch.exp(- ((lY - gray) ** 2) / denom_g)[:, :, None, None].expand_as(lL)
        LY = lG * lL

        muY = torch.sum((LY * muY_seq), dim=1) / torch.sum(LY, dim=1)
        muY_sq = muY * muY
        l_map = (2 * muX * muY + C1) / (muX_sq + muY_sq + C1)
    else:
        l_map = torch.Tensor([1.0])
        if Ys.is_cuda:
            l_map = l_map.cuda(Ys.get_device())
    ssim = l_map * cs_map
    return torch.mean(ssim)

class MEF_MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False, gray=0.5):
        super(MEF_MSSSIM, self).__init__()

        self.window_size = window_size

        self.denom_g = 2 * sigma_g**2
        self.denom_l = 2 * sigma_l**2
        self.C1 = c1**2
        self.C2 = c2**2
        self.is_lum = is_lum
        self.gray = gray

    def forward(self, X, Ys):
        (_, channel, _, _) = Ys.size()
        window = create_window(self.window_size, channel)

        if Ys.is_cuda:
            window = window.cuda(Ys.get_device())
        window = window.type_as(Ys)

        self.window = window

        return _mef_ssim(X, Ys, window, self.window_size,
                          self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum, self.gray)

class Loss_MEFSSIM(nn.Module):
    def __init__(self, lum, gray, sig_g, sig_l, win):
        super(Loss_MEFSSIM, self).__init__()
        self.operator = MEF_MSSSIM(is_lum=False, gray=gray, window_size=win, sigma_g=sig_g, sigma_l=sig_l)

    def forward(self, x, y):
        loss = -self.operator(x, y) + 1
        return loss
    
class Loss_Fuzzy(nn.Module):
    def __init__(self):
        super(Loss_Fuzzy, self).__init__()

    def forward(self, x):
        if x.size(1) > 1:
            x = x[:, 0, :, :]  # 获取第一个通道
        else:
            x = x.squeeze(1)  # 若只有一个通道则直接去掉该维度

        # 获取图像的大小
        batch_size, height, width = x.shape
        num_pixels = height * width

        # 转换为浮点类型，计算隶属度函数 q(x, y)
        x = x.float()
        max_x = torch.max(x, dim=1, keepdim=True)[0]
        max_x = torch.max(max_x, dim=2, keepdim=True)[0]

        q = torch.sin((3.141592653589793 / 2) * (1 - x / max_x))  # 计算隶属度函数 q(x, y)

        # 计算 min(q, 1 - q)，并求和得到模糊度指标
        min_q = torch.minimum(q, 1 - q)
        eta = (2.0 / num_pixels) * torch.sum(min_q, dim=[1, 2])  # 对 height 和 width 维度求和
        loss = torch.mean(eta)  # 平均损失值，适用于批量数据

        return loss

    
class Loss_MEF(nn.Module):
    def __init__(self, lum, gray, sig_g, sig_l, win):
        super(Loss_MEF, self).__init__()
        self.ssim_loss = Loss_MEFSSIM(lum, gray, sig_g, sig_l, win)

    def forward(self, x, y):
        loss_ssim = self.ssim_loss(x, y)
        loss = loss_ssim
        return loss
