import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import os

def _populate_train_list(IR_images_path):
    # 正确使用glob模式匹配多种图像格式
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_list_IR = []
    for pattern in patterns:
        # 完全路径的构建应考虑到操作系统的差异
        image_list_IR.extend(glob.glob(os.path.join(IR_images_path, pattern)))
    return image_list_IR


class ASM_train_loader(data.Dataset):
    def __init__(self, IR_images_path, minmax=True):
        self.train_list = _populate_train_list(IR_images_path)
        self.size = 512
        self.data_list = self.train_list
        self.ctl = minmax
        print("dataset is prepared. dataset size:", len(self.train_list))

    def __getitem__(self, index):
        data_IR_path = self.data_list[index]
        data_IR = Image.open(data_IR_path)
        if data_IR.mode == 'RGBA' or data_IR.mode == 'LA':
            data_IR = data_IR.convert('L')
        data_IR = data_IR.resize((self.size, self.size), Image.Resampling.LANCZOS)
        data_IR = np.asarray(data_IR)
        # 确保图像是单通道灰度图，如果是彩色图像则转换为灰度
        if data_IR.ndim == 3:
            data_IR = data_IR.mean(axis=2)
        data_IR = (data_IR / 255.0).astype(np.float32)
        data_IR = torch.from_numpy(data_IR).unsqueeze(0)  # [H, W] -> [1, H, W]

        if self.ctl:
            min_val = data_IR.min()
            max_val = data_IR.max()
            data_IR = (data_IR - min_val) / (max_val - min_val)

        return data_IR

    def __len__(self):
        return len(self.data_list)

class ASM_test_loader(data.Dataset):
    def __init__(self, IR_images_path, minmax=True):
        self.train_list = _populate_train_list(IR_images_path)
        self.data_list = self.train_list
        self.ctl = minmax
        print("dataset is prepared. dataset size:", len(self.train_list))

    def __getitem__(self, index):
        data_IR_path = self.data_list[index]
        data_IR = Image.open(data_IR_path)
        # 转换为 NumPy 数组
        data_IR = np.asarray(data_IR)
        # 确保图像是单通道灰度图，如果是彩色图像则转换为灰度
        if data_IR.ndim == 3:
            data_IR = data_IR.mean(axis=2)
        data_IR = (data_IR / 255.0).astype(np.float32)
        data_IR = torch.from_numpy(data_IR).unsqueeze(0)

        # 进行min-max拉伸
        if self.ctl:
            min_val = data_IR.min()
            max_val = data_IR.max()
            data_IR = (data_IR - min_val) / (max_val - min_val)

        return data_IR, os.path.basename(data_IR_path)

    def __len__(self):
        return len(self.data_list)
