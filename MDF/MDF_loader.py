import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def populate_train_list(IR_images_path):
    # 正确使用glob模式匹配多种图像格式
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_list_IR = []
    for pattern in patterns:
        # 完全路径的构建应考虑到操作系统的差异
        image_list_IR.extend(glob.glob(os.path.join(IR_images_path, pattern)))
    return image_list_IR


def populate_folder_list(root_path):
    # 获取根路径下所有文件夹的列表
    folder_list = [os.path.join(root_path, folder) for folder in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, folder))]
    return folder_list
    
def ASM_populate_train_list(IR_images_path):
    # 正确使用glob模式匹配多种图像格式
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_list_IR = []
    for pattern in patterns:
        # 完全路径的构建应考虑到操作系统的差异
        image_list_IR.extend(glob.glob(os.path.join(IR_images_path, pattern)))
    return image_list_IR


class MDF_train_loader(data.Dataset):
    def __init__(self, root_path, num_folders=7):
        self.train_list = populate_folder_list(root_path)
        self.size = 512
        self.dnsize = 256
        self.num_folders = num_folders

        print("Dataset num", len(self.train_list))

        # Ensure the number of folders matches num_folders
        assert len \
            (self.train_list) == self.num_folders, f"Expected {self.num_folders} folders, but found {len(self.train_list)}."

        self.image_names = populate_train_list(self.train_list[0])
        self.num_images = len(self.image_names)
        print("Datsset size:", self.num_images)

        for folder in self.train_list:
            each_length = len(populate_train_list(folder))
            assert each_length >= self.num_images, f"Folder {folder} contains fewer images than {self.num_images}"

    def __getitem__(self, index):
        images = []
        dn_images = []

        # Get the same image by name from each folder
        image_name = self.image_names[index]
        image_name = os.path.basename(image_name)

        for folder_path in self.train_list:
            image_path = os.path.join(folder_path, image_name)
            image = Image.open(image_path)
            ori = image

            image_resized = ori.resize((640, 512), Image.Resampling.LANCZOS)
            dn_image_resized = ori.resize((320, 256), Image.Resampling.LANCZOS)

            # image_resized = image
            # dn_image_resized = ori.resize((image.width // 2, image.height // 2), Image.Resampling.LANCZOS)

            # 转换为 NumPy 数组
            image_resized = np.asarray(image_resized)
            dn_image_resized = np.asarray(dn_image_resized)

            # 确保图像是单通道灰度图，如果是彩色图像则转换为灰度
            if image_resized.ndim == 3:
                image_resized = image_resized.mean(axis=2)
            if dn_image_resized.ndim == 3:
                dn_image_resized = dn_image_resized.mean(axis=2)

            # 将数据标准化到[0, 1]
            image_resized = (image_resized / 255.0).astype(np.float32)
            dn_image_resized = (dn_image_resized / 255.0).astype(np.float32)

            image_resized = torch.from_numpy(image_resized)
            dn_image_resized = torch.from_numpy(dn_image_resized)

            images.append(image_resized)
            dn_images.append(dn_image_resized)
        # 将所有图像组合成一个张量
        images = torch.stack(images)  # [num_folders, 1, H, W]
        dn_images = torch.stack(dn_images)  # [num_folders, 1, H, W]
        return images, dn_images

    def __len__(self):
        return self.num_images


class MDF_test_loader(data.Dataset):
    def __init__(self, root_path, num_folders=7):
        self.train_list = populate_folder_list(root_path)
        self.num_folders = num_folders

        print("Dataset num:", len(self.train_list))

        # Ensure the number of folders matches num_folders
        assert len \
            (self.train_list) == self.num_folders, f"Expected {self.num_folders} folders, but found {len(self.train_list)}."

        self.image_names = populate_train_list(self.train_list[0])
        self.num_images = len(self.image_names)
        print("Dataset size:", self.num_images)

        for folder in self.train_list:
            each_length = len(populate_train_list(folder))
            assert each_length >= self.num_images, f"Folder {folder} contains fewer images than {self.num_images}"

    def __getitem__(self, index):
        images = []
        dn_images = []

        # Get the same image by name from each folder
        image_name = self.image_names[index]
        image_name = os.path.basename(image_name)

        for folder_path in self.train_list:
            image_path = os.path.join(folder_path, image_name)
            image = Image.open(image_path)
            ori = image

            # Convert PIL images to NumPy arrays
            image_resized = ori.resize((640, 512), Image.Resampling.LANCZOS)
            dn_image_resized = ori.resize((320, 256), Image.Resampling.LANCZOS)

            image_resized = np.asarray(image_resized)
            dn_image_resized = np.asarray(dn_image_resized)

            # Ensure the images are single-channel grayscale; convert if they are color images
            if image_resized.ndim == 3:
                image_resized = image_resized.mean(axis=2)
            if dn_image_resized.ndim == 3:
                dn_image_resized = dn_image_resized.mean(axis=2)

            # Standardize data to [0, 1]
            image_resized = (image_resized / 255.0).astype(np.float32)
            dn_image_resized = (dn_image_resized / 255.0).astype(np.float32)

            # Convert to torch tensors and add channel dimension
            image_resized = torch.from_numpy(image_resized)
            dn_image_resized = torch.from_numpy(dn_image_resized)

            images.append(image_resized)
            dn_images.append(dn_image_resized)
        # Stack all images into one tensor
        images = torch.stack(images)
        dn_images = torch.stack(dn_images)
        return images, dn_images, image_name

    def __len__(self):
        return self.num_images

class ASM_train_loader(data.Dataset):
    def __init__(self, IR_images_path, minmax=False):
        self.train_list = ASM_populate_train_list(IR_images_path)
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
    def __init__(self, IR_images_path, minmax=False):
        self.train_list = ASM_populate_train_list(IR_images_path)
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
