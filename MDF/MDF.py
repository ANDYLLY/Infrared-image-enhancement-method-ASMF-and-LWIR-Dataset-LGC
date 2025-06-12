import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
import random
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(current_dir)
sys.path.append(current_dir)

from .network import CAN
from .loss import mef_ssim

import MDF_loader
import os
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import time
import torchvision

class MDF_Net(nn.Module):
    def __init__(self,
                 experiment="LGC_std",
                 nn_img_num=6,
                 nn_layer=5,
                 nn_features=24,
                 nn_gf_r=3,
                 sigmoid_mode=True,
                 fuzzy_mode=True,
                 loss_sig_g=0.2,
                 loss_sig_l=0.5,
                 loss_gray=0.3,
                 loss_lum=1,
                 loss_win=17,
                 train_bn=4,
                 train_workers=16,
                 pretrained="",
                 lr=1e-3,
                 weight_decay=3e-5,
                 grad_clip=0.5,
                 train_epoch=150
                 ):
        super(MDF_Net, self).__init__()

        self.sigmoid_mode = sigmoid_mode
        self.fuzzy_mode = fuzzy_mode
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.train_epochs = train_epoch
        self.nn = CAN.MDF_Net(img_num=nn_img_num, layer=nn_layer, features=nn_features, radius=nn_gf_r, sig=self.sigmoid_mode)

        self.loss = mef_ssim.Loss_MEF(lum=loss_lum, gray=loss_gray, sig_g=loss_sig_g, sig_l=loss_sig_l, win=loss_win)
        self.fuzzy_loss = mef_ssim.Loss_Fuzzy()
        self.fusion_num = nn_img_num

        self.current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(self.current_file_dir)

        self.data_root = os.path.join(self.root_dir, 'experiment', experiment, 'ASMS', 'result')
        self.train_dir = os.path.join(self.data_root, 'valid')
        self.test_dir = os.path.join(self.data_root, 'test')
        self.check_dir = os.path.join(self.data_root, 'check')
        self.train_data = MDF_loader.MDF_train_loader(root_path=self.train_dir, num_folders=self.fusion_num)
        self.valid_data = MDF_loader.MDF_test_loader(root_path=self.train_dir, num_folders=self.fusion_num)
        self.check_data = MDF_loader.MDF_test_loader(root_path=self.check_dir, num_folders=self.fusion_num)
        self.test_data = MDF_loader.MDF_test_loader(root_path=self.test_dir, num_folders=self.fusion_num)

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=train_bn, shuffle=False,
                                                        num_workers=train_workers, pin_memory=False)
        self.check_loader = torch.utils.data.DataLoader(self.check_data, batch_size=1, shuffle=False,
                                                        num_workers=1, pin_memory=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=1, shuffle=False,
                                                       num_workers=1, pin_memory=False)

        self.valid_loader = torch.utils.data.DataLoader(self.valid_data, batch_size=1, shuffle=False,
                                                        num_workers=1, pin_memory=False)
        self.check_test_loader = torch.utils.data.DataLoader(self.check_data, batch_size=1, shuffle=False,
                                                        num_workers=1, pin_memory=False)

        self.record_root_dir = os.path.join(self.root_dir, 'experiment', experiment, 'MDF')
        self.record_snapshots = os.path.join(self.record_root_dir, 'snapshots')
        self.record_effect = os.path.join(self.record_root_dir, 'effect')
        self.record_run = os.path.join(self.record_root_dir, 'run')
        os.makedirs(self.record_root_dir, exist_ok=True)
        os.makedirs(self.record_snapshots, exist_ok=True)
        os.makedirs(self.record_effect, exist_ok=True)
        os.makedirs(self.record_run, exist_ok=True)

        if pretrained:
            self.pretrained = True
            self.pretrained_root = os.path.join(self.root_dir, 'experiment', pretrained, 'MDF')
            self.pretrained_snapshots_dir = os.path.join(self.pretrained_root, 'snapshots')
            self.pretrained_snapshots_file = MDF_Net.get_latest_weight_file(self.pretrained_snapshots_dir)
        else:
            self.pretrained = False
            self.pretrained_root = None
            self.pretrained_snapshots_dir = None
            self.pretrained_snapshots_file = None
        if self.pretrained_snapshots_file == None:
            self.pretrained = False
            self.pretrained_snapshots = None
        else:
            self.pretrained_snapshots = os.path.join(self.pretrained_snapshots_dir, self.pretrained_snapshots_file)

        self.record_result = os.path.join(self.record_root_dir, 'result')
        os.makedirs(self.record_result, exist_ok=True)

    @staticmethod
    def write_model_summary(file_path, params):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            for key, value in params.items():
                file.write(f"{key}={value}\n")

    @staticmethod
    def get_latest_weight_file(weights_directory):
        files = os.listdir(weights_directory)
        pattern = re.compile(r'Epoch(\d+)\.pth')

        max_epoch = -1
        latest_file = None

        for file in files:
            match = pattern.match(file)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    latest_file = file

        if latest_file is None:
            print(f"No weight files found in directory: {weights_directory}")
        return latest_file

    @staticmethod
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    @staticmethod
    def load_pretrained_weights(model, pretrained_path):
        if pretrained_path:
            print('load weights')
            if os.path.isfile(pretrained_path):
                print(f"Loading pretrained weights from {pretrained_path}")
                model.load_state_dict(MDF_Net.remove_module_prefix(torch.load(pretrained_path)))
            else:
                print(f"No pretrained weights found at {pretrained_path}")

    def check_mdf(self):
        for iteration, (images, dn_images, filename) in enumerate(self.check_loader):
            img_IR = images.cuda()
            dn_img_IR = dn_images.cuda()

            O_hr, _ = self.nn(dn_img_IR, img_IR)
            torchvision.utils.save_image(O_hr, os.path.join(self.record_effect, f'{iteration}.png'))

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  
            torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False  

    def train_mdf(self):
        self.set_seed(3407)
        self.nn = self.nn.cuda()
        print(torch.cuda.current_device())
        if self.pretrained:
            MDF_Net.load_pretrained_weights(self.nn, self.pretrained_snapshots)
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.lr, betas=(0.9, 0.999),
                                     weight_decay=self.weight_decay)
        self.nn.train()
        print("MDF: Start training!")
        for epoch in range(self.train_epochs):
            self.nn.train()
            # print(self.nn)
            for iteration, (images, dn_images) in enumerate(self.train_loader):
                img_IR = images.cuda()
                dn_img_IR = dn_images.cuda()
                O_hr, W= self.nn(dn_img_IR, img_IR)
                loss_mef = self.loss(O_hr, img_IR)
                if self.fuzzy_mode:
                    loss_fuzzy = 0.1*self.fuzzy_loss(O_hr)
                    loss_total = loss_mef + loss_fuzzy
                else:
                    loss_total = loss_mef
                optimizer.zero_grad()
                loss_total.backward()
                nn.utils.clip_grad_norm_(self.nn.parameters(), self.grad_clip)
                optimizer.step()
                if ((iteration + 1) % 10) == 0:
                    if self.fuzzy_mode:
                        print("Loss at iteration", iteration + 1, ":", loss_total.item(), "fuzzy", loss_fuzzy.item(), 
                        "mef", loss_mef.item())
                    else:
                        print("Loss at iteration", iteration + 1, ":", loss_total.item())
            if ((epoch + 1) % 1) == 0:
                torch.save(self.nn.state_dict(), self.record_snapshots + "/Epoch" + str(epoch) + '.pth')
            print(f'Epoch {epoch}')
            self.nn.eval()
            for iteration, (images, dn_images, filename) in enumerate(self.check_loader):
                img_IR = images.cuda()
                dn_img_IR = dn_images.cuda()
                O_hr, _ = self.nn(dn_img_IR, img_IR)
                torchvision.utils.save_image(O_hr, os.path.join(self.record_effect, f'{iteration}.png'))
        print("training end")

    def test_mdf(self):
        self.nn.eval()
        self.nn = self.nn.cuda()
        self.check_mdf()
        for images, dn_images, filename in self.test_loader:
            img_IR = images.cuda()
            dn_img_IR = dn_images.cuda()
            O_hr, _= self.nn(dn_img_IR, img_IR)
            for j in range(O_hr.size(0)):
                torchvision.utils.save_image(O_hr[j], os.path.join(self.record_result, filename[j]))
        print("test set over")

    def load_mdf(self):
        MDF_Net.load_pretrained_weights(self.nn, self.pretrained_snapshots)

    def infer_mdf(self, input):
        self.nn.eval()
        self.nn.cuda()
        [K, C, H, W] = input.size()
        dn_input = torch.nn.functional.interpolate(input, size=[H // 2, W // 2], mode='bilinear', align_corners=False)
        input = input.cuda()
        dn_input = dn_input.cuda()
        O_hr, _= self.nn(dn_input, input)
        return O_hr

