from MDF import MDF
from ASMS import ASMS
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision
class ASMF(nn.Module):
    def __init__(self, name, iter, mdf_epochs, Dataset):
        super(ASMF, self).__init__()
        self.asm_subnet=None
        self.mdf_subnet=None
        self.name = name
        self.iter = iter
        self.mdf_epochs = mdf_epochs
        self.Dataset = Dataset

    def init_asm_subnet(self):
        self.asm_subnet = ASMS.ASMS_Net(
        experiment=self.name, 
        # network
        nn_num_layer=8,
        nn_num_channel=64,
        nn_backbone="default",
        nn_lowbound_enable=1,
        # loss
        W_T=100,
        W_Dark=20,
        W_G=1,
        enable_T=1,

        dataset=self.Dataset,
        train_bn=4,
        train_workers=16,
        pretrained=self.name,
        lr=1e-4,
        weight_decay=3e-5,
        grad_clip=3e-5,
        inference_iteration=self.iter
        )

    def init_mdf_subnet(self):
        self.mdf_subnet=MDF.MDF_Net(
            experiment=self.name,
            nn_img_num=self.iter+1,
            nn_layer=6,
            nn_features=64,
            nn_gf_r=3,
            sigmoid_mode=True,
            fuzzy_mode=True,
            loss_sig_g=0.2,
            loss_sig_l=0.5,
            loss_gray=0.3,
            loss_lum=1,
            loss_win=17,
            train_bn=2,
            train_workers=16,
            pretrained= self.name,
            lr=1e-6,
            weight_decay=3e-5,
            grad_clip=0.5,
            train_epoch=self.mdf_epochs            
        )

    def train_asm(self):
        self.init_asm_subnet()
        self.asm_subnet.train_asm()
    
    def train_mdf(self):
        self.init_asm_subnet()
        self.asm_subnet.test_asm_for_train()
        self.init_mdf_subnet()
        self.mdf_subnet.load_mdf()
        self.mdf_subnet.train_mdf()

    def test_mdf(self):
        self.init_mdf_subnet()
        self.mdf_subnet.load_mdf()
        self.mdf_subnet.test_mdf()

    def train(self):
        self.train_asm()
        self.train_mdf()

    def test(self):
        self.init_asm_subnet()
        self.init_mdf_subnet()
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        test_data_loader = self.asm_subnet.get_inputdata()
        asmnet = self.asm_subnet.get_asm()
        asmnet.eval()
        asmnet = asmnet.cuda()
        os.makedirs(os.path.join(current_file_dir, self.name), exist_ok=True)
        self.mdf_subnet.load_mdf()
        for data, filename in test_data_loader:
            data = data.cuda()
            J, t = asmnet(data)
            D = torch.cat([J, data], dim=1)
            D = D.cuda()
            O = self.mdf_subnet.infer_mdf(D)
            for j in range(O.size(0)):
                torchvision.utils.save_image(O[j], os.path.join(current_file_dir, self.name, filename[j]))
