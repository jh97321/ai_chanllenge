from __future__ import print_function

import copy
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet101, Bottleneck
from model.layers import Cosine, AdaptiveConcatPool2d, Flatten


def make_model(args):
    return PDA(args)
        
class PDA(nn.Module):
    def __init__(self, base_features=2048, num_features=256):
        super(PDA, self).__init__()
        num_classes = 4267  
        print(num_classes)

        resnet = resnet50(pretrained=True)
        # resnet = resnet101(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        pda_conv4 = nn.Sequential(
            Bottleneck(2048//2, 2048//4,
                       downsample=nn.Sequential(
                           nn.Conv2d(2048//2, 2048, 1, bias=False),
                           nn.BatchNorm2d(2048)
                       )),
            Bottleneck(2048, 2048//4),
            Bottleneck(2048, 2048//4))
        pda_conv4.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(deepcopy(pda_conv4))
        self.p2 = nn.Sequential(deepcopy(pda_conv4))
        self.p3 = nn.Sequential(deepcopy(pda_conv4))
        self.p4 = nn.Sequential(deepcopy(pda_conv4))

        pool = AdaptiveConcatPool2d
        self.pool_zg_p1 = pool((1, 1))
        self.pool_zg_p2 = pool((1, 1))
        self.pool_zg_p3 = pool((1, 1))
        self.pool_zg_p4 = pool((1, 1))
        self.pool_part_max = nn.MaxPool2d((2, 2))
        self.pool_zp2 = pool((3, 1))
        self.pool_zp3 = pool((5, 1))
        self.pool_zp4 = pool((7, 1))

        self.features = nn.Sequential(
            nn.BatchNorm2d(2*2048, affine=False),
            nn.Conv2d(2*2048, num_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features, affine=False),
            nn.PReLU(num_features),
            Flatten()
        )

        self.fc_id_256_0 = Cosine(num_features, num_classes)
        self.fc_id_256_1 = Cosine(num_features, num_classes)
        self.fc_id_256_2 = Cosine(num_features, num_classes)
        self.fc_id_256_3 = Cosine(num_features, num_classes)      
        
        self.fc_id_256_1_0 = Cosine(num_features, num_classes)
        self.fc_id_256_1_1 = Cosine(num_features, num_classes)

        self.fc_id_256_2_0 = Cosine(num_features, num_classes)
        self.fc_id_256_2_1 = Cosine(num_features, num_classes)
        self.fc_id_256_2_2 = Cosine(num_features, num_classes)
        self.fc_id_256_2_3 = Cosine(num_features, num_classes)

        self.fc_id_256_3_0 = Cosine(num_features, num_classes)
        self.fc_id_256_3_1 = Cosine(num_features, num_classes)
        self.fc_id_256_3_2 = Cosine(num_features, num_classes)
        self.fc_id_256_3_3 = Cosine(num_features, num_classes)
        self.fc_id_256_3_4 = Cosine(num_features, num_classes)
        self.fc_id_256_3_5 = Cosine(num_features, num_classes)


    def forward(self, x):
        base = self.backbone(x)

        p1 = self.p1(base)
        p2 = self.p2(base)
        p3 = self.p3(base)
        p4 = self.p4(base)

        zg_p1 = self.pool_zg_p1(p1)
        zg_p2 = self.pool_zg_p2(p2)
        zg_p3 = self.pool_zg_p3(p3)
        zg_p4 = self.pool_zg_p4(p4)

        zp2 = self.pool_zp2(p2)
        z0_p2 = zp2[:, :, 0:2, :]
        z1_p2 = zp2[:, :, 1:3, :]

        zp3 = self.pool_zp3(p3)
        z0_p3 = zp3[:, :, 0:2, :]
        z1_p3 = zp3[:, :, 1:3, :]
        z2_p3 = zp3[:, :, 2:4, :]
        z3_p3 = zp3[:, :, 3:5, :]

        zp4 = self.pool_zp4(p4)
        z0_p4 = zp4[:, :, 0:2, :]
        z1_p4 = zp4[:, :, 1:3, :]
        z2_p4 = zp4[:, :, 2:4, :]
        z3_p4 = zp4[:, :, 3:5, :]
        z4_p4 = zp4[:, :, 4:6, :]
        z5_p4 = zp4[:, :, 5:7, :]

        fg_p1 = self.features(zg_p1)
        fg_p2 = self.features(zg_p2)
        fg_p3 = self.features(zg_p3)
        fg_p4 = self.features(zg_p4)
        f_g_m = (fg_p1+fg_p2+fg_p3+fg_p4)/4

        f0_p2_max = self.features(self.pool_part_max(z0_p2))
        f1_p2_max = self.features(self.pool_part_max(z1_p2))
        f_p2_m_max = torch.cat([f0_p2_max, f1_p2_max], dim=1)
        f_p2_m = (f0_p2_max+f1_p2_max)/2

        f0_p3_max = self.features(self.pool_part_max(z0_p3))
        f1_p3_max = self.features(self.pool_part_max(z1_p3))
        f2_p3_max = self.features(self.pool_part_max(z2_p3))
        f3_p3_max = self.features(self.pool_part_max(z3_p3))
        f_p3_m_max = torch.cat([f0_p3_max, f1_p3_max, f2_p3_max, f3_p3_max], dim=1)
        f_p3_m = (f0_p3_max+f1_p3_max+f2_p3_max+f3_p3_max)/4

        f0_p4_max = self.features(self.pool_part_max(z0_p4))
        f1_p4_max = self.features(self.pool_part_max(z1_p4))
        f2_p4_max = self.features(self.pool_part_max(z2_p4))
        f3_p4_max = self.features(self.pool_part_max(z3_p4))
        f4_p4_max = self.features(self.pool_part_max(z4_p4))
        f5_p4_max = self.features(self.pool_part_max(z5_p4))
        f_p4_m_max = torch.cat([f0_p4_max, f1_p4_max, f2_p4_max, f3_p4_max, f4_p4_max, f5_p4_max], dim=1)
        f_p4_m = (f0_p4_max+f1_p4_max+f2_p4_max+f3_p4_max+f4_p4_max+f5_p4_max)/6

        l_p1 = self.fc_id_256_0(fg_p1)
        l_p2 = self.fc_id_256_1(fg_p2)
        l_p3 = self.fc_id_256_2(fg_p3)
        l_p4 = self.fc_id_256_3(fg_p4)

        l0_p2_max = self.fc_id_256_1_0(f0_p2_max)
        l1_p2_max = self.fc_id_256_1_1(f1_p2_max)

        l0_p3_max = self.fc_id_256_2_0(f0_p3_max)
        l1_p3_max = self.fc_id_256_2_1(f1_p3_max)
        l2_p3_max = self.fc_id_256_2_2(f2_p3_max)
        l3_p3_max = self.fc_id_256_2_3(f3_p3_max)

        l0_p4_max = self.fc_id_256_3_0(f0_p4_max)
        l1_p4_max = self.fc_id_256_3_1(f1_p4_max)
        l2_p4_max = self.fc_id_256_3_2(f2_p4_max)
        l3_p4_max = self.fc_id_256_3_3(f3_p4_max)
        l4_p4_max = self.fc_id_256_3_4(f4_p4_max)
        l5_p4_max = self.fc_id_256_3_5(f5_p4_max)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, fg_p4, f0_p2_max, f1_p2_max, f0_p3_max, f1_p3_max, f2_p3_max, f3_p3_max, f0_p4_max, f1_p4_max, f2_p4_max, f3_p4_max, f4_p4_max, f5_p4_max], dim=1)
        # predict = torch.cat([fg_p1, fg_p2, fg_p3, fg_p4, f0_p2_max, f1_p2_max, f_p3_m, f_p4_m], dim=1) # 2048
        # predict = torch.cat([f_g_m, f_p2_m, f_p3_m, f_p4_m], dim=1) # 1024

        return predict, fg_p1, fg_p2, f_p2_m_max, fg_p3, f_p3_m_max, fg_p4, f_p4_m_max, l_p1, l_p2, l_p3, l_p4, l0_p2_max, l1_p2_max, l0_p3_max, l1_p3_max, l2_p3_max, l3_p3_max, l0_p4_max, l1_p4_max, l2_p4_max, l3_p4_max, l4_p4_max, l5_p4_max
        # return predict, fg_p1, fg_p2, f_p2_m, fg_p3, f_p3_m, fg_p4, f_p4_m, l_p1, l_p2, l_p3, l_p4, l0_p2_max, l1_p2_max, l0_p3_max, l1_p3_max, l2_p3_max, l3_p3_max, l0_p4_max, l1_p4_max, l2_p4_max, l3_p4_max, l4_p4_max, l5_p4_max

