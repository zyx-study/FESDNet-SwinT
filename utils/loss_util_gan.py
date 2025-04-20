import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from utils.common import *
from torchvision import models as tv
from torch.nn.parameter import Parameter
import os

# class DCGANDiscriminator(nn.Module):
#     def __init__(self, img_channels):
#         super(DCGANDiscriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),  # 64
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0)  # 输出一个值
#         )

#     def forward(self, x):
#         return self.model(x)

class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, generator, discriminator, lam=1, lam_p=1):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam = lam
        self.lam_p = lam_p
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, out1, out2, out3, gt1, feature_layers=[2]):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)

        # 感知损失
        loss1 = self.lam_p * self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam * F.l1_loss(out1, gt1)
        loss2 = self.lam_p * self.loss_fn(out2, gt2, feature_layers=feature_layers) + self.lam * F.l1_loss(out2, gt2)
        loss3 = self.lam_p * self.loss_fn(out3, gt3, feature_layers=feature_layers) + self.lam * F.l1_loss(out3, gt3)

        # 对抗损失
        fake_output = self.discriminator(out1)
        real_output = self.discriminator(gt1)
        adv_loss = F.binary_cross_entropy(fake_output, torch.ones_like(fake_output)) + F.binary_cross_entropy(real_output, torch.zeros_like(real_output))

        return loss1 + loss2 + loss3 + adv_loss
     

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss