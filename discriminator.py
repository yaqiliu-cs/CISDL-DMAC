#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: liuyaqi
"""
import torch
import torch.nn as nn
import spectral_normalization as SN


leak = 0.1

class Discriminator(nn.Module):
    def __init__(self,pool_stride):
        super(Discriminator, self).__init__()
        
        self.img_pool = nn.AvgPool2d(pool_stride,stride=pool_stride)
        
        self.input_dim = 6
        
        'Feature extraction blocks.'
        self.conv = nn.Sequential(
                SN.SpectralNorm(nn.Conv2d(self.input_dim, 32, 3, 1, 1)),                
                nn.LeakyReLU(leak),
                SN.SpectralNorm(nn.Conv2d(32, 64, 3, 1, 1)),
                nn.LeakyReLU(leak),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                SN.SpectralNorm(nn.Conv2d(64, 128, 3, 1, 1)),
                nn.LeakyReLU(leak),
                SN.SpectralNorm(nn.Conv2d(128, 256, 3, 1, 1)),
                nn.LeakyReLU(leak),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
        
        'Classifier to discriminate whether it is fake or real.'
        self.classifier_dis = SN.SpectralNorm(nn.Linear(256*8*8,1))
        
    def forward(self,x1,x2,m1_0,m2_0,m1_1,m2_1):
        x1 = self.img_pool(x1)
        x2 = self.img_pool(x2)
        
        x1_0 = torch.mul(x1,m1_0)
        x2_0 = torch.mul(x2,m2_0)
        x1_1 = torch.mul(x1,m1_1)
        x2_1 = torch.mul(x2,m2_1)
        
        x1 = torch.cat((x1_0,x1_1),1)
        x2 = torch.cat((x2_0,x2_1),1)
        
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        
        x1 = x1.view(x1.size(0),-1)
        x2 = x2.view(x2.size(0),-1)
        
        x1_dis = self.classifier_dis(x1)
        x2_dis = self.classifier_dis(x2)
        
        
        return x1_dis,x2_dis

