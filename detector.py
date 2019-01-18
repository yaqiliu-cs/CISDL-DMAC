#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: liuyaqi
"""

import torch
import torch.nn as nn

import math

affine_par = True

class Detector(nn.Module):
    def __init__(self,pool_stride):
        super(Detector, self).__init__()
        'The pooling of images needs to be researched.'
        self.img_pool = nn.AvgPool2d(pool_stride,stride=pool_stride)
        
        self.input_dim = 3
        
        'Feature extraction blocks.'
        self.conv = nn.Sequential(
                nn.Conv2d(self.input_dim, 16, 3, 1, 1),
                nn.BatchNorm2d(16,affine = affine_par),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, 1, 1),
                nn.BatchNorm2d(32,affine = affine_par),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64,affine = affine_par),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128,affine = affine_par),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
        
        
        
        'Detection branch.'
        self.classifier_det = nn.Sequential(
                nn.Linear(128*8*8,1024),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(1024,2),
                )
        
        self._initialize_weights()
        
    def forward(self,x1,x2,m1,m2):
        x1 = self.img_pool(x1)
        x2 = self.img_pool(x2)
        
        x1 = torch.mul(x1,m1)
        x2 = torch.mul(x2,m2)
        
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        
        x1 = x1.view(x1.size(0),-1)
        x2 = x2.view(x2.size(0),-1)
        
        x12_abs = torch.abs(x1-x2)
        
        x_det = self.classifier_det(x12_abs)
        
        
        return x_det
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
