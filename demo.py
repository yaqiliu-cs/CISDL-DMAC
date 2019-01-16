#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
A simple demo for testing DMAC-adv
"""
import torch
from torch.autograd import Variable
import torch.nn as nn

import sys
import os
import cv2
import numpy as np

import dmac_vgg_skip as dmac_vgg

class SoftmaxMask(nn.Module):
    def __init__(self):
        super(SoftmaxMask,self).__init__()
        self.softmax = nn.Softmax2d()
        
    def forward(self,x):
        x = self.softmax(x)
        return x[:,1,:,:]
    
def vis_fun(output0,output1,outputname1,outputname2):
    softmax_mask = SoftmaxMask()
    
    output0 = softmax_mask(output0).cpu().data[0].numpy()
    output1 = softmax_mask(output1).cpu().data[0].numpy()
        
    output0 = np.uint8(output0 * 255)
    output1 = np.uint8(output1 * 255)
    mask0 = cv2.applyColorMap(output0,cv2.COLORMAP_JET)
    mask1 = cv2.applyColorMap(output1,cv2.COLORMAP_JET)
    cv2.imwrite(outputname1+'_colormask.png', mask0)
    cv2.imwrite(outputname2+'_colormask.png', mask1)

def CISDL(probe,donor):
    if os.path.exists(probe) and os.path.exists(donor):
        print probe + ' and ' + donor + ' are proccessed!'
    else:
        print 'File none exsit!'
        return
    input_scale = 256
    upsample_layer = nn.UpsamplingBilinear2d(size=(input_scale, input_scale))
    
    img1 = np.zeros((input_scale,input_scale,3))
    img2 = np.zeros((input_scale,input_scale,3))
        
    img = cv2.imread(probe)
    if img is None:
        print 'Reading error occurred!'
        return
    img = img.astype(float)
    img = cv2.resize(img,(input_scale,input_scale)).astype(float)
    img[:,:,0] = img[:,:,0] - 104.008
    img[:,:,1] = img[:,:,1] - 116.669
    img[:,:,2] = img[:,:,2] - 122.675
    img1[:img.shape[0],:img.shape[1],:] = img
        
    img = cv2.imread(donor)
    if img is None:
        print 'Reading error occurred!'
        return
    img = img.astype(float)
    img = cv2.resize(img,(input_scale,input_scale)).astype(float)
    img[:,:,0] = img[:,:,0] - 104.008
    img[:,:,1] = img[:,:,1] - 116.669
    img[:,:,2] = img[:,:,2] - 122.675
    img2[:img.shape[0],:img.shape[1],:] = img
    
    gpu0 = 0
    model = dmac_vgg.DMAC_VGG(2, gpu0, input_scale)
    model.eval()
    model.cuda(gpu0)
    model_path = 'DMAC-adv.pth'
    saved_state_dict = torch.load(model_path)
    model.load_state_dict(saved_state_dict)
    
    image1 = Variable(torch.from_numpy(img1[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda(gpu0)
    image2 = Variable(torch.from_numpy(img2[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda(gpu0)
    output = model(image1, image2)
    output0 = upsample_layer(output[0])
    output1 = upsample_layer(output[1])
    outputname1 = probe[:-4]
    outputname2 = donor[:-4]
    vis_fun(output0,output1,outputname1,outputname2)


def main():
    if len(sys.argv) > 1:
        probe = sys.argv[1]
    else:
        probe = 'im1_1.jpg'
        
    if len(sys.argv) > 2:
        donor = sys.argv[2]
    else:
        donor = 'im1_2.jpg'

    CISDL(probe,donor)
    

if __name__ == "__main__":
    main()        
        