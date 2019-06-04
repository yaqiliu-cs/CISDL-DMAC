#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: liuyaqi
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

import cv2
import random
import numpy as np

import dmac_vgg_skip as dmac_vgg
import os
import math


def load_pairs_csv( input_pairs_csv_file) :
    '''Load input csv file for splicing detection and localization
    '''
    if ( not os.path.isfile( input_pairs_csv_file ) ) :
        raise IOError, "ERROR: cannot locate input splicing task csv file %s" % input_pairs_csv_file
    with open( input_pairs_csv_file, 'r' ) as IN :
        lines = [ line.strip() for line in IN.readlines() ]
    headers = [ h.lower() for h in lines.pop(0).split(',') ]
    assert ( 'image1' in headers ) and ( 'image2' in headers ) and ( 'label' in headers) and ('gt1' in headers) and ('gt2' in headers), "ERROR: csv file error"
    pair_list = []
    for line in lines :
        fields = line.split(',')
        lut = dict( zip( headers, fields ) )
        pair_list.append((lut['image1'], lut['image2' ], lut['label'], lut['gt1'], lut['gt2']))
    
    return pair_list

def imreadtonumpy(data_path,piece,input_scale):
    img1 = np.zeros((input_scale,input_scale,3))   
    img_temp = cv2.imread(os.path.join(data_path,piece)).astype(float)       
    img_original1 = img_temp
    img_temp = cv2.resize(img_temp,(input_scale,input_scale)).astype(float)
    img_temp[:,:,0] = img_temp[:,:,0] - 104.008
    img_temp[:,:,1] = img_temp[:,:,1] - 116.669
    img_temp[:,:,2] = img_temp[:,:,2] - 122.675
    img1[:img_temp.shape[0],:img_temp.shape[1],:] = img_temp
    return img1,img_original1


def chunker(seq, size):
    return (seq[pos:pos+size] for pos in xrange(0,len(seq), size))


def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I


def get_data_from_chunk(data_path,chunk,dim):
    images1 = np.zeros((dim,dim,3,len(chunk)))
    images2 = np.zeros((dim,dim,3,len(chunk)))
    labels = np.zeros((len(chunk)))
    gt1 = np.zeros((dim,dim,1,len(chunk)))
    gt2 = np.zeros((dim,dim,1,len(chunk)))
    for i,piece in enumerate(chunk):
        flip_p = random.uniform(0, 1)
        img_temp = cv2.imread(os.path.join(data_path,piece[0])).astype(float)       
        img_temp = cv2.resize(img_temp,(dim,dim)).astype(float)
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img_temp = flip(img_temp,flip_p)
        images1[:,:,:,i] = img_temp
               
        img_temp = cv2.imread(os.path.join(data_path,piece[1])).astype(float)        
        img_temp = cv2.resize(img_temp,(dim,dim)).astype(float)
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img_temp = flip(img_temp,flip_p)
        images2[:,:,:,i] = img_temp
               
        label_tmp = int(piece[2])
        labels[i] = label_tmp
        if label_tmp == 1:
            gt_temp = cv2.imread(os.path.join(data_path,piece[3]))[:,:,0]
            gt_temp[gt_temp == 255] = 1
            gt_temp = cv2.resize(gt_temp,(dim,dim) , interpolation = cv2.INTER_NEAREST)
            gt_temp = flip(gt_temp,flip_p)
            gt1[:,:,0,i] = gt_temp
               
            gt_temp = cv2.imread(os.path.join(data_path,piece[4]))[:,:,0]
            gt_temp[gt_temp == 255] = 1
            gt_temp = cv2.resize(gt_temp,(dim,dim) , interpolation = cv2.INTER_NEAREST)
            gt_temp = flip(gt_temp,flip_p)
            gt2[:,:,0,i] = gt_temp
            
            

    images1 = images1.transpose((3,2,0,1))
    images1 = torch.from_numpy(images1).float()
    images2 = images2.transpose((3,2,0,1))
    images2 = torch.from_numpy(images2).float()
    a = dmac_vgg.outS(dim)
    gt1 = resize_label(gt1,a)
    gt1 = gt1.transpose((3,2,0,1))
    gt1 = torch.from_numpy(gt1).float()
    gt2 = resize_label(gt2,a)
    gt2 = gt2.transpose((3,2,0,1))
    gt2 = torch.from_numpy(gt2).float()
    labels = torch.from_numpy(labels).long()
    
    flip_im = random.uniform(0, 1)
    if flip_im > 0.5:
        images = images1
        gt = gt1
        images1 = images2
        gt1 = gt2
        images2 = images
        gt2 = gt
    
    return images1, images2, labels, gt1, gt2

def resize_label(label, size):
    label_resized = np.zeros((size,size,1,label.shape[3]))
    interp = nn.UpsamplingBilinear2d(size=(size, size))
    labelVar = Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)

    return label_resized

def load_pairs(data_path, subpath_list):
    pair_list = []
    for subpath in subpath_list:
        list_path = data_path + 'labelfiles/' + subpath
        pair_list_ = load_pairs_csv(list_path)
        pair_list.extend(pair_list_)
        np.random.shuffle(pair_list)
    print 'Pair list length:',len(pair_list)
    return pair_list
    
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def get_NMM(hist,gt):
    gt_size = float(np.sum(gt))
    tp_size = float(hist[1,1])
    fn_size = float(hist[1,0])
    fp_size = float(hist[0,1])
    if gt_size == 0:
        return 0
    nmm = (tp_size - fn_size - fp_size)/gt_size
    if nmm < -1.0:
        nmm = -1.0
    return nmm

def get_MCC(hist):
    tp = float(hist[1,1])
    tn = float(hist[0,0])
    fn = float(hist[1,0])
    fp = float(hist[0,1])
    denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if denominator == 0:
        return 0
    mcc = (tp*tn - fp*fn)/denominator
    return mcc
