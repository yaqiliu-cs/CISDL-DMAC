#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: liuyaqi
"""
import utils
import os
import train_ce
import torch.backends.cudnn as cudnn

subpath_list = ['train2014_combine_1_fore_diff.csv',
                'train2014_combine_1_fore_easy.csv',
                'train2014_combine_1_fore_medi.csv',
                'train2014_combine_1_neg_diff.csv',
                'train2014_combine_1_neg_easy.csv',
                'train2014_combine_1_neg_medi.csv',
                'train2014_combine_1_back_diff.csv',
                'train2014_combine_1_back_easy.csv',
                'train2014_combine_1_back_medi.csv',
                'train2014_combine_2_fore_diff.csv',
                'train2014_combine_2_fore_easy.csv',
                'train2014_combine_2_fore_medi.csv',
                'train2014_combine_2_neg_diff.csv',
                'train2014_combine_2_neg_easy.csv',
                'train2014_combine_2_neg_medi.csv',
                'train2014_combine_2_back_diff.csv',
                'train2014_combine_2_back_easy.csv',
                'train2014_combine_2_back_medi.csv',
                'train2014_combine_3_fore_diff.csv',
                'train2014_combine_3_fore_easy.csv',
                'train2014_combine_3_fore_medi.csv',
                'train2014_combine_3_neg_diff.csv',
                'train2014_combine_3_neg_easy.csv',
                'train2014_combine_3_neg_medi.csv',
                'train2014_combine_3_back_diff.csv',
                'train2014_combine_3_back_easy.csv',
                'train2014_combine_3_back_medi.csv',
                'train2014_combine_4_fore_diff.csv',
                'train2014_combine_4_fore_easy.csv',
                'train2014_combine_4_fore_medi.csv',
                'train2014_combine_4_neg_diff.csv',
                'train2014_combine_4_neg_easy.csv',
                'train2014_combine_4_neg_medi.csv',
                'train2014_combine_4_back_diff.csv',
                'train2014_combine_4_back_easy.csv',
                'train2014_combine_4_back_medi.csv',
                'train2014_combine_5_fore_diff.csv',
                'train2014_combine_5_fore_easy.csv',
                'train2014_combine_5_fore_medi.csv',
                'train2014_combine_5_neg_diff.csv',
                'train2014_combine_5_neg_easy.csv',
                'train2014_combine_5_neg_medi.csv',
                'train2014_combine_5_back_diff.csv',
                'train2014_combine_5_back_easy.csv',
                'train2014_combine_5_back_medi.csv']


data_path = 'mfc2018/dataprepare/DMAC-COCO/train2014/'

snapshot_loc_dir = 'data/snapshots-loc/'

if not os.path.exists(snapshot_loc_dir):
    os.makedirs(snapshot_loc_dir)

cudnn.enabled = False

class ArgsLocal:
    pass

args_ = ArgsLocal()

args_.epoch = 3
args_.pair_list = utils.load_pairs(data_path, subpath_list)
args_.epoch_len = len(args_.pair_list)
args_.batch_size = 24
args_.gpu = 0



args_.loc_update_stride = 1
args_.snapshot_prefix_loc = snapshot_loc_dir + 'DMAC_loc_'
args_.snapshot_stride = 1000

args_.data_path = data_path

args_.start_epoch_idx = 0
args_.start_iter_idx = 0


args_.loc_pretrained = True

args_.loc_pretrain_model = 'DMAC_vgg_pretrained_init.pth'

args_.nolabel = 2
args_.input_scale = 256

tc = train_ce.CELearning(args_)
tc.train()
