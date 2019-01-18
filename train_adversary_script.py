#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: liuyaqi
"""
import utils
import os
import train_adversary
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

snapshot_loc_dir = 'data/snapshots-adver-loc/'
snapshot_dis_dir = 'data/snapshots-adver-dis/'
snapshot_det_dir = 'data/snapshots-adver-det/'

if not os.path.exists(snapshot_loc_dir):
    os.makedirs(snapshot_loc_dir)

if not os.path.exists(snapshot_dis_dir):
    os.makedirs(snapshot_dis_dir)
    
if not os.path.exists(snapshot_det_dir):
    os.makedirs(snapshot_det_dir)

cudnn.enabled = False

class ArgsLocal:
    pass

args_ = ArgsLocal()

args_.epoch = 1
args_.pair_list = utils.load_pairs(data_path, subpath_list)
args_.epoch_len = len(args_.pair_list)
args_.batch_size = 24
args_.gpu = 0
args_.input_scale = 256

args_.lambda_loc = 1
args_.lambda_det = 0.01
args_.lambda_dis = 0.01

args_.loc_update_stride = 1
args_.snapshot_prefix_loc = snapshot_loc_dir + 'DMAC_loc_'
args_.snapshot_prefix_dis = snapshot_dis_dir + 'DMAC_dis_'
args_.snapshot_prefix_det = snapshot_det_dir + 'DMAC_det_'
args_.snapshot_stride = 1000

args_.data_path = data_path

args_.start_epoch_idx = 0
args_.start_iter_idx = 0

args_.lr_loc = 0.00001
args_.lr_dis = 0.0002
args_.lr_det = 0.0002
args_.beta1 = 0.5
args_.beta2 = 0.999

args_.loc_pretrained = True
args_.dis_pretrained = False
args_.det_pretrained = False

args_.loc_pretrain_model = 'data/DMAC.pth'


args_.nolabel = 2

args_.adapt_gt_flag = False
args_.norm_im_flag = False
# {'BCE','HIG'}
args_.loss_type = 'BCE'

ta = train_adversary.AdversaryLearning(args_)
ta.train()
