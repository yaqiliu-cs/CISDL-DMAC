#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: liuyaqi
"""
import torch
import torch.nn as nn


import cv2
import numpy as np

import os

import dmvn

from utils import load_pairs_csv,imreadtonumpy

import time
from utils import fast_hist,get_NMM,get_MCC


class Test_COCO(object):
    def __init__(self, args):
        
        """
        data prepare
        """
        self.data_path = args.data_path
        self.list_path = args.list_path
        self.test_num = args.test_num
        pair_list_full = load_pairs_csv(os.path.join(self.data_path,self.list_path))
        if len(pair_list_full) < self.test_num:
            print 'The test number is larger than the data length!'
            return
        self.pair_list = pair_list_full[:self.test_num]
        
        """
        model initialization
        """
        self.model_path = args.model_path
        
        self.gpu = args.gpu_idx
        self.nolabel = args.nolabel
        self.input_scale = args.input_scale
        
        self.loc = dmvn.DMVN_VGG(self.nolabel, self.gpu, self.input_scale)
        loc_saved_state_dict = torch.load(self.model_path)
        self.loc.load_state_dict(loc_saved_state_dict)
        self.loc.cuda(self.gpu)
        
        """
        visualization options
        """
        self.vis_flag = args.vis_flag
        self.vis_base_path = args.vis_base_path
        self.vis_task_path = args.vis_task_path
        if self.vis_flag == True:
            if not os.path.exists(self.vis_base_path):
                os.makedirs(self.vis_base_path)
                print 'mkdir '+self.vis_base_path
            else:
                print self.vis_base_path + ' exist!'
            
            if not os.path.exists(os.path.join(self.vis_base_path,self.vis_task_path)):
                os.makedirs(os.path.join(self.vis_base_path,self.vis_task_path))
                print 'mkdir ' + os.path.join(self.vis_base_path,self.vis_task_path)
            else:
                print os.path.join(self.vis_base_path,self.vis_task_path) + ' exist!'
                os.makedirs(os.path.join(self.vis_base_path,self.vis_task_path))
                print 'mkdir ' + os.path.join(self.vis_base_path,self.vis_task_path)
        
        """
        score options
        """
        self.score_save_path = args.score_save_path
        self.score_save_filename = args.score_save_filename
        
        if not os.path.exists(self.score_save_path):
            os.makedirs(self.score_save_path)
            print 'mkdir ' + self.score_save_path
            
        self.NMM1 = 0
        self.NMM2 = 0
        self.NMM = 0
        
        self.MCC1 = 0
        self.MCC2 = 0
        self.MCC = 0
        
        self.iou1 = 0
        self.iou2 = 0
        self.iou = 0
        
        self.interp256 = nn.UpsamplingBilinear2d(size=(self.input_scale, self.input_scale))
        self.sigmoid_mod = nn.Sigmoid()
        
    def test(self):
        self.loc.eval()

        start = time.time()
        count = 0
        for piece in self.pair_list:            
            img1,_ = imreadtonumpy(self.data_path,piece[0],self.input_scale)
            img2,_ = imreadtonumpy(self.data_path,piece[1],self.input_scale)
        
            label_tmp = int(piece[2])
            if label_tmp == 1:
                gt_temp = cv2.imread(os.path.join(self.data_path,piece[3]))[:,:,0]
                gt_temp[gt_temp == 255] = 1
                gt_temp = cv2.resize(gt_temp,(self.input_scale,self.input_scale) , interpolation = cv2.INTER_NEAREST)
                gt1 = gt_temp
               
                gt_temp = cv2.imread(os.path.join(self.data_path,piece[4]))[:,:,0]
                gt_temp[gt_temp == 255] = 1
                gt_temp = cv2.resize(gt_temp,(self.input_scale,self.input_scale) , interpolation = cv2.INTER_NEAREST)
                gt2 = gt_temp
            else:
                gt1 = np.zeros((self.input_scale,self.input_scale))
                gt2 = np.zeros((self.input_scale,self.input_scale))


            image1 = torch.from_numpy(img1[np.newaxis, :].transpose(0,3,1,2)).float().cuda(self.gpu)
            image2 = torch.from_numpy(img2[np.newaxis, :].transpose(0,3,1,2)).float().cuda(self.gpu)
            output = self.loc(image1, image2)
            
            output0 = self.sigmoid_mod(output[0]).cpu().data[0].numpy()
            output0 = output0[:,:self.input_scale,:self.input_scale]
            output1 = self.sigmoid_mod(output[1]).cpu().data[0].numpy()
            output1 = output1[:,:self.input_scale,:self.input_scale]
#        
            output0 = output0.squeeze(axis=0)
            output0[output0 > 0.5] = 1
            output0[output0 <= 0.5] = 0   
            output1 = output1.squeeze(axis=0)
            output1[output1 > 0.5] = 1
            output1[output1 <= 0.5] = 0
            output0 = output0.astype(int)
            output1 = output1.astype(int)
            
            hist1 = fast_hist(gt1.flatten(),output0.flatten(),self.nolabel+1).astype(float)
            hist2 = fast_hist(gt2.flatten(),output1.flatten(),self.nolabel+1).astype(float)
        
            self.NMM1 += get_NMM(hist1,gt1)
            self.NMM2 += get_NMM(hist2,gt2)
        
            self.MCC1 += get_MCC(hist1)
            self.MCC2 += get_MCC(hist2)
        
            iou1_tmp = np.diag(hist1) / (hist1.sum(1) + hist1.sum(0) - np.diag(hist1))
            iou2_tmp = np.diag(hist2) / (hist2.sum(1) + hist2.sum(0) - np.diag(hist2))
            self.iou1 += iou1_tmp[1]
            self.iou2 += iou2_tmp[1]
            
            if self.vis_flag == True:
                self.vis_fun(piece,img1,img2,gt1,gt2,output)
            
            print 'item ' + str(count) + ' processed!'
            count += 1
        
        stop = time.time()
        
        print (stop-start)/float(count)
            
        self.iou = (self.iou1 + self.iou2) / float(count * 2)
        self.iou1 = self.iou1 / float(count)
        self.iou2 = self.iou2 / float(count)
        
        self.NMM = (self.NMM1 + self.NMM2) / float(count * 2)
        self.NMM1 = self.NMM1 / float(count)
        self.NMM2 = self.NMM2 / float(count)
        
        self.MCC = (self.MCC1 + self.MCC2) / float(count * 2)
        self.MCC1 = self.MCC1 / float(count)
        self.MCC2 = self.MCC2 / float(count)
        
        self.printscores()

    def printscores(self):
        self.iou1 = ("%.4f" % self.iou1)
        self.iou2 = ("%.4f" % self.iou2)
        self.iou = ("%.4f" % self.iou)
        
        self.NMM1 = ("%.4f" % self.NMM1)
        self.NMM2 = ("%.4f" % self.NMM2)
        self.NMM = ("%.4f" % self.NMM)
        
        self.MCC1 = ("%.4f" % self.MCC1)
        self.MCC2 = ("%.4f" % self.MCC2)
        self.MCC = ("%.4f" % self.MCC)
        
        print "iou   1   = ",self.iou1
        print "iou   2   = ",self.iou2
        print "iou  avg  = ",self.iou
        print "NMM   1   = ",self.NMM1
        print "NMM   2   = ",self.NMM2
        print "NMM  avg  = ",self.NMM
        print "MCC   1   = ",self.MCC1
        print "MCC   2   = ",self.MCC2
        print "MCC  avg  = ",self.MCC
        
        save_file = os.path.join(self.score_save_path,self.score_save_filename)
        
        file_out = open(save_file, 'w')
        headers = 'iou1,' + 'iou2,' + 'iou,' \
                + 'NMM1,' + 'NMM2,' + 'NMM,' \
                + 'MCC1,' + 'MCC2,' + 'MCC\n'
        
        file_out.write(headers)
        write_line = self.iou1 + ',' + self.iou2 + ',' + self.iou + ',' \
                   + self.NMM1 + ',' + self.NMM2 + ',' + self.NMM + ',' \
                   + self.MCC1 + ',' + self.MCC2 + ',' + self.MCC + '\n'
        file_out.write(write_line)
        file_out.close()
                   
        
    def vis_fun(self,piece,img1,img2,gt1,gt2,output):
        piece_0 = piece[0].split('/')
        piece_1 = piece[1].split('/')
        p_0_len = len(piece_0)
        p_1_len = len(piece_1)
        piece_0_ = piece_0[p_0_len-1].split('.')
        piece_1_ = piece_1[p_1_len-1].split('.')
        piece_dir = os.path.join(self.vis_base_path, self.vis_task_path, piece_0_[0] + '_' + piece_1_[0])        
        if not os.path.exists(piece_dir):
            os.makedirs(piece_dir)
            print 'mkdir '+ piece_dir
        else:
            print piece_dir + ' exist!'
        
        img1[:,:,0] = img1[:,:,0] + 104.008
        img1[:,:,1] = img1[:,:,1] + 116.669
        img1[:,:,2] = img1[:,:,2] + 122.675
        img2[:,:,0] = img2[:,:,0] + 104.008
        img2[:,:,1] = img2[:,:,1] + 116.669
        img2[:,:,2] = img2[:,:,2] + 122.675
        gt1 = np.uint8(gt1 * 255)
        gt2 = np.uint8(gt2 * 255)
        
        output0 = self.softmax_mask(self.interp256(output[0])).cpu().data[0].numpy()
        output1 = self.softmax_mask(self.interp256(output[1])).cpu().data[0].numpy()
        
        output0 = np.uint8(output0 * 255)
        output1 = np.uint8(output1 * 255)
        mask0 = cv2.applyColorMap(output0,cv2.COLORMAP_JET)
        mask1 = cv2.applyColorMap(output1,cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(piece_dir,'img_1.jpg'), img1)
        cv2.imwrite(os.path.join(piece_dir,'img_2.jpg'), img2)
        cv2.imwrite(os.path.join(piece_dir,'img_1_gt.png'), gt1)
        cv2.imwrite(os.path.join(piece_dir,'img_2_gt.png'), gt2)
        cv2.imwrite(os.path.join(piece_dir,'img_1_mask.png'), output0)
        cv2.imwrite(os.path.join(piece_dir,'img_2_mask.png'), output1)
        cv2.imwrite(os.path.join(piece_dir,'img_1_colormask.png'), mask0)
        cv2.imwrite(os.path.join(piece_dir,'img_2_colormask.png'), mask1)
        
        

"""
The parameters for testing.
"""
class ArgsLocal:
    pass

args = ArgsLocal()

args.data_path = os.getcwd()
args.list_path = os.path.join(args.data_path,'val2014_combine_fore_norm.csv')
args.test_num = 3000

args.model_path = 'DMVN-BN.pth'

args.gpu_idx = 0
args.nolabel = 1
args.input_scale = 256

args.vis_flag = False
args.vis_base_path = 'vis'
args.vis_task_path = 'coco_combine_6'

args.score_save_path = 'scores'
args.score_save_filename = 'val2014_combine_fore_norm_dmvn_bn.csv'

tc = Test_COCO(args)
tc.test()