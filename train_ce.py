#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: liuyaqi
"""
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import utils
import dmac_vgg_skip as dmac_vgg
import time

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    

def snapshot(model, prefix, epoch, iter):
    print 'taking snapshot ...'
    torch.save(model.state_dict(), prefix + str(epoch) + '_' + str(iter) + '.pth')

def trainhist_snapshot(train_hist, prefix, epoch, iter):
    filename = prefix + str(epoch) + '_' + str(iter) + '_loss.log'
    file = open(filename,'w')
    file.write(str(train_hist))
    file.close()
    

class CELearning(object):
    def __init__(self, args):
        self.epoch = args.epoch
        self.pair_list = args.pair_list
        self.epoch_len = args.epoch_len
        self.batch_size = args.batch_size
        self.gpu = args.gpu
        self.loc_update_stride = args.loc_update_stride
        self.snapshot_stride = args.snapshot_stride
        self.input_scale = args.input_scale
        self.nolabel = args.nolabel
        
        self.start_epoch_idx = args.start_epoch_idx
        self.start_iter_idx = args.start_iter_idx
        
        self.snapshot_prefix_loc = args.snapshot_prefix_loc
        
        self.data_path = args.data_path
        
        self.loc = dmac_vgg.DMAC_VGG(self.nolabel, self.gpu, self.input_scale)
        
        if args.loc_pretrained:
            loc_saved_state_dict = torch.load(args.loc_pretrain_model)
            self.loc.load_state_dict(loc_saved_state_dict)    
        
        self.loc.cuda(self.gpu)
        
        self.loc_optimizer = optim.Adadelta(self.loc.parameters())
        
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda(self.gpu)
        self.ce_criterion = nn.NLLLoss2d().cuda(self.gpu)
        
        
        
        print('---------- Networks architecture -------------')
        print_network(self.loc)
        print('-----------------------------------------------')
    
    
        
    def train(self):
        self.train_hist = {}
        self.train_hist['loc_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        
        self.loc.train()
        
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.start_epoch_idx, self.epoch):
            epoch_start_time = time.time()
            data_gen = utils.chunker(self.pair_list, self.batch_size)
            if epoch == self.start_epoch_idx:
                start_iter_idx = self.start_iter_idx
            else:
                start_iter_idx = 0
            for iter in range(start_iter_idx, self.epoch_len):
                if iter == self.epoch_len // self.batch_size:
                    break
                
                # read images
                chunk = data_gen.next()
                images1, images2, labels, gt1, gt2 = utils.get_data_from_chunk(self.data_path,chunk,self.input_scale)
                images1 = images1.cuda(self.gpu)
                images2 = images2.cuda(self.gpu)
                
                # gt masks variable
                gt1_ = torch.squeeze(gt1,dim=1).long()
                gt2_ = torch.squeeze(gt2,dim=1).long()
                gt1_ = gt1_.cuda(self.gpu)
                gt2_ = gt2_.cuda(self.gpu)
                
                
                # localization
                output1, output2 = self.loc(images1,images2)
                      
                
                #localization update
                if (iter+1) % self.loc_update_stride == 0:
                    
                    self.loc_optimizer.zero_grad()
                    
                    #localization net update
                    log_o1 = self.logsoftmax(output1)
                    log_o2 = self.logsoftmax(output2)
                    loc_loss_1 = self.ce_criterion(log_o1,gt1_)
                    loc_loss_2 = self.ce_criterion(log_o2,gt2_)
                    
                                     
                    loc_loss = loc_loss_1 + loc_loss_2
                    
                    self.train_hist['loc_loss'].append(loc_loss.data)
                    
                    loc_loss.backward()
                    self.loc_optimizer.step()
                    
                    if (iter+1) % 10 == 0:
                        print '********************************************************************************'
                        print 'iter = ',iter, '  epoch = ', epoch, 'completed, loc_loss = ', loc_loss.data.cpu().numpy()
                        print 'iter = ',iter, '  epoch = ', epoch,'completed, loc_loss_1 = ', loc_loss_1.data.cpu().numpy()
                        print 'iter = ',iter, '  epoch = ', epoch,'completed, loc_loss_2 = ', loc_loss_2.data.cpu().numpy()
                
                if (iter + 1) % self.snapshot_stride == 0:
                    snapshot(self.loc, self.snapshot_prefix_loc, epoch, iter)       
                    trainhist_snapshot(self.train_hist['loc_loss'],self.snapshot_prefix_loc, epoch, iter)
                    self.train_hist['loc_loss'] = []
                    
                    
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            
            
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")
                
                
        
        
        