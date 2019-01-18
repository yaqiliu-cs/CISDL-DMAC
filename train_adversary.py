#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: liuyaqi
"""
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import discriminator as dis
import detector as det
import utils
import dmac_vgg_skip as dmac_vgg
import time

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    
    
class SoftmaxMask(nn.Module):
    def __init__(self):
        super(SoftmaxMask,self).__init__()
        self.softmax = nn.Softmax2d()
        
    def forward(self,x):
        x = self.softmax(x)
        x = torch.chunk(x,2,dim=1)
        return x[0],x[1]
    
def loc_loss_calc(out, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w
    # label shape batch_size x h x w
    # print 'loss computation begin!'
    # label = torch.from_numpy(label).long()
    m = nn.LogSoftmax().cuda(gpu)
    criterion = nn.NLLLoss2d().cuda(gpu)
    out = m(out)
    return criterion(out,label)

def snapshot(model, prefix, epoch, iter):
    print 'taking snapshot ...'
    torch.save(model.state_dict(), prefix + str(epoch) + '_' + str(iter) + '.pth')
    
low_prob = 0.1
up_prob = 0.9
def generate_adap_gt(gt, mask, gpu):
    s0,s1,s2,s3 = gt.size()
    gt1 = gt
    gt0 = torch.ones((s0,s1,s2,s3)).cuda(gpu)-gt1
    gt = torch.clamp(gt,min=low_prob,max=up_prob)
    gt = gt + nn.ReLU()(torch.mul(mask-gt,gt1))
    gt = gt - nn.ReLU()(torch.mul(gt-mask,gt0))
    return gt

def norm_img(image):
    img_temp = image
    img_temp[:,0,:,:] = image[:,0,:,:] + 104.008
    img_temp[:,1,:,:] = image[:,1,:,:] + 116.669
    img_temp[:,2,:,:] = image[:,2,:,:] + 122.675
    img_temp = torch.div(img_temp,255.00)
    img_temp = torch.div(img_temp - 0.5, 0.5)
    return img_temp
    

class AdversaryLearning(object):
    def __init__(self, args):
        self.epoch = args.epoch
        self.pair_list = args.pair_list
        self.epoch_len = args.epoch_len
        self.batch_size = args.batch_size
        self.gpu = args.gpu
        self.loc_update_stride = args.loc_update_stride
        self.snapshot_stride = args.snapshot_stride
        
        
        self.lambda_loc = args.lambda_loc
        self.lambda_det = args.lambda_det
        self.lambda_dis = args.lambda_dis
        
        self.start_epoch_idx = args.start_epoch_idx
        self.start_iter_idx = args.start_iter_idx
        
        self.snapshot_prefix_loc = args.snapshot_prefix_loc
        self.snapshot_prefix_dis = args.snapshot_prefix_dis
        self.snapshot_prefix_det = args.snapshot_prefix_det
        
        self.data_path = args.data_path
        
        self.input_scale = args.input_scale
        
        self.loc = dmac_vgg.DMAC_VGG(args.nolabel, self.gpu, self.input_scale)
        
        self.dis = dis.Discriminator(8)
        self.det = det.Detector(8)
        
        self.adapt_gt_flag = args.adapt_gt_flag
        self.norm_im_flag = args.norm_im_flag
        
        if args.loc_pretrained:
            loc_saved_state_dict = torch.load(args.loc_pretrain_model)
            self.loc.load_state_dict(loc_saved_state_dict)
        
        if args.dis_pretrained:
            dis_saved_state_dict = torch.load(args.dis_pretrain_model)
            self.dis.load_state_dict(dis_saved_state_dict)
            
        if args.det_pretrained:
            det_saved_state_dict = torch.load(args.det_pretrain_model)
            self.det.load_state_dict(det_saved_state_dict)
        
        self.soft_max = SoftmaxMask()
        
        self.CE_Loss = nn.CrossEntropyLoss().cuda(self.gpu)
        self.loss_type = args.loss_type
        self.BCElog_Loss = nn.BCEWithLogitsLoss().cuda(self.gpu)
        self.loss_type = args.loss_type   
        
        self.loc.cuda(self.gpu)
        self.dis.cuda(self.gpu)
        self.det.cuda(self.gpu)
        
        self.soft_max.cuda(self.gpu)
        
        
        self.loc_optimizer = optim.Adam(self.loc.parameters(),lr=args.lr_loc,betas=(args.beta1,args.beta2))
        self.dis_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.dis.parameters()),lr=args.lr_dis,betas=(args.beta1,args.beta2))
        self.det_optimizer = optim.Adam(self.det.parameters(),lr=args.lr_det,betas=(args.beta1,args.beta2))


        print('---------- Networks architecture -------------')
        print_network(self.loc)
        print_network(self.dis)
        print_network(self.det)
        print('-----------------------------------------------')
    
    
        
    def train(self):
        self.train_hist = {}
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        
        self.loc.train()
        self.dis.train()
        self.det.train()
        
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
                if self.norm_im_flag:
                    images1_d = norm_img(images1)
                    images2_d = norm_img(images2)
                else:
                    images1_d = images1
                    images2_d = images2
                
                # gt masks variable
                gt1_ = torch.squeeze(gt1,dim=1).long()
                gt2_ = torch.squeeze(gt2,dim=1).long()
                gt1_ = gt1_.cuda(self.gpu)
                gt2_ = gt2_.cuda(self.gpu)
                
                gt1 = gt1.cuda(self.gpu)
                gt2 = gt2.cuda(self.gpu)
                
                # dis labels variable
                dis_label_gt, dis_label_ge = torch.ones((self.batch_size,1)).cuda(self.gpu), torch.zeros(self.batch_size, 1).cuda(self.gpu)
                
                det_label = labels.cuda(self.gpu)
                
                # localization
                output1, output2 = self.loc(images1,images2)
                
                #localization update
                if (iter+1) % self.loc_update_stride == 0:
                    mask1_0,mask1_1 = self.soft_max(output1)
                    mask2_0,mask2_1 = self.soft_max(output2)
                    self.loc_optimizer.zero_grad()
                    dis_label_1_ge,dis_label_2_ge = self.dis(images1_d,images2_d,mask1_0,mask2_0,mask1_1,mask2_1)
                    det_label_ge = self.det(images1_d,images2_d,mask1_1,mask2_1)
                    #localization net update
                    loc_loss_1 = loc_loss_calc(output1,gt1_,self.gpu)
                    loc_loss_2 = loc_loss_calc(output2,gt2_,self.gpu)
                    if self.loss_type == 'BCE':
                        dis_loss_1_ge_ = self.BCElog_Loss(dis_label_1_ge,dis_label_gt)
                        dis_loss_2_ge_ = self.BCElog_Loss(dis_label_2_ge,dis_label_gt)
                    else:
                        dis_loss_1_ge_ = -dis_label_1_ge.mean()
                        dis_loss_2_ge_ = -dis_label_2_ge.mean()
                    det_loss_ge_ = self.CE_Loss(det_label_ge,det_label)
                    
                                     
                    loc_loss = self.lambda_loc * (loc_loss_1 + loc_loss_2) + self.lambda_dis * (dis_loss_1_ge_ + dis_loss_2_ge_) + self.lambda_det * det_loss_ge_
                    

                    
                    loc_loss.backward()
                    self.loc_optimizer.step()
                    
                    if (iter+1) % 10 == 0:
                        print '********************************************************************************'
                        print 'iter = ',iter, '  epoch = ', epoch, 'completed, loc_loss = ', loc_loss.data.cpu().numpy()
                        print 'iter = ',iter, '  epoch = ', epoch,'completed, loc_loss_1 = ', loc_loss_1.data.cpu().numpy()
                        print 'iter = ',iter, '  epoch = ', epoch,'completed, loc_loss_2 = ', loc_loss_2.data.cpu().numpy()
                        print 'iter = ',iter, '  epoch = ', epoch,'completed, det_loss_ge = ', det_loss_ge_.data.cpu().numpy()
                        print 'iter = ',iter, '  epoch = ', epoch,'completed, dis_loss_1_ge = ', dis_loss_1_ge_.data.cpu().numpy()
                        print 'iter = ',iter, '  epoch = ', epoch,'completed, dis_loss_2_ge = ', dis_loss_2_ge_.data.cpu().numpy()
                
                
                output1 = output1.detach()
                output2 = output2.detach()
                mask1_0,mask1_1 = self.soft_max(output1)
                mask2_0,mask2_1 = self.soft_max(output2)
                
                if self.adapt_gt_flag:
                    gt1 = generate_adap_gt(gt1,mask1_1,self.gpu)
                    gt2 = generate_adap_gt(gt2,mask2_1,self.gpu) 
                gt1_0 = torch.ones((self.batch_size,1,self.input_scale/8,self.input_scale/8)).cuda(self.gpu)-gt1
                gt2_0 = torch.ones((self.batch_size,1,self.input_scale/8,self.input_scale/8)).cuda(self.gpu)-gt2
                #discrimination
                # generated masks
                dis_label_1_ge,dis_label_2_ge = self.dis(images1_d,images2_d,mask1_0,mask2_0,mask1_1,mask2_1)
                det_label_ge = self.det(images1_d,images2_d,mask1_1,mask2_1)
                # gt masks
                dis_label_1_gt,dis_label_2_gt = self.dis(images1_d,images2_d,gt1_0,gt2_0,gt1,gt2)
                det_label_gt = self.det(images1_d,images2_d,gt1,gt2)

                
                self.dis_optimizer.zero_grad()
                self.det_optimizer.zero_grad()
                
                #discriminator update
                if self.loss_type == 'BCE':
                    dis_loss_1_ge = self.BCElog_Loss(dis_label_1_ge,dis_label_ge)
                    dis_loss_2_ge = self.BCElog_Loss(dis_label_2_ge,dis_label_ge)
                
                    dis_loss_1_gt = self.BCElog_Loss(dis_label_1_gt,dis_label_gt)
                    dis_loss_2_gt = self.BCElog_Loss(dis_label_2_gt,dis_label_gt)
                    
                    dis_loss = dis_loss_1_ge + dis_loss_2_ge + dis_loss_1_gt + dis_loss_2_gt
                elif self.loss_type == 'HIG':
                    dis_loss_1_gt = nn.ReLU()(1.0-dis_label_1_gt).mean()
                    dis_loss_1_ge = nn.ReLU()(1.0+dis_label_1_ge).mean()
                    dis_loss_1 = dis_loss_1_gt + dis_loss_1_ge
                    dis_loss_2_gt = nn.ReLU()(1.0-dis_label_2_gt).mean()
                    dis_loss_2_ge = nn.ReLU()(1.0+dis_label_2_ge).mean()
                    dis_loss_2 = dis_loss_2_gt + dis_loss_2_ge
                    dis_loss = dis_loss_1 + dis_loss_2
                    
                det_loss_ge = self.CE_Loss(det_label_ge,det_label)
                det_loss_gt = self.CE_Loss(det_label_gt,det_label)
                
                
                dis_loss.backward()
                self.dis_optimizer.step()
                
                det_loss = det_loss_ge + det_loss_gt
                det_loss.backward()
                self.det_optimizer.step()
                
                if (iter+1) % 10 == 0:
                    print '********************************************************************************'
                    print 'iter = ',iter, '  epoch = ', epoch, 'completed, dis_loss = ', dis_loss.data.cpu().numpy()
                    print 'iter = ',iter, '  epoch = ', epoch,'completed, dis_loss_1_ge = ', dis_loss_1_ge.data.cpu().numpy()
                    print 'iter = ',iter, '  epoch = ', epoch,'completed, dis_loss_2_ge = ', dis_loss_2_ge.data.cpu().numpy()
                    print 'iter = ',iter, '  epoch = ', epoch,'completed, det_loss_ge = ', det_loss_ge.data.cpu().numpy()
                    print 'iter = ',iter, '  epoch = ', epoch,'completed, dis_loss_1_gt = ', dis_loss_1_gt.data.cpu().numpy()
                    print 'iter = ',iter, '  epoch = ', epoch,'completed, dis_loss_2_gt = ', dis_loss_2_gt.data.cpu().numpy()
                    print 'iter = ',iter, '  epoch = ', epoch,'completed, det_loss_gt = ', det_loss_gt.data.cpu().numpy()
                
                
                
                     
                        
                
                if (iter + 1) % self.snapshot_stride == 0:
                    snapshot(self.loc, self.snapshot_prefix_loc, epoch, iter)
                    snapshot(self.dis, self.snapshot_prefix_dis, epoch, iter)
                    snapshot(self.det, self.snapshot_prefix_det, epoch, iter)
                    
                    
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            
            
        
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")
                
                
        
        
        