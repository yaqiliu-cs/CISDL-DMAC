#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: liuyaqi
"""
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


use_bn = True

###############################################################################

def get_mapping_indices(n_rows, n_cols ):
    new_indices = []
    for r_x in range( n_rows ):
        for c_x in range( n_cols ):
            for r_b in range( n_rows ):
                r_a = ( r_b + r_x ) % n_rows
                for c_b in range( n_cols ):
                    c_a = ( c_b + c_x ) % n_cols
                    idx_a = r_a * n_cols + c_a
                    idx_b = r_b * n_cols + c_b
                    idx = idx_a * ( n_rows * n_cols ) + idx_b
                    new_indices.append( idx )
    return new_indices

class Correlation_Module(nn.Module):
    
    def __init__(self):
        super(Correlation_Module, self).__init__()

    
    def forward(self, x1, x2, new_indices):
        
        [bs1, c1, h1, w1] = x1.size()
        pixel_num = h1 * w1
        
        x1 = torch.div(x1.view(bs1, c1, pixel_num),c1)
        x2 = x2.view(bs1, c1, pixel_num).permute(0,2,1)
        
        x1_x2 = torch.bmm(x2,x1)
        
        x1_x2 = torch.index_select(x1_x2.view(bs1, -1),1,new_indices)
        x1_x2 = x1_x2.view(bs1,pixel_num,h1,w1)
        
        return x1_x2
        
class Poolopt_on_Corrmat(nn.Module):
    
    def __init__(self, select_indices):
        super(Poolopt_on_Corrmat, self).__init__()
        self.select_indices = select_indices
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        
    def forward(self, corr):
        
        max_corr = torch.max(corr,1,keepdim=True)
        avg_corr = torch.mean(corr,1,keepdim=True)
        
        value1 = torch.sum(corr,3)
        value = torch.sum(value1,2)
        
        sort_v,sort_v_idx = torch.sort(value,1,descending=True)
        sort_v_idx = torch.index_select(sort_v_idx,1,self.select_indices)
        [bs,idx_num] = sort_v_idx.size()
        
        for bs_idx in range(bs):
            sort_corr_tmp = torch.index_select(torch.unsqueeze(corr[bs_idx],0),1,sort_v_idx[bs_idx])
            if bs_idx == 0:
                sort_corr = sort_corr_tmp
            else:
                sort_corr = torch.cat([sort_corr, sort_corr_tmp],0)
        corr = torch.cat((max_corr[0],avg_corr, sort_corr),1)
        return corr



###############################################################################

class Inception_pool(nn.Module):

    def __init__(self, in_channels=8, nb_inceptors=32):
        super(Inception_pool, self).__init__()
        self.branch_pool = BasicConv2d(in_channels, nb_inceptors, kernel_size=1)
        
        self.branch1x1 = BasicConv2d(in_channels, nb_inceptors, kernel_size=1)        

        self.branch3x3dbl_1 = BasicConv2d(in_channels, nb_inceptors, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(nb_inceptors, nb_inceptors, kernel_size=3, padding=1)
        
        self.branch5x5_1 = BasicConv2d(in_channels, nb_inceptors, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(nb_inceptors, nb_inceptors, kernel_size=5, padding=2)

        
        
        self.final_conv = BasicConv2d(nb_inceptors*4, nb_inceptors, kernel_size=3, padding=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        branch1x1 = self.branch1x1(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        

        outputs = [branch_pool, branch1x1, branch3x3dbl, branch5x5]
        final_output = torch.cat(outputs, 1)
        final_output = self.final_conv(final_output)
        return final_output
		
		
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        if use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)
		

class Inception_mask_small(nn.Module):

    def __init__(self, in_channels=8, nb_inceptors=4):
        super(Inception_mask_small, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, nb_inceptors, kernel_size=1)

        self.branch3x3 = BasicConv2d(in_channels, nb_inceptors, kernel_size=3, padding=1)
        
        self.branch5x5 = BasicConv2d(in_channels, nb_inceptors, kernel_size=5, padding=2)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3(x)
        
        branch5x5 = self.branch5x5(x)

        outputs = [branch1x1, branch3x3, branch5x5]
        
        final_output = torch.cat(outputs, 1)
        return final_output
		
class Inception_mask_large(nn.Module):

    def __init__(self, in_channels=8, nb_inceptors=4):
        super(Inception_mask_large, self).__init__()
		
        self.branch5x5 = BasicConv2d(in_channels, nb_inceptors, kernel_size=5, padding=2)

        self.branch7x7 = BasicConv2d(in_channels, nb_inceptors, kernel_size=7, padding=3)
        
        self.branch11x11 = BasicConv2d(in_channels, nb_inceptors, kernel_size=11, padding=5)

    def forward(self, x):

        branch5x5 = self.branch5x5(x)

        branch7x7 = self.branch7x7(x)
        
        branch11x11 = self.branch11x11(x)

        outputs = [branch5x5, branch7x7, branch11x11]
        
        final_output = torch.cat(outputs, 1)
        return final_output
		
		
class mask_decoder(nn.Module):
    def __init__(self, input_dim=8):
        super(mask_decoder, self).__init__()
        self.incep1 = Inception_mask_small(input_dim, nb_inceptors=4)
        self.incep2 = Inception_mask_small(12, nb_inceptors=3)
        self.incep3 = Inception_mask_small(9, nb_inceptors=2)
        self.incep4 = Inception_mask_small(6, nb_inceptors=1)
        self.incep5 = Inception_mask_large(3, nb_inceptors=1)
        self.conv = nn.Conv2d(3, 1, 1, stride=1, padding=0)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.upsample(x,size=(32,32),mode='bilinear',align_corners=True)
	x = self.incep1(x)
        x = F.upsample(x,size=(64,64),mode='bilinear',align_corners=True)
        x = self.incep2(x)
        x = F.upsample(x,size=(128,128),mode='bilinear',align_corners=True)
        x = self.incep3(x)
        x = F.upsample(x,size=(256,256),mode='bilinear',align_corners=True)
        x = self.incep4(x)
        x = self.incep5(x)
        x = self.conv(x)	
        return x

#################################################
# VGG features
################################################
def base_feature():
    vgg = torchvision.models.vgg16()
    model = nn.Sequential()
    [model.add_module(name=str(i), module=layer) for i, layer in enumerate(list(vgg.features)[:-7])]
    return model

class dmvn(nn.Module):
    def __init__(self,NoLabels,gpu_idx,h,w):
        super(dmvn, self).__init__()
        
        self.base = base_feature()
        
        
        '''
        Inception function module
        '''
        
        self.pool_inceptor = Inception_pool()
        
        self.classifier = mask_decoder(64)
        
        '''
        The module used to measure the correspondence of two tensors
        '''        
        self.gpu = gpu_idx
        new_indices = get_mapping_indices(h, w)        
        self.new_indices = torch.tensor(new_indices,dtype=torch.long).cuda(self.gpu)
        self.Corr = Correlation_Module()
        
        '''
        The corr maps pooling modules
        '''
        sort_indices = [0,1,2,3,4,5]
        sort_indices = torch.tensor(sort_indices,dtype=torch.long).cuda(self.gpu)
        self.poolopt_on_corrmat = Poolopt_on_Corrmat(sort_indices)
        
        

    def forward(self,x1,x2):
        # base feature extraction
        x1 = self.base(x1)
        x2 = self.base(x2)
        # feature correlation
        corr12 = self.Corr(x1,x2,self.new_indices)
        corr21 = self.Corr(x2,x1,self.new_indices)
        corr11 = self.Corr(x1,x1,self.new_indices)
        corr22 = self.Corr(x2,x2,self.new_indices)
    
        corr12 = self.poolopt_on_corrmat(corr12)
        corr21 = self.poolopt_on_corrmat(corr21)
    
        corr11 = self.poolopt_on_corrmat(corr11)
        corr22 = self.poolopt_on_corrmat(corr22)
		
        corr12 = self.pool_inceptor(corr12)
        corr21 = self.pool_inceptor(corr21)
        corr11 = self.pool_inceptor(corr11)
        corr22 = self.pool_inceptor(corr22)
        
        corr1 = torch.cat((corr12,corr11),1)
        corr2 = torch.cat((corr21,corr22),1)
        
        # output masks
        mask1 = self.classifier(corr1)
        mask2 = self.classifier(corr2)
        
        return mask1, mask2
    
def DMVN_VGG(NoLabels,gpu_idx, dim):
    w = dim/16
    h = dim/16
    model = dmvn(NoLabels,gpu_idx, h, w)
    return model
