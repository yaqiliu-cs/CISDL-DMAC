"""
@author: liuyaqi
"""
import torch
import torch.nn as nn

import math
import numpy as np


affine_par = True

def outS(i):
    i = int(i)
    i = int(np.floor((i+1)/2.0))
    i = int(np.floor((i+1)/2.0))
    i = int(np.floor((i+1)/2.0))
    return i

###############################################################################
"VGG mudule"
class VGG_feas(nn.Module):

    def __init__(self, features):
        super(VGG_feas, self).__init__()
        self.features = features
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

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


class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_, dilation_, batch_norm=False):
        super(Convblock,self).__init__()
        self.bnflag = batch_norm
        self.convb = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding_, dilation=dilation_)
        if self.bnflag:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.convb(x)
        if self.bnflag:
            x = self.bn(x)
        x = self.relu(x)
        return x

def make_layers(cfg, in_channels = 3, batch_norm=False):
    layers = []
    
    for v in cfg:
        if v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'D512':
            cb = Convblock(in_channels, 512, padding_=2, dilation_ = 2)
            layers += [cb]
            in_channels = 512
        else:
            cb = Convblock(in_channels, v, padding_=1, dilation_ = 1)
            layers += [cb]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'vgg16-3': [64, 64, 'M2', 128, 128, 'M2', 256, 256, 256, 'M2'],
    'vgg16-4': [ 512, 512, 512, 'M1'],
    'vgg16-5': ['D512', 'D512', 'D512', 'M1'],
}

def vgg16_feas(block, cfg_flag, in_channels = 3, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_feas(block(cfg_flag,in_channels), **kwargs)
    return model

###############################################################################
"the correlation layer"

"The mapping index function"
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
    
        
# classify module
class Classify_Module(nn.Module):

    def __init__(self,dilation_series,padding_series,inputscale,NoLabels):
        super(Classify_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            # Pure
            self.conv2d_list.append(nn.Sequential(
                    nn.Conv2d(inputscale,inputscale*2,kernel_size=3,stride=1, padding = padding, dilation = dilation,bias = True),
                    nn.BatchNorm2d(inputscale*2,affine = affine_par),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(inputscale*2,inputscale*2,kernel_size=1),
                    nn.BatchNorm2d(inputscale*2,affine = affine_par),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(inputscale*2,NoLabels,kernel_size=1)
                ))

        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                


    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


def corr_fun(x1, x2, Corr, poolopt_on_corrmat, new_indices):
    
    corr12 = Corr(x1,x2,new_indices)
    corr21 = Corr(x2,x1,new_indices)
    
    corr11 = Corr(x1,x1,new_indices)
    corr22 = Corr(x2,x2,new_indices)
    
    corr12 = poolopt_on_corrmat(corr12)
    corr21 = poolopt_on_corrmat(corr21)
    
    corr11 = poolopt_on_corrmat(corr11)
    corr22 = poolopt_on_corrmat(corr22)
    
    corr1 = torch.cat((corr12,corr11),1)
    corr2 = torch.cat((corr21,corr22),1)
    
    return corr1,corr2



class DMAC_VGG_Module(nn.Module):
    '''
    The DMAC VGG class module
    init parameters:
        block: the vgg block function
        other parameters refer to the DMAC_VGG function
    '''
    def __init__(self,block,NoLabels,gpu_idx, h, w):
        super(DMAC_VGG_Module,self).__init__()
        self.Scale_3 = vgg16_feas(block, cfg['vgg16-3'], 3)
        self.Scale_4 = vgg16_feas(block, cfg['vgg16-4'], 256)
        self.Scale_5 = vgg16_feas(block, cfg['vgg16-5'], 512)        
        
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
        
        
        '''
        DMAC function module
        '''
        self.classifier = self._make_pred_layer(Classify_Module, [6,12,18,24], [6,12,18,24], 16*3, NoLabels)
    

    def forward(self,x1,x2):
        x1_3 = self.Scale_3(x1)
        x2_3 = self.Scale_3(x2)
        c1_3, c2_3 = corr_fun(x1_3, x2_3, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        
        x1_4 = self.Scale_4(x1_3)
        x2_4 = self.Scale_4(x2_3)
        c1_4, c2_4 = corr_fun(x1_4, x2_4, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        
        x1_5 = self.Scale_5(x1_4)
        x2_5 = self.Scale_5(x2_4)
        c1_5, c2_5 = corr_fun(x1_5, x2_5, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        
        c1 = torch.cat((c1_3,c1_4,c1_5),1)
        c2 = torch.cat((c2_3,c2_4,c2_5),1)
        
        x1 = self.classifier(c1)
        x2 = self.classifier(c2)
        
        return x1,x2#,c1,c2
    
    def _make_pred_layer(self, block, dilation_series, padding_series, inputscale, NoLabels):
        return block(dilation_series,padding_series,inputscale,NoLabels)


def DMAC_VGG(NoLabels, gpu_idx, dim):
    '''The interface function of DMAC. Users only needs to call this function for training or testing.
    INPUT:
        NoLabels: the number of output class. default 2
        gpu_idx: the gpu used
        dim: the input size of the image. default 256
    OUTPUT:
        DMAC model
    '''
    w = outS(dim)
    h = outS(dim)
    model = DMAC_VGG_Module(make_layers,NoLabels,gpu_idx, h, w)
    return model
