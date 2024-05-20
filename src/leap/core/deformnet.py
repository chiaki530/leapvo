import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import pdb

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, dilation=1, bias=None):
        super(DeformConv2d, self).__init__()

        self.offset_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.mask_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation)
        self.deform_conv = ops.DeformConv2d(inc, outc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        out = self.deform_conv(x, offset=offset, mask=mask)
        return out

class DeformNet(nn.Module):
    def __init__(self, inc, hidc, outc, kernel_size=3, norm_fn='instance', levels=[1,2,4]):
        super().__init__()
        
        self.norm_fn = norm_fn            
        conv_list = []
        for i in levels:
            if self.norm_fn == 'group':
                norm_l = nn.GroupNorm(num_groups=8, num_channels=hidc)
                
            elif self.norm_fn == 'batch':
                norm_l = nn.BatchNorm2d(hidc)

            elif self.norm_fn == 'instance':
                norm_l = nn.InstanceNorm2d(hidc)

            elif self.norm_fn == 'none':
                norm_l = nn.Sequential()
            
            conv_list.append(nn.Sequential(
                DeformConv2d(inc, hidc, kernel_size=kernel_size, padding=i, dilation=i),
                norm_l,
                nn.ReLU(inplace=True)
            ))
               
        self.convs = nn.ModuleList(conv_list)
        self.out_conv = nn.Sequential(
            # self.norm1,
            # nn.ReLU(inplace=True),
            nn.Conv2d(hidc * len(levels), outc, kernel_size=1)
        )
    
    def forward(self, x):
        out_list = []
        for conv_l in self.convs:
            out_list.append(conv_l(x))
        out = self.out_conv(torch.cat(out_list, dim=1))
        return x + out
    
class DeformNetV1(nn.Module):
    def __init__(self, inc, hidc, outc, kernel_size=3, norm_fn='instance', levels=[1,2,4]):
        super().__init__()
        
        self.norm_fn = norm_fn            
        conv_list = []
        for i in levels:
            if self.norm_fn == 'group':
                norm_l = nn.GroupNorm(num_groups=8, num_channels=hidc)
                
            elif self.norm_fn == 'batch':
                norm_l = nn.BatchNorm2d(hidc)

            elif self.norm_fn == 'instance':
                norm_l = nn.InstanceNorm2d(hidc)

            elif self.norm_fn == 'none':
                norm_l = nn.Sequential()
            
            conv_list.append(nn.Sequential(
                DeformConv2d(inc, inc, kernel_size=kernel_size, padding=i, dilation=i),
                norm_l,
                nn.ReLU(inplace=True)
            ))
               
        self.convs = nn.ModuleList(conv_list)
        self.out_conv = nn.Sequential(
            nn.Conv2d(inc, hidc, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidc, outc, kernel_size=1)
        )
    
    def forward(self, x):
        y = x
        for conv_l in self.convs:
            y = conv_l(y)
        y = self.out_conv(y + x)
        return y + x