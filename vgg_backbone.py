import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import sys

import misc



class Conv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False, activation=True):
        super().__init__()
        self.activation = activation
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        out = self.relu(self.bn(self.conv(out))) if self.activation else self.conv(out)
        return out

class CNNModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2_1 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv3_1 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_2 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv4_1 = Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_2 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = misc.Flatten()
    
    def forward(self, x, stream='D'):
        dict_fm = {}
        out = self.conv1_1(x)
        out = self.conv1_2(out) 
        key = 'feature_'+stream+'_blk1'
        dict_fm[key] = out

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.mpool(out) 
        key = 'feature_'+stream+'_blk2'
        dict_fm[key] = out

        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out3 = out
        out = self.mpool(out)
        key = 'feature_'+stream+'_blk3'
        dict_fm[key] = out


        out = self.conv4_1(out)
        out = self.conv4_2(out)
        key = 'feature_'+stream+'_blk4'
        dict_fm[key] = out
        
        return out, out3, dict_fm



