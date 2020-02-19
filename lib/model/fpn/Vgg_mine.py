#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:54:02 2019

@author: liuqiang
"""

import torch
from torch import nn, optim
from collections import OrderedDict
import torch.nn.functional as F
from model.fpn.normalization import InstanceNormalization,GroupNorm,GroupBatchnorm2d
normalization = [nn.BatchNorm2d,InstanceNormalization,GroupNorm,GroupBatchnorm2d]
normal = normalization[3]

activate_function = [nn.ReLU,nn.LeakyReLU,nn.PReLU,nn.ELU]
activate = activate_function[1]()

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c0', nn.Conv2d(3, 6, kernel_size=(3, 3),stride=(1,1),padding=(1,1))),
            ('relu0', activate),
            ('c1', nn.Conv2d(6, 16, kernel_size=(3,3),stride=(1,1),padding=(1,1))),
            ('bn1',normal(16)),
            ('relu1', activate),
            ('s2', nn.MaxPool2d(kernel_size=(1,1), stride=2)),
            ("dropout",nn.Dropout(0.2)),
            ('c3', nn.Conv2d(16, 32, kernel_size=(3, 3),padding=(1,1))),
            ('bn2', normal(32)),
            ('relu3', activate),
            ('s4', nn.MaxPool2d(kernel_size=(1,1), stride=1)),
            ("dropout2", nn.Dropout(0.2)),
            ('c5', nn.Conv2d(32, 64, kernel_size=(3, 3),padding=(1,1))),
            ('bn3', normal(64)),
            ('relu5', activate),
            # ('s3', nn.AdaptiveAvgPool2d(7)),
            # ('bn4', normal(64)),
        ]))
        self.fc = nn.Sequential(OrderedDict([
            #('bachnorm',nn.BatchNorm2d(480, affine=False))
            ('f6', nn.Linear(3136, 1024)),
            ('relu6', activate),
            ("dropout1", nn.Dropout(0.4)),
            ('f7', nn.Linear(1024, 120)),
            ('relu7', activate),
            ("dropout2", nn.Dropout(0.4)),
            ('f8', nn.Linear(120, 64)),
        ]))

        self.fc_new = nn.Sequential(OrderedDict([
            ('f9', nn.Linear(64, 3)),
        ]))

    def forward(self, img):
        output = self.convnet(img)
        # AdaptiveAvgpool = nn.AdaptiveAvgPool2d(7)
        # output = AdaptiveAvgpool(output)
        output = output.view(output.shape[0],-1)
        output = self.fc(output)
        output = F.dropout(output,p=0.3)
        return output






