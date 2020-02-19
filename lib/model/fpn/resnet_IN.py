# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 19:14:38 2019

@author: Administrator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append("../..")
from model.utils.config import cfg
from model.fpn.fpn_resnet import _FPN_res
from model.fpn.normalization import InstanceNormalization,GroupNorm,GroupBatchnorm2d
import pickle as cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

import torch.utils.model_zoo as model_zoo
import pdb

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']
normalization = [nn.BatchNorm2d,InstanceNormalization,GroupNorm,GroupBatchnorm2d]
normal = normalization[3]
activate_function = [nn.ReLU,nn.LeakyReLU,nn.PReLU,nn.ELU]
activate = activate_function[1](0.2)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.norm_layer = normal
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = self.norm_layer(planes)
        self.leakyrelu = activate
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = self.norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.leakyrelu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.norm_layer = normal
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = self.norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        self.bn2 = self.norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = self.norm_layer(planes * 4)
        # self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = activate
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu(out)
        out = self.leakyrelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)
        out = self.leakyrelu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.norm_layer = normal
        self.bn1 = self.norm_layer(16)
        self.leakyrelu = activate
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=1)
        # self.avgpool = nn.AvgPool2d(7)
        ## changed 2019.6.6
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self.norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        # x = F.dropout(x, p=0.2)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    # if pretrained:
    # 	model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    # if pretrained:
    # 	model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    # model = ResNet(BasicBlock, [2, 2, 2, 2])
    # if pretrained:
    # 	model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

class resnet(_FPN_res):
    def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
        self.model_path = '/home/lab30202/lq/ai_future/single_classification/model_save/galxay_star_classification.pth'
        self.dout_base_model = 64    ## 256 -> 24
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _FPN_res.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        resnet = resnet50()
        print("pretrained:",self.pretrained)

        if self.pretrained == True:
            print("Loading pretrained weights from %s" %(self.model_path))
            # state_dict = torch.load(self.model_path)
            # num_ftrs = resnet.fc.in_features
            # resnet.fc = nn.Linear(num_ftrs * 49, 3)
            # resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
            res_pretrained_dict = torch.load(self.model_path)
            resnet_statedict = resnet.state_dict()
            res_pretrained_dict = {k: v for k, v in res_pretrained_dict.items() if k in resnet_statedict}  
            resnet_statedict.update(res_pretrained_dict) 
            num_ftrs = resnet.fc.in_features
            resnet.fc = nn.Linear(num_ftrs * 49, 3)
            resnet.load_state_dict(resnet_statedict) 

        self.RCNN_layer0 = nn.Sequential(resnet.conv1,resnet.conv2,resnet.conv3, resnet.bn1, resnet.leakyrelu, resnet.maxpool)
        self.RCNN_layer1 = nn.Sequential(resnet.layer1)
        self.RCNN_layer2 = nn.Sequential(resnet.layer2)
        self.RCNN_layer3 = nn.Sequential(resnet.layer3)
        self.RCNN_layer4 = nn.Sequential(resnet.layer4)

        # Top layer
        self.RCNN_toplayer = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)  # reduce channel

        # Smooth layers
        self.RCNN_smooth1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.RCNN_latlayer1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer2 = nn.Conv2d( 128, 64, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer3 = nn.Conv2d( 64, 64, kernel_size=1, stride=1, padding=0)

        # ROI Pool feature downsampling
        self.RCNN_roi_feat_ds = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.RCNN_top = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE, padding=0),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(True)
        )

        # self.RCNN_cls_score = nn.Linear(256, self.n_classes)
        self.RCNN_cls_score = nn.Linear(256, 64)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(256, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(256, 4 * self.n_classes)

        # Fix blocks
        for p in self.RCNN_layer0[0].parameters(): p.requires_grad=False
        for p in self.RCNN_layer0[1].parameters(): p.requires_grad=False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_layer3.parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_layer2.parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_layer1.parameters(): p.requires_grad=False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.RCNN_layer0.apply(set_bn_fix)
        self.RCNN_layer1.apply(set_bn_fix)
        self.RCNN_layer2.apply(set_bn_fix)
        self.RCNN_layer3.apply(set_bn_fix)
        self.RCNN_layer4.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_layer0.eval()
            self.RCNN_layer1.eval()
            self.RCNN_layer2.train()
            self.RCNN_layer3.train()
            self.RCNN_layer4.train()

            self.RCNN_smooth1.train()
            self.RCNN_smooth2.train()
            self.RCNN_smooth3.train()

            self.RCNN_latlayer1.train()
            self.RCNN_latlayer2.train()
            self.RCNN_latlayer3.train()

            self.RCNN_toplayer.train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_layer0.apply(set_bn_eval)
            self.RCNN_layer1.apply(set_bn_eval)
            self.RCNN_layer2.apply(set_bn_eval)
            self.RCNN_layer3.apply(set_bn_eval)
            self.RCNN_layer4.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        block5 = self.RCNN_top(pool5)
        fc7 = block5.mean(3).mean(2)
        return fc7

if __name__ == "__main__":
    res50 = resnet50()
    from IPython import embed;
    embed()










