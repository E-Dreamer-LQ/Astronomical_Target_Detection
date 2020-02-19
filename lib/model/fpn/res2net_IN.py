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
from model.fpn.fpn import _FPN
import pickle as cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

__all__ = ['ImageNetRes2Net', 'res2net50', 'res2net101',
           'res2net152', 'res2next50_32x4d', 'se_res2net50',
           'CifarRes2Net', 'res2next29_6cx24wx4scale',
           'res2next29_8cx25wx4scale', 'res2next29_6cx24wx6scale',
           'res2next29_6cx24wx4scale_se', 'res2next29_8cx25wx4scale_se',
           'res2next29_6cx24wx6scale_se']

activate_function = [nn.ReLU,nn.LeakyReLU,nn.PReLU,nn.ELU]
activate = activate_function[1](0.2)

# courtesy: https://github.com/darkstar112358/fast-neural-style/blob/master/neural_style/transformer_net.py
class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = activate
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class Res2NetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=False,  norm_layer=None):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        self.Use_IN = False
        if self.Use_IN:
            if norm_layer is None:
                norm_layer = InstanceNormalization
        else:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
        bottleneck_planes = groups * planes
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        self.conv3 = conv1x1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = activate
        self.se = SEModule(planes * self.expansion) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ImageNetRes2Net(nn.Module):
    def __init__(self, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width=8, scales=2, se=False, norm_layer=None):  ### ori ： 16 - 4
        super(ImageNetRes2Net, self).__init__()
        self.Use_IN =False
        if self.Use_IN:
            if norm_layer is None:
                norm_layer = InstanceNormalization
        else:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d

        planes = [int(width * scales * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]
        # self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = activate
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Res2NetBottleneck, planes[0], layers[0], scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.layer2 = self._make_layer(Res2NetBottleneck, planes[1], layers[1], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.layer3 = self._make_layer(Res2NetBottleneck, planes[2], layers[2], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.layer4 = self._make_layer(Res2NetBottleneck, planes[3], layers[3], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.fc = nn.Linear(planes[3] * Res2NetBottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Res2NetBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, scales=4, groups=1, se=False, norm_layer=None):
        if self.Use_IN:
            if norm_layer is None:
                norm_layer = InstanceNormalization
        else:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, downsample, stride=stride, scales=scales, groups=groups, se=se, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, scales=scales, groups=groups, se=se, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class CifarRes2Net(nn.Module):
    def __init__(self, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width=64, scales=4, se=False, norm_layer=None):
        super(CifarRes2Net, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        planes = [int(width * scales * 2 ** i) for i in range(3)]
        self.inplanes = planes[0]
        self.conv1 = conv3x3(3, planes[0])
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Res2NetBottleneck, planes[0], layers[0], scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.layer2 = self._make_layer(Res2NetBottleneck, planes[1], layers[1], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.layer3 = self._make_layer(Res2NetBottleneck, planes[2], layers[2], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[2] * Res2NetBottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Res2NetBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, scales=4, groups=1, se=False, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, downsample, stride=stride, scales=scales, groups=groups, se=se, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, scales=scales, groups=groups, se=se, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50(**kwargs):
    """Constructs a Res2Net-50 model.
    """
    # model = ImageNetRes2Net([3, 4, 6, 3], **kwargs)
    model = ImageNetRes2Net([1, 1, 1, 1], **kwargs)
    return model


def res2net101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ImageNetRes2Net([3, 4, 23, 3], **kwargs)
    return model


def res2net152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ImageNetRes2Net([3, 8, 36, 3], **kwargs)
    return model


def res2next50_32x4d(**kwargs):
    """Constructs a Res2NeXt-50_32x4d model.
    """
    model = ImageNetRes2Net([3, 4, 6, 3], groups=32, width=4, **kwargs)
    return model


def res2next101_32x8d(**kwargs):
    """Constructs a Res2NeXt-101_32x8d model.
    """
    model = ImageNetRes2Net([3, 4, 23, 3], groups=32, width=8, **kwargs)
    return model


def se_res2net50(**kwargs):
    """Constructs a SE-Res2Net-50 model.
    """
    model = ImageNetRes2Net([3, 4, 6, 3], se=True, **kwargs)
    return model


def res2next29_6cx24wx4scale(**kwargs):
    """Constructs a Res2NeXt-29, 6cx24wx4scale model.
    """
    model = CifarRes2Net([3, 3, 3], groups=6, width=24, scales=4, **kwargs)
    return model


def res2next29_8cx25wx4scale(**kwargs):
    """Constructs a Res2NeXt-29, 8cx25wx4scale model.
    """
    model = CifarRes2Net([3, 3, 3], groups=8, width=25, scales=4, **kwargs)
    return model


def res2next29_6cx24wx6scale(**kwargs):
    """Constructs a Res2NeXt-29, 6cx24wx6scale model.
    """
    model = CifarRes2Net([3, 3, 3], groups=6, width=24, scales=6, **kwargs)
    return model

def res2next29_6cx24wx4scale_se(**kwargs):
    """Constructs a Res2NeXt-29, 6cx24wx4scale-SE model.
    """
    model = CifarRes2Net([3, 3, 3], groups=6, width=24, scales=4, se=True, **kwargs)
    return model


def res2next29_8cx25wx4scale_se(**kwargs):
    """Constructs a Res2NeXt-29, 8cx25wx4scale-SE model.
    """
    model = CifarRes2Net([3, 3, 3], groups=8, width=25, scales=4, se=True, **kwargs)
    return model


def res2next29_6cx24wx6scale_se(**kwargs):
    """Constructs a Res2NeXt-29, 6cx24wx6scale-SE model.
    """
    model = CifarRes2Net([3, 3, 3], groups=6, width=24, scales=6, se=True, **kwargs)
    return model


class resnet(_FPN):
    def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
        self.model_path = '/home/lab30202/lq/ai_future/galaxy_star_detect/fpn_galaxy_star/data/pretrained_model/best_galaxy_star_IN.pth'
        self.dout_base_model =  64   ## 256 -> 24
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _FPN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        resnet = res2net50()
        print("pretrained:",self.pretrained)

        if self.pretrained == True:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            num_ftrs = resnet.fc.in_features
            resnet.fc = nn.Linear(num_ftrs * 49, 3)
            resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

        self.RCNN_layer0 = nn.Sequential(resnet.conv1,resnet.conv2,resnet.conv3, resnet.bn1, resnet.relu, resnet.maxpool)
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

        self.RCNN_cls_score = nn.Linear(256, self.n_classes)
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


### add in 2019.6.8 L1 and L2 regularization
class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.weight_list = self.get_weight(model)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay)

        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        reg_loss = 0
        for name, w in weight_list:
            l1_reg = torch.norm(w,p=1)
            l2_reg = torch.norm(w,p=2)
            reg_loss = reg_loss + l2_reg + l1_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss


if __name__ == "__main__":
    res2net_50 = res2net50()
    from IPython import embed;
    embed()


