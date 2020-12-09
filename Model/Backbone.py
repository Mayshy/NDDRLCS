#!/usr/bin/env python  
#-*- coding:utf-8 _*-
from collections import OrderedDict

import torch.nn as nn
from resnest.torch import resnest50, resnest101, resnest200, resnest269
import torch.nn.functional as F
from torchvision.models.inception import BasicConv2d

from Model.MTLModel import NddrDenseNet, apply_cross, NddrLayer, _DenseBlock, _Transition
from Model._utils import IntermediateLayerGetter, testModel
from torchvision import models
import torch


class Inception_BB(nn.Module):
    version_set = ['inception_v3']

    def __init__(self, in_channels, pretrained=False, version='inception_v3'):
        super(Inception_BB, self).__init__()
        version = version.strip()
        if version == 'inception_v3':
            inception = models.inception_v3(pretrained, init_weights=True)
        else:
            raise NotImplementedError('version {} is not supported as of now'.format(version))

        inception.Conv2d_1a_3x3 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2)
        return_layers = {'Mixed_7c': 'out'}
        self.backbone = IntermediateLayerGetter(inception, return_layers)
        self.out_channels = 2048

    def forward(self, x):
        return self.backbone(x)

class VGG_BB(nn.Module):
    version_set = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']

    def __init__(self, in_channels, pretrained=False, version='vgg19_bn', clf=None):
        super(VGG_BB, self).__init__()
        version = version.strip()
        if version == 'vgg11':
            vgg = models.vgg11(pretrained)
        elif version == 'vgg11_bn':
            vgg = models.vgg11_bn(pretrained)
        elif version == 'vgg13':
            vgg = models.vgg13(pretrained)
        elif version == 'vgg13_bn':
            vgg = models.vgg13_bn(pretrained)
        elif version == 'vgg16':
            vgg = models.vgg16(pretrained)
        elif version == 'vgg16_bn':
            vgg = models.vgg16_bn(pretrained)
        elif version == 'vgg19_bn':
            vgg = models.vgg19_bn(pretrained)
        elif version == 'vgg19':
            vgg = models.vgg19(pretrained)
        else:
            raise NotImplementedError('version {} is not supported as of now'.format(version))
        vgg.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        return_layers = {'features': 'out'}
        self.backbone = IntermediateLayerGetter(vgg, return_layers)
        self.out_channels = 512

    def forward(self, x):
        return self.backbone(x)


class ResNet_BB(nn.Module):
    version_set = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2',
                   'resnest50', 'resnest101', 'resnest200', 'resnest269']

    def __init__(self, in_channels, pretrained=False, version='resnet101', clf=None):
        super(ResNet_BB, self).__init__()
        version = version.strip()
        if version == 'resnet18':
            resnet = models.resnet18(pretrained)
        elif version == 'resnet34':
            resnet = models.resnet34(pretrained)
        elif version == 'resnet50':
            resnet = models.resnet50(pretrained)
        elif version == 'resnet101':
            resnet = models.resnet101(pretrained)
        elif version == 'resnet152':
            resnet = models.resnet152(pretrained)
        elif version == 'resnext50_32x4d':
            resnet = models.resnext50_32x4d(pretrained)
        elif version == 'resnext101_32x8d':
            resnet = models.resnext101_32x8d(pretrained)
        elif version == 'wide_resnet50_2':
            resnet = models.wide_resnet50_2(pretrained)
        elif version == 'wide_resnet101_2':
            resnet = models.wide_resnet101_2(pretrained)
        elif version == 'resnest50':
            resnet = resnest50(pretrained)
        elif version == 'resnest101':
            resnet = resnest101(pretrained)
        elif version == 'resnest200':
            resnet = resnest200(pretrained)
        elif version == 'resnest269':
            resnet = resnest269(pretrained)
        else:
            raise NotImplementedError('version {} is not supported as of now'.format(version))
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                            bias=False)
        if clf == 'deeplabv3Plus':
            return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        elif clf == 'PointRend':
            return_layers = {'layer4': 'out', 'layer2': 'fine_grained'}
        else:
            return_layers = {'layer4': 'out'}
        self.backbone = IntermediateLayerGetter(resnet, return_layers)
        self.out_channels = 2048
        self.low_channels = 256
        self.fine_grained_channels = 512

    def forward(self, x):
        return self.backbone(x)

class DenseNet_BB(nn.Module):
    version_set = ['densenet121', 'densenet169', 'densenet201', 'densenet161']
    def __init__(self, in_channels, pretrained=False, version='densenet161'):
        super(DenseNet_BB, self).__init__()
        if version == 'densenet161':
            denseNet = models.densenet161(pretrained=pretrained)
        elif version == 'densenet121':
            denseNet = models.densenet121(pretrained=pretrained)
        elif version == 'densenet169':
            denseNet = models.densenet169(pretrained=pretrained)
        elif version == 'densenet161':
            denseNet = models.densenet161(pretrained=pretrained)
        else:
            raise NotImplementedError('version {} is not supported as of now'.format(version))
        denseNet.features.conv0 = nn.Conv2d(in_channels, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        return_layers = {'denseblock4': 'out'}
        self.backbone = IntermediateLayerGetter(denseNet.features, return_layers)
        self.out_channels = 2208

    def forward(self, x):
        return self.backbone(x)

class TwoInput_NDDRLSC_BB(nn.Module):
    model_modes = ['NddrPure', 'NddrLSC', 'NddrLS', 'NddrCross3', 'NddrCross5', 'NddrCross35',
             'SingleTasks', 'MultiTasks', 'SIDCCross3', 'SIDCCross34', 'SIDCCross345', 'SIDCCross35', 'SIDCPure']
    modes = {'LayersLearningMixutres', 'EasyCat', 'WeightCat', 'LayersWeightCat', 'WeightMixtures'}
    def __init__(self, in_channels, in_aux_channels, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=4, drop_rate=0, num_classes=4, length_aux = 10 , mode='WeightMixtures', nddr_drop_rate = 0, memory_efficient=True, clf=None):
        super(TwoInput_NDDRLSC_BB, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.features_aux = nn.Sequential(OrderedDict([
            ('conv0_aux', nn.Conv2d(in_aux_channels, num_init_features, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
            ('norm0_aux', nn.BatchNorm2d(num_init_features)),
            ('relu0_aux', nn.ReLU(inplace=True)),
            ('pool0_aux', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.nddrf = NddrLayer(net0_channels=num_init_features, net1_channels=num_init_features,
                               drop_rate=nddr_drop_rate)

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            block_aux = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            if (i == 0):
                self.block0 = block
                self.block0_aux = block_aux
            elif (i == 1):
                self.block1 = block
                self.block1_aux = block_aux
            elif (i == 2):
                self.block2 = block
                self.block2_aux = block_aux
            elif (i == 3):
                self.block3 = block
                self.block3_aux = block_aux
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                trans_aux = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                if (i == 0):
                    self.transition0 = trans
                    self.transition0_aux = trans_aux
                    self.nddr0 = NddrLayer(net0_channels=num_features // 2, net1_channels=num_features // 2,
                                           drop_rate=nddr_drop_rate)
                    self.sluice0_conv1 = nn.Conv2d(num_features // 2, 2208, 1)
                    self.sluice0_aux_conv1 = nn.Conv2d(num_features // 2, 2208, 1)

                elif (i == 1):
                    self.transition1 = trans
                    self.transition1_aux = trans_aux
                    self.nddr1 = NddrLayer(net0_channels=num_features // 2, net1_channels=num_features // 2,
                                           drop_rate=nddr_drop_rate)
                    self.sluice1_conv1 = nn.Conv2d(num_features // 2, 2208, 1)
                    self.sluice1_aux_conv1 = nn.Conv2d(num_features // 2, 2208, 1)
                elif (i == 2):
                    self.transition2 = trans
                    self.transition2_aux = trans_aux
                    self.nddr2 = NddrLayer(net0_channels=num_features // 2, net1_channels=num_features // 2,
                                           drop_rate=nddr_drop_rate)
                    self.sluice2_conv1 = nn.Conv2d(num_features // 2, 2208, 1)
                    self.sluice2_aux_conv1 = nn.Conv2d(num_features // 2, 2208, 1)
                num_features = num_features // 2

        # Final batch norm

        self.final_bn = nn.BatchNorm2d(num_features)
        self.final_bn_aux = nn.BatchNorm2d(num_features)
        self.nddr3 = NddrLayer(net0_channels=num_features, net1_channels=num_features, drop_rate=nddr_drop_rate)
        # self.sluice3_conv1 = nn.Conv2d(num_features // 2, 2208, 1)
        # self.sluice3_aux_conv1 = nn.Conv2d(num_features // 2, 2208, 1)



        self.cross3 = nn.Linear(num_features * 2, num_features * 2, bias=False)
        self.cross4 = nn.Linear(2000, 2000, bias=False)
        self.cross5 = nn.Linear(6, 6, bias=False)

        self.betas_2layer = nn.Parameter(torch.tensor([0.7, 0.3]))
        self.betas_8layer = nn.Parameter(torch.tensor([0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2]))

        self.mode = mode
        self.out_channels = 2208
        self.clf = clf
        self.fine_grained_channels = 384

    def forward(self, x, y):
        result = OrderedDict()
        features = self.features(x)
        features_aux = self.features_aux(y)

        features, features_aux = self.nddrf(features, features_aux)

        block0 = self.block0(features)
        transition0 = self.transition0(block0)
        block0_aux = self.block0_aux(features_aux)
        transition0_aux = self.transition0_aux(block0_aux)



        transition0, transition0_aux = self.nddr0(transition0, transition0_aux)
        if self.clf == 'PointRend':
            result['fine_grained'] = torch.cat((transition0, transition0_aux), dim=1)
        block1 = self.block1(transition0)
        transition1 = self.transition1(block1)
        block1_aux = self.block1_aux(transition0_aux)
        transition1_aux = self.transition1_aux(block1_aux)

        # if self.clf == 'PointRend':
        #     result['fine_grained'] = torch.cat((transition0, transition0_aux), dim=1)

        transition1, transition1_aux = self.nddr1(transition1, transition1_aux)
        block2 = self.block2(transition1)
        transition2 = self.transition2(block2)
        block2_aux = self.block2_aux(transition1_aux)
        transition2_aux = self.transition2_aux(block2_aux)


        transition2, transition2_aux = self.nddr2(transition2, transition2_aux)
        block3 = self.block3(transition2)
        transition3 = self.final_bn(block3)
        block3_aux = self.block3_aux(transition2_aux)
        transition3_aux = self.final_bn_aux(block3_aux)

        transition3, transition3_aux = self.nddr3(transition3, transition3_aux)


        # 多层次加权求和(将各级level缩放到一起)
        if self.mode == 'LayersLearningMixutres':
            transition0 = self.sluice0_conv1(transition0)
            transition0 = F.adaptive_avg_pool2d(transition0, (7, 7))
            transition0_aux = self.sluice0_aux_conv1(transition0_aux)
            transition0_aux = F.adaptive_avg_pool2d(transition0_aux, (7, 7))

            transition1 = self.sluice1_conv1(transition1)
            transition1 = F.adaptive_avg_pool2d(transition1, (7, 7))
            transition1_aux = self.sluice1_aux_conv1(transition1_aux)
            transition1_aux = F.adaptive_avg_pool2d(transition1_aux, (7, 7))

            transition2 = self.sluice2_conv1(transition2)
            transition2 = F.adaptive_avg_pool2d(transition2, (7, 7))
            transition2_aux = self.sluice2_aux_conv1(transition2_aux)
            transition2_aux = F.adaptive_avg_pool2d(transition2_aux, (7, 7))

            out = torch.mul(transition0, self.betas_8layer[0]) + torch.mul(transition0_aux, self.betas_8layer[1]) + torch.mul(transition1, self.betas_8layer[2]) + torch.mul(transition1_aux, self.betas_8layer[3]) + torch.mul(transition2, self.betas_8layer[4]) + torch.mul(transition2_aux, self.betas_8layer[5]) + torch.mul(transition3, self.betas_8layer[6]) + torch.mul(transition3_aux, self.betas_8layer[7])
            result['out'] = out
            return result

        if self.mode == 'EasyCat':
            out = torch.cat((transition3, transition3_aux), dim=1)
            result['out'] = out
            return result
        if self.mode == 'WeightCat':
            out_0 = torch.mul(transition3, self.betas_2layer[0])
            out_1 = torch.mul(transition3_aux, self.betas_2layer[1])
            out = torch.cat((out_0, out_1), dim=1)
            result['out'] = out
            return result
        if self.mode == 'LayersWeightCat':
            transition0 = self.sluice0_conv1(transition0)
            transition0 = F.adaptive_avg_pool2d(transition0, (7, 7))
            transition0_aux = self.sluice0_aux_conv1(transition0_aux)
            transition0_aux = F.adaptive_avg_pool2d(transition0_aux, (7, 7))

            transition1 = self.sluice1_conv1(transition1)
            transition1 = F.adaptive_avg_pool2d(transition1, (7, 7))
            transition1_aux = self.sluice1_aux_conv1(transition1_aux)
            transition1_aux = F.adaptive_avg_pool2d(transition1_aux, (7, 7))

            transition2 = self.sluice2_conv1(transition2)
            transition2 = F.adaptive_avg_pool2d(transition2, (7, 7))
            transition2_aux = self.sluice2_aux_conv1(transition2_aux)
            transition2_aux = F.adaptive_avg_pool2d(transition2_aux, (7, 7))

            out_0 = torch.mul(transition0, self.betas_8layer[0])
            out_1 = torch.mul(transition0_aux, self.betas_8layer[1])
            out_2 = torch.mul(transition1, self.betas_8layer[2])
            out_3 = torch.mul(transition1_aux, self.betas_8layer[3])
            out_4 = torch.mul(transition2, self.betas_8layer[4])
            out_5 = torch.mul(transition2_aux, self.betas_8layer[5])
            out_6 = torch.mul(transition3, self.betas_8layer[6])
            out_7 = torch.mul(transition3_aux, self.betas_8layer[7])
            out = torch.cat((out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7), dim=1)
            result['out'] = out
            return result

        if self.mode == 'WeightMixtures':
            out = torch.mul(transition3, self.betas_2layer[0]) + torch.mul(transition3_aux, self.betas_2layer[1])

            result['out'] = out
            return result





class Dense_BB_ForUNet(nn.Module):
    def __init__(self, in_channels, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=4, drop_rate=0, length_aux = 10 , mode = None, nddr_drop_rate = 0, memory_efficient=True):
        super(Dense_BB_ForUNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )

            if (i == 0):
                self.block0 = block
            elif (i == 1):
                self.block1 = block
            elif (i == 2):
                self.block2 = block
            elif (i == 3):
                self.block3 = block
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                if (i == 0):
                    self.transition0 = trans

                elif (i == 1):
                    self.transition1 = trans

                elif (i == 2):
                    self.transition2 = trans

                num_features = num_features // 2

        # Final batch norm

        self.final_bn = nn.BatchNorm2d(num_features)




    # x5是最新的
    def forward(self, x):
        features = self.features(x)

        block0 = self.block0(features)
        transition0 = self.transition0(block0)

        block1 = self.block1(transition0)
        transition1 = self.transition1(block1)

        block2 = self.block2(transition1)
        transition2 = self.transition2(block2)

        block3 = self.block3(transition2)
        transition3 = self.final_bn(block3)


        return features, transition0, transition1, transition2, transition3


class NDDRLSC_BB_ForUNet(nn.Module):
    def __init__(self, in_channels, in_aux_channels, num_classes, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=4, drop_rate=0, length_aux = 10 , mode = None, nddr_drop_rate = 0, memory_efficient=True):
        super(NDDRLSC_BB_ForUNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.features_aux = nn.Sequential(OrderedDict([
            ('conv0_aux', nn.Conv2d(in_aux_channels, num_init_features, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
            ('norm0_aux', nn.BatchNorm2d(num_init_features)),
            ('relu0_aux', nn.ReLU(inplace=True)),
            ('pool0_aux', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.nddrf = NddrLayer(net0_channels=num_init_features, net1_channels=num_init_features,
                               drop_rate=nddr_drop_rate)

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            block_aux = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            if (i == 0):
                self.block0 = block
                self.block0_aux = block_aux
            elif (i == 1):
                self.block1 = block
                self.block1_aux = block_aux
            elif (i == 2):
                self.block2 = block
                self.block2_aux = block_aux
            elif (i == 3):
                self.block3 = block
                self.block3_aux = block_aux
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                trans_aux = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                if (i == 0):
                    self.transition0 = trans
                    self.transition0_aux = trans_aux
                    self.nddr0 = NddrLayer(net0_channels=num_features // 2, net1_channels=num_features // 2,
                                           drop_rate=nddr_drop_rate)
                    self.sluice0_reshape = nn.Linear(num_features // 2, 2208)
                    self.sluice0_aux_reshape = nn.Linear(num_features // 2, 2208)
                    # self.nddr0_conv1d = nn.Conv1d(num_features // 2 *2, num_features // 2)
                    # self.nddr0_conv1d_aux =
                    # self.nddr0_bn = nn.BatchNorm2d(num_features // 2)
                    # self.nddr0_bn_aux = nn.BatchNorm2d(num_features // 2)
                elif (i == 1):
                    self.transition1 = trans
                    self.transition1_aux = trans_aux
                    self.nddr1 = NddrLayer(net0_channels=num_features // 2, net1_channels=num_features // 2,
                                           drop_rate=nddr_drop_rate)
                    self.sluice1_reshape = nn.Linear(num_features // 2, 2208)
                    self.sluice1_aux_reshape = nn.Linear(num_features // 2, 2208)
                elif (i == 2):
                    self.transition2 = trans
                    self.transition2_aux = trans_aux
                    self.nddr2 = NddrLayer(net0_channels=num_features // 2, net1_channels=num_features // 2,
                                           drop_rate=nddr_drop_rate)
                    self.sluice2_reshape = nn.Linear(num_features // 2, 2208)
                    self.sluice2_aux_reshape = nn.Linear(num_features // 2, 2208)
                num_features = num_features // 2

        # Final batch norm

        self.final_bn = nn.BatchNorm2d(num_features)
        self.final_bn_aux = nn.BatchNorm2d(num_features)
        self.nddr3 = NddrLayer(net0_channels=num_features, net1_channels=num_features, drop_rate=nddr_drop_rate)
        self.sluice3_reshape = nn.Linear(num_features, 2208)
        self.sluice3_aux_reshape = nn.Linear(num_features, 2208)


        # Official init from torch repo.

        self.betas_5layer = nn.Parameter(torch.tensor([0.05, 0.1, 0.1, 0.25, 0.5]))
        self.betas_5layer_aux = nn.Parameter(torch.tensor([0.05, 0.1, 0.1, 0.25, 0.5]))
        self.betas_6layer = nn.Parameter(torch.tensor([0.05, 0.1, 0.1, 0.25, 0.5, 0.3]))
        self.betas_6layer_aux = nn.Parameter(torch.tensor([0.05, 0.1, 0.1, 0.25, 0.5, 0.3]))

    def forward(self, x, y):
        features = self.features(x)
        features_aux = self.features_aux(y)
        features, features_aux = self.nddrf(features, features_aux)

        block0 = self.block0(features)
        transition0 = self.transition0(block0)
        block0_aux = self.block0_aux(features_aux)
        transition0_aux = self.transition0_aux(block0_aux)
        transition0, transition0_aux = self.nddr0(transition0, transition0_aux)

        block1 = self.block1(transition0)
        transition1 = self.transition1(block1)
        block1_aux = self.block1_aux(transition0_aux)
        transition1_aux = self.transition1_aux(block1_aux)
        transition1, transition1_aux = self.nddr1(transition1, transition1_aux)

        block2 = self.block2(transition1)
        transition2 = self.transition2(block2)
        block2_aux = self.block2_aux(transition1_aux)
        transition2_aux = self.transition2_aux(block2_aux)
        transition2, transition2_aux = self.nddr2(transition2, transition2_aux)

        block3 = self.block3(transition2)
        transition3 = self.final_bn(block3)
        block3_aux = self.block3_aux(transition2_aux)
        transition3_aux = self.final_bn_aux(block3_aux)
        transition3, transition3_aux = self.nddr3(transition3, transition3_aux)

        return features, transition0, transition1, transition2, transition3

class BestPractice2I(nn.Module):
    model_modes = ['NddrPure', 'NddrLSC', 'NddrLS', 'NddrCross3', 'NddrCross5', 'NddrCross35',
             'SingleTasks', 'MultiTasks', 'SIDCCross3', 'SIDCCross34', 'SIDCCross345', 'SIDCCross35', 'SIDCPure']
    modes = {'LayersLearningMixutres', 'EasyCat', 'WeightCat', 'LayersWeightCat', 'WeightMixtures'}
    def __init__(self, in_channels, in_aux_channels, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=4, drop_rate=0, num_classes=4, length_aux = 10 , mode='WeightMixtures', nddr_drop_rate = 0, memory_efficient=True, clf='PointRendBiased'):
        super(BestPractice2I, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.features_aux = nn.Sequential(OrderedDict([
            ('conv0_aux', nn.Conv2d(in_aux_channels, num_init_features, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
            ('norm0_aux', nn.BatchNorm2d(num_init_features)),
            ('relu0_aux', nn.ReLU(inplace=True)),
            # ('pool0_aux', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.nddrf = NddrLayer(net0_channels=num_init_features, net1_channels=num_init_features,
                               drop_rate=nddr_drop_rate)

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            block_aux = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            if (i == 0):
                self.block0 = block
                self.block0_aux = block_aux
            elif (i == 1):
                self.block1 = block
                self.block1_aux = block_aux
            elif (i == 2):
                self.block2 = block
                self.block2_aux = block_aux
            elif (i == 3):
                self.block3 = block
                self.block3_aux = block_aux
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2, asc=True)
                trans_aux = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2, asc=True)
                if (i == 0):
                    self.transition0 = trans
                    self.transition0_aux = trans_aux
                    self.nddr0 = NddrLayer(net0_channels=num_features // 2, net1_channels=num_features // 2,
                                           drop_rate=nddr_drop_rate)
                    self.sluice0_conv1 = nn.Conv2d(num_features // 2, 2208, 1)
                    self.sluice0_aux_conv1 = nn.Conv2d(num_features // 2, 2208, 1)

                elif (i == 1):
                    self.transition1 = trans
                    self.transition1_aux = trans_aux
                    self.nddr1 = NddrLayer(net0_channels=num_features // 2, net1_channels=num_features // 2,
                                           drop_rate=nddr_drop_rate)
                    self.sluice1_conv1 = nn.Conv2d(num_features // 2, 2208, 1)
                    self.sluice1_aux_conv1 = nn.Conv2d(num_features // 2, 2208, 1)
                elif (i == 2):
                    self.transition2 = trans
                    self.transition2_aux = trans_aux
                    self.nddr2 = NddrLayer(net0_channels=num_features // 2, net1_channels=num_features // 2,
                                           drop_rate=nddr_drop_rate)
                    self.sluice2_conv1 = nn.Conv2d(num_features // 2, 2208, 1)
                    self.sluice2_aux_conv1 = nn.Conv2d(num_features // 2, 2208, 1)
                num_features = num_features // 2

        # Final batch norm

        self.final_bn = nn.BatchNorm2d(num_features)
        self.final_bn_aux = nn.BatchNorm2d(num_features)
        self.nddr3 = NddrLayer(net0_channels=num_features, net1_channels=num_features, drop_rate=nddr_drop_rate)



        self.cross3 = nn.Linear(num_features * 2, num_features * 2, bias=False)
        self.cross4 = nn.Linear(2000, 2000, bias=False)
        self.cross5 = nn.Linear(6, 6, bias=False)

        self.betas_2layer = nn.Parameter(torch.tensor([0.7, 0.3]))
        self.betas_8layer = nn.Parameter(torch.tensor([0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2]))

        self.mode = mode
        self.out_channels = 2208
        self.clf = clf
        if clf == 'PointRendBoth':
            self.fine_grained_channels = 384
        elif clf == 'PointRendBiased':
            self.fine_grained_channels = 768

    def forward(self, x, y):
        result = OrderedDict()
        features = self.features(x)
        features_aux = self.features_aux(y)

        features, features_aux = self.nddrf(features, features_aux)

        block0 = self.block0(features)
        transition0 = self.transition0(block0)
        block0_aux = self.block0_aux(features_aux)
        transition0_aux = self.transition0_aux(block0_aux)


        transition0, transition0_aux = self.nddr0(transition0, transition0_aux)

        block1 = self.block1(transition0)
        transition1 = self.transition1(block1)
        block1_aux = self.block1_aux(transition0_aux)
        transition1_aux = self.transition1_aux(block1_aux)

        if self.clf == 'PointRendBoth':
            result['fine_grained'] = torch.cat((transition0, transition0_aux), dim=1)
        elif self.clf == 'PointRendBiased':
            result['fine_grained'] = block1

        transition1, transition1_aux = self.nddr1(transition1, transition1_aux)
        block2 = self.block2(transition1)
        transition2 = self.transition2(block2)
        block2_aux = self.block2_aux(transition1_aux)
        transition2_aux = self.transition2_aux(block2_aux)


        transition2, transition2_aux = self.nddr2(transition2, transition2_aux)
        block3 = self.block3(transition2)
        transition3 = self.final_bn(block3)
        block3_aux = self.block3_aux(transition2_aux)
        transition3_aux = self.final_bn_aux(block3_aux)

        transition3, transition3_aux = self.nddr3(transition3, transition3_aux)


        # 多层次加权求和(将各级level缩放到一起)
        if self.mode == 'LayersLearningMixutres':
            transition0 = self.sluice0_conv1(transition0)
            transition0 = F.adaptive_avg_pool2d(transition0, (7, 7))
            transition0_aux = self.sluice0_aux_conv1(transition0_aux)
            transition0_aux = F.adaptive_avg_pool2d(transition0_aux, (7, 7))

            transition1 = self.sluice1_conv1(transition1)
            transition1 = F.adaptive_avg_pool2d(transition1, (7, 7))
            transition1_aux = self.sluice1_aux_conv1(transition1_aux)
            transition1_aux = F.adaptive_avg_pool2d(transition1_aux, (7, 7))

            transition2 = self.sluice2_conv1(transition2)
            transition2 = F.adaptive_avg_pool2d(transition2, (7, 7))
            transition2_aux = self.sluice2_aux_conv1(transition2_aux)
            transition2_aux = F.adaptive_avg_pool2d(transition2_aux, (7, 7))

            out = torch.mul(transition0, self.betas_8layer[0]) + torch.mul(transition0_aux, self.betas_8layer[1]) + torch.mul(transition1, self.betas_8layer[2]) + torch.mul(transition1_aux, self.betas_8layer[3]) + torch.mul(transition2, self.betas_8layer[4]) + torch.mul(transition2_aux, self.betas_8layer[5]) + torch.mul(transition3, self.betas_8layer[6]) + torch.mul(transition3_aux, self.betas_8layer[7])
            result['out'] = out
            return result

        if self.mode == 'EasyCat':
            out = torch.cat((transition3, transition3_aux), dim=1)
            result['out'] = out
            return result
        if self.mode == 'WeightCat':
            out_0 = torch.mul(transition3, self.betas_2layer[0])
            out_1 = torch.mul(transition3_aux, self.betas_2layer[1])
            out = torch.cat((out_0, out_1), dim=1)
            result['out'] = out
            return result
        if self.mode == 'LayersWeightCat':
            transition0 = self.sluice0_conv1(transition0)
            transition0 = F.adaptive_avg_pool2d(transition0, (7, 7))
            transition0_aux = self.sluice0_aux_conv1(transition0_aux)
            transition0_aux = F.adaptive_avg_pool2d(transition0_aux, (7, 7))

            transition1 = self.sluice1_conv1(transition1)
            transition1 = F.adaptive_avg_pool2d(transition1, (7, 7))
            transition1_aux = self.sluice1_aux_conv1(transition1_aux)
            transition1_aux = F.adaptive_avg_pool2d(transition1_aux, (7, 7))

            transition2 = self.sluice2_conv1(transition2)
            transition2 = F.adaptive_avg_pool2d(transition2, (7, 7))
            transition2_aux = self.sluice2_aux_conv1(transition2_aux)
            transition2_aux = F.adaptive_avg_pool2d(transition2_aux, (7, 7))

            out_0 = torch.mul(transition0, self.betas_8layer[0])
            out_1 = torch.mul(transition0_aux, self.betas_8layer[1])
            out_2 = torch.mul(transition1, self.betas_8layer[2])
            out_3 = torch.mul(transition1_aux, self.betas_8layer[3])
            out_4 = torch.mul(transition2, self.betas_8layer[4])
            out_5 = torch.mul(transition2_aux, self.betas_8layer[5])
            out_6 = torch.mul(transition3, self.betas_8layer[6])
            out_7 = torch.mul(transition3_aux, self.betas_8layer[7])
            out = torch.cat((out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7), dim=1)
            result['out'] = out
            return result

        if self.mode == 'WeightMixtures':
            out = torch.mul(transition3, self.betas_2layer[0]) + torch.mul(transition3_aux, self.betas_2layer[1])

            result['out'] = out
            return result


class BestPractice1I(nn.Module):
    def __init__(self, in_channels, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=4, drop_rate=0, length_aux = 10 , mode = None, nddr_drop_rate = 0, memory_efficient=True, clf='PointRend'):
        super(BestPractice1I, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )

            if (i == 0):
                self.block0 = block
            elif (i == 1):
                self.block1 = block
            elif (i == 2):
                self.block2 = block
            elif (i == 3):
                self.block3 = block
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2, asc=True)
                if (i == 0):
                    self.transition0 = trans

                elif (i == 1):
                    self.transition1 = trans

                elif (i == 2):
                    self.transition2 = trans

                num_features = num_features // 2

        # Final batch norm

        self.final_bn = nn.BatchNorm2d(num_features)
        self.out_channels = 2208
        if clf == 'PointRend':
            self.fine_grained_channels = 768




    # x5是最新的
    def forward(self, x):
        result = OrderedDict()
        features = self.features(x)

        block0 = self.block0(features)
        transition0 = self.transition0(block0)

        block1 = self.block1(transition0)
        transition1 = self.transition1(block1)

        block2 = self.block2(transition1)
        transition2 = self.transition2(block2)

        block3 = self.block3(transition2)
        transition3 = self.final_bn(block3)
        result['fine_grained'] = block1
        result['out'] = transition3
        return result