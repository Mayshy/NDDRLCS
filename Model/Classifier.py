#!/usr/bin/env python
#-*- coding:utf-8 _*-
# TODO:实现SegNet
# 实现DeepLabV1
from collections import OrderedDict

import torch.nn as nn
from torch.nn import functional as F
import torch

from Model.DeepLabV3Plus import ASPPPlus, convert_conv2_to_separable_conv
from Model.UNet import Up, OutConv


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)



# 输入输出size不变
class DeepLabV3Head(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV3Head, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        # 48维是论文中实验过的一个参数，具体看论文
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPPPlus(in_channels, aspp_dilate)

        # use double Conv2d get better performance
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        convert_conv2_to_separable_conv(self)
        self._init_weight()

    # 高级特征来自aspp的输出
    # 提供细节信息的低级特征似乎可以有更优美的输入
    def forward(self, feature):
        low_level_feature = self.project(feature['fine_grained'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        # output_feature 通常output stride为16
        # low_level_feature 通常output stride为4
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
        # 最后仍然需要上采样4倍

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        # padding = dilation to keep the size invariant
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # base condition: only 1x1 conv to transfer channels
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)




class UNet_Classifier(nn.Module):
    densenet_in_channels = (3264, 768, 384, 192)
    def __init__(self, in_channels=(3264, 768, 384, 192), num_classes=1, origin_size=336):
        super(UNet_Classifier, self).__init__()
        factor = 2
        self.origin_size = origin_size
        self.up1 = Up(in_channels[0], in_channels[1] // factor)
        self.up2 = Up(in_channels[1], in_channels[2] // factor)
        self.up3 = Up(in_channels[2], in_channels[3] // factor)
        self.up4 = Up(in_channels[3], 64)
        self.outc = OutConv(64, num_classes)
    def forward(self, x1, x2, x3, x4, x5):
        x1 = F.interpolate(x1, size=self.origin_size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=self.origin_size//2, mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=self.origin_size//4, mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=self.origin_size//8, mode='bilinear', align_corners=True)
        x5 = F.interpolate(x5, size=self.origin_size//16, mode='bilinear', align_corners=True)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

# TODO: UNet Familiy, including
# class U


if __name__ == '__main__':
    classifier = DeepLabHeadV3Plus(3,6, 1)
    low_features = torch.rand((2, 6, 224, 224))
    out_features = torch.rand((2, 3, 56, 56))
    features = OrderedDict()
    features['low_level'] = low_features
    features['out'] = out_features
    out = classifier(features)
    print(out.shape)