#!/usr/bin/env python
#-*- coding:utf-8 _*-
# TODO:实现SegNet
# 实现DeepLabV1
import torch.nn as nn
from torch.nn import functional as F
import torch


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




class DeepLabV3Head(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV3Head, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
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




