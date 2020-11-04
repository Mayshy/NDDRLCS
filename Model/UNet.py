#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:shy
@license: Apache Licence 
@file: DilatedUNet.py
@time: 2020/10/13
@contact: justbeshy@outlook.com
@site:
@software: PyCharm

@description:

# Programs must be written for people to read.
# Good code is its own best documentation.
# Focus on your question, not your function.
"""

# Code for KiU-Net
# Author: Jeya Maria Jose
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models



class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)

        self.decoder1 = nn.Conv2d(1024, 512, 3, stride=1, padding=2)  # b, 16, 5, 5
        self.decoder2 = nn.Conv2d(512, 256, 3, stride=1, padding=2)  # b, 8, 15, 1
        self.decoder3 = nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(64, 2, 3, stride=1, padding=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))

        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))

        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        out = self.soft(out)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(AutoEncoder, self).__init__()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1

        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.down2(x2)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()

        self.encoder1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1, padding=1)  # b, 16, 5, 5
        self.decoder2 = nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(32, 1, 3, stride=1, padding=1)

        # self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out
        out = F.relu(F.max_pool2d(self.encoder4(out), 2, 2))
        t4 = out
        out = F.relu(F.max_pool2d(self.encoder5(out), 2, 2))

        # t2 = out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        # print(out.shape,t4.shape)
        out = torch.add(out, t4)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        out = torch.add(out, t3)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        # print(out.shape)

        return out


class kinetwithsk(nn.Module):
    def __init__(self):
        super(kinetwithsk, self).__init__()

        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.encoder4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1, padding=2)  # b, 16, 5, 5
        self.decoder2 = nn.Conv2d(256, 128, 3, stride=1, padding=2)  # b, 8, 15, 1

        self.decoder3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(32, 2, 3, stride=1, padding=1)

        self.decoderf1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoderf2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoderf3 = nn.Conv2d(32, 2, 3, stride=1, padding=1)

        self.encoderf1 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoderf2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoderf3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu(F.interpolate(self.encoder1(x), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        t1 = out
        out = F.relu(F.interpolate(self.encoder2(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        t2 = out
        out = F.relu(F.interpolate(self.encoder3(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        # print(out.shape)

        out = F.relu(F.max_pool2d(self.decoder3(out), 2, 2))
        out = torch.add(out, t2)
        out = F.relu(F.max_pool2d(self.decoder4(out), 2, 2))
        out = torch.add(out, t1)
        out = F.relu(F.max_pool2d(self.decoder5(out), 2, 2))

        out = self.soft(out)
        return out


class kinet(nn.Module):

    def __init__(self):
        super(kinet, self).__init__()

        self.encoder1 = nn.Conv2d(1, 32, 15, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv2d(32, 64, 8, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv2d(64, 128, 5, stride=1, padding=1)

        self.encoder4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1, padding=2)  # b, 16, 5, 5
        self.decoder2 = nn.Conv2d(256, 128, 3, stride=1, padding=2)  # b, 8, 15, 1

        self.decoder3 = nn.Conv2d(128, 64, 5, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv2d(64, 32, 8, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(32, 2, 15, stride=1, padding=1)

        self.decoderf1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoderf2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoderf3 = nn.Conv2d(32, 2, 3, stride=1, padding=1)

        self.encoderf1 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoderf2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoderf3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu(F.interpolate(self.encoder1(x), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        out = F.relu(F.interpolate(self.encoder2(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        out = F.relu(F.interpolate(self.encoder3(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))

        out = F.relu(F.max_pool2d(self.decoder3(out), 2, 2))
        out = F.relu(F.max_pool2d(self.decoder4(out), 2, 2))
        out = F.relu(F.max_pool2d(self.decoder5(out), 2, 2))

        out = self.soft(out)

        return out

class KiNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(KiNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.up1 = Upsample(n_channels, 16, kernel_size=3, padding=1, scale_factor=2)  # b, 16, 10, 10
        self.up2 = Upsample(16, 32, kernel_size=3, padding=1, scale_factor=2)  # b, 8, 3, 3
        self.up3 = Upsample(32, 64, kernel_size=3, padding=1, scale_factor=2)
        self.down1 = Down(64, 32, kernel_size=3, isDoubleConv= True)
        self.down2 = Down(32, 16, kernel_size=3, isDoubleConv= True)
        self.down3 = Down(16, n_classes, kernel_size=3, isDoubleConv= True)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x


# https://github.com/usuyama/pytorch-unet
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )
class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

class Encoder(nn.Module):
    def __init__(self, in_channels,  n_block=3):
        super().__init__()
        self.downs = nn.ModuleList([Down(in_channels * (2 ** i), in_channels * (2 ** (i + 1))) for i in range(n_block)])
        self.n_blocks = n_block
    def forward(self, x):
        skip = []
        for i, down in enumerate(self.downs):
            skip.append(x)
            x = down(x)
        return x, skip

class BottleNeck(nn.Module):
    def __init__(self, in_channels, n_layers=6):
        super().__init__()
        self.n_layers = n_layers
        # self.dilated_layers = nn.ModuleList([nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1, dilation= i + 1) if i == 0 else nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1) for i in range(n_layers)])
        self.dilated_layers = nn.ModuleList([nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=i + 1, dilation=i + 1)  for i in range(n_layers)])
        self.batch_normalizations = nn.ModuleList([nn.BatchNorm2d(in_channels) for i in range(n_layers)])
        self.down = nn.Conv2d(in_channels, in_channels //2, kernel_size=3, padding=1)
    def forward(self, x_count):
        for i in range(self.n_layers):
            x_count = self.dilated_layers[i](x_count)
            x_count = F.relu(x_count,inplace=True)
            x_count = self.batch_normalizations[i](x_count)
            if i == 0:
                x_sum = x_count
            else:
                x_sum = torch.add(x_sum, x_count)

        return self.down(x_sum)

class Decoder(nn.Module):
    def __init__(self, in_channels, n_block=3):
        super().__init__()
        self.ups = nn.ModuleList([Up(in_channels // (2 ** i), in_channels // (2 ** (i + 2))) for i in range(n_block)])
    def forward(self, x, strip):
        for i, up in enumerate(self.ups):
            x = up(x, strip[- i - 1])
        return x

class DilatedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DilatedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.encoder = Encoder(32, n_block=4)
        self.bottleneck = BottleNeck(32 * (2 **self.encoder.n_blocks), n_layers=6)
        self.decoder = Decoder(32 * (2 ** (self.encoder.n_blocks)), n_block=4)
        self.outc = OutConv(16, n_classes)
    def forward(self, x):
        x = self.inc(x)
        x, strip = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, strip)
        x = self.outc(x)
        return x


class kiunet(nn.Module):

    def __init__(self):
        super(kiunet, self).__init__()

        self.encoder1 = nn.Conv2d(1, 16, 3, stride=1,
                                  padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB
        self.en1_bn = nn.BatchNorm2d(16)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.en2_bn = nn.BatchNorm2d(32)
        self.encoder3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.en3_bn = nn.BatchNorm2d(64)

        self.decoder1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.de1_bn = nn.BatchNorm2d(32)
        self.decoder2 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.de2_bn = nn.BatchNorm2d(16)
        self.decoder3 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.de3_bn = nn.BatchNorm2d(8)

        self.decoderf1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.def1_bn = nn.BatchNorm2d(32)
        self.decoderf2 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.def2_bn = nn.BatchNorm2d(16)
        self.decoderf3 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.def3_bn = nn.BatchNorm2d(8)

        self.encoderf1 = nn.Conv2d(1, 16, 3, stride=1,
                                   padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB
        self.enf1_bn = nn.BatchNorm2d(16)
        self.encoderf2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.enf2_bn = nn.BatchNorm2d(32)
        self.encoderf3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.enf3_bn = nn.BatchNorm2d(64)

        self.intere1_1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm2d(16)
        self.intere2_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.inte2_1bn = nn.BatchNorm2d(32)
        self.intere3_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.inte3_1bn = nn.BatchNorm2d(64)

        self.intere1_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm2d(16)
        self.intere2_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.inte2_2bn = nn.BatchNorm2d(32)
        self.intere3_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.inte3_2bn = nn.BatchNorm2d(64)

        self.interd1_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.intd1_1bn = nn.BatchNorm2d(32)
        self.interd2_1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.intd2_1bn = nn.BatchNorm2d(16)
        self.interd3_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.intd3_1bn = nn.BatchNorm2d(64)

        self.interd1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.intd1_2bn = nn.BatchNorm2d(32)
        self.interd2_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.intd2_2bn = nn.BatchNorm2d(16)
        self.interd3_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.intd3_2bn = nn.BatchNorm2d(64)

        self.final = nn.Conv2d(8, 2, 1, stride=1, padding=0)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu(self.en1_bn(F.max_pool2d(self.encoder1(x), 2, 2)))  # U-Net branch
        out1 = F.relu(
            self.enf1_bn(F.interpolate(self.encoderf1(x), scale_factor=(2, 2), mode='bilinear', align_corners=True)))  # Ki-Net branch
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))), scale_factor=(0.25, 0.25),
                                           mode='bilinear', align_corners=True))  # CRFB
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(tmp))), scale_factor=(4, 4),
                                             mode='bilinear', align_corners=True))  # CRFB

        u1 = out  # skip conn
        o1 = out1  # skip conn

        out = F.relu(self.en2_bn(F.max_pool2d(self.encoder2(out), 2, 2)))
        out1 = F.relu(self.enf2_bn(F.interpolate(self.encoderf2(out1), scale_factor=(2, 2), mode='bilinear')))
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))), scale_factor=(0.0625, 0.0625),
                                           mode='bilinear', align_corners=True))
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(tmp))), scale_factor=(16, 16),
                                             mode='bilinear', align_corners=True))

        u2 = out
        o2 = out1

        out = F.relu(self.en3_bn(F.max_pool2d(self.encoder3(out), 2, 2)))
        out1 = F.relu(self.enf3_bn(F.interpolate(self.encoderf3(out1), scale_factor=(2, 2), mode='bilinear', align_corners=True)))
        tmp = out
        out = torch.add(out,
                        F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))), scale_factor=(0.015625, 0.015625),
                                      mode='bilinear', align_corners=True))
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(tmp))), scale_factor=(64, 64),
                                             mode='bilinear', align_corners=True))

        ### End of encoder block

        ### Start Decoder

        out = F.relu(self.de1_bn(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear', align_corners=True)))  # U-NET
        out1 = F.relu(self.def1_bn(F.max_pool2d(self.decoderf1(out1), 2, 2)))  # Ki-NET
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))), scale_factor=(0.0625, 0.0625),
                                           mode='bilinear', align_corners=True))
        out1 = torch.add(out1, F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(tmp))), scale_factor=(16, 16),
                                             mode='bilinear', align_corners=True))

        out = torch.add(out, u2)  # skip conn
        out1 = torch.add(out1, o2)  # skip conn

        out = F.relu(self.de2_bn(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear', align_corners=True)))
        out1 = F.relu(self.def2_bn(F.max_pool2d(self.decoderf2(out1), 2, 2)))
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))), scale_factor=(0.25, 0.25),
                                           mode='bilinear', align_corners=True))
        out1 = torch.add(out1, F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(tmp))), scale_factor=(4, 4),
                                             mode='bilinear', align_corners=True))

        out = torch.add(out, u1)
        out1 = torch.add(out1, o1)

        out = F.relu(self.de3_bn(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear', align_corners=True)))
        out1 = F.relu(self.def3_bn(F.max_pool2d(self.decoderf3(out1), 2, 2)))

        out = torch.add(out, out1)  # fusion of both branches

        out = F.relu(self.final(out))  # 1*1 conv

        out = self.soft(out)

        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size = 3, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ConvBatchNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size = 3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size = 3, isDoubleConv = True):
        super().__init__()
        if isDoubleConv:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels, kernel_size)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBatchNormReLU(in_channels, out_channels, kernel_size)
            )

    def forward(self, x):
        return self.maxpool_conv(x)

class Upsample(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size = 3, scale_factor = 2, padding=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding = padding)
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.post_process = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return self.post_process(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size = 3, padding=1, is_double_conv = True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if is_double_conv:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size = kernel_size, padding = padding, )
            else:
                self.conv = ConvBatchNormReLU(in_channels, out_channels, kernel_size=kernel_size,
                                       padding=padding)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            if is_double_conv:
                self.conv = DoubleConv(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
            else:
                self.conv = ConvBatchNormReLU(in_channels, out_channels, kernel_size=kernel_size,
                                      padding=padding)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

