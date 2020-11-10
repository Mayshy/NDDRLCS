#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
import torch
import torch.nn.functional as F

from torch import nn

from Model._utils import testModel


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





class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size)
        )


    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size = 3, padding=1, is_double_conv = True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size = kernel_size, padding = padding, )

        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)

            self.conv = DoubleConv(in_channels, out_channels, kernel_size = kernel_size, padding = padding)

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
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class DeepMadSeg(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DeepMadSeg, self).__init__()
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
        self.outc = OutConv(n_classes * 4, n_classes)
        self.after_upsample6 = DoubleConv(256, n_classes)
        self.after_upsample7 = DoubleConv(128, n_classes)
        self.after_upsample8 = DoubleConv(64, n_classes)
        self.after_upsample9 = DoubleConv(64, n_classes)
        self.layers_weights = nn.Parameter(torch.tensor([0.1, 0.2,  0.3, 0.4]))

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        x6 = F.interpolate(x6, size=size, mode='bilinear', align_corners=True)
        x6 = self.after_upsample6(x6)
        x6 = torch.mul(x6, self.layers_weights[0])
        x7 = F.interpolate(x7, size=size, mode='bilinear', align_corners=True)
        x7 = self.after_upsample7(x7)
        x7 = torch.mul(x7, self.layers_weights[1])
        x8 = F.interpolate(x8, size=size, mode='bilinear', align_corners=True)
        x8 = self.after_upsample8(x8)
        x8 = torch.mul(x8, self.layers_weights[2])
        x9 = F.interpolate(x9, size=size, mode='bilinear', align_corners=True)
        x9 = self.after_upsample9(x9)
        x9 = torch.mul(x9, self.layers_weights[3])

        x = torch.cat([x6, x7, x8, x9], dim=1)
        x = self.outc(x)

        return x

if __name__ == '__main__':
    testModel(DeepMadSeg(5, 1))