#!/usr/bin/env python  
#-*- coding:utf-8 _*-  


# Code for KiU-Net
# Author: Jeya Maria Jose
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torch import optim
from torch.nn import init

from Model._utils import testBackward

class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3,
                              padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

class DUCFCN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DUCFCN, self).__init__()

        self.num_classes = num_classes

        resnet = models.resnet50(pretrained=True)
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        else:
            self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.duc1 = DUC(2048, 2048*2)
        self.duc2 = DUC(1024, 1024*2)
        self.duc3 = DUC(512, 512*2)
        self.duc4 = DUC(128, 128*2)
        self.duc5 = DUC(64, 64*2)

        self.out1 = self._classifier(1024)
        self.out2 = self._classifier(512)
        self.out3 = self._classifier(128)
        self.out4 = self._classifier(64)
        self.out5 = self._classifier(32)

        self.transformer = nn.Conv2d(320, 128, kernel_size=1)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes//2, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes//2, self.num_classes, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        dfm1 = fm3 + self.duc1(fm4)
        out16 = self.out1(dfm1)

        dfm2 = fm2 + self.duc2(dfm1)
        out8 = self.out2(dfm2)

        dfm3 = fm1 + self.duc3(dfm2)

        dfm3_t = self.transformer(torch.cat((dfm3, pool_x), 1))
        out4 = self.out3(dfm3_t)

        dfm4 = conv_x + self.duc4(dfm3_t)
        out2 = self.out4(dfm4)

        dfm5 = self.duc5(dfm4)
        out = self.out5(dfm5)




        return out

class SegNet(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)


    def forward(self, x):

        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)


        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x4p, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(model_path).state_dict() # load the weigths
        # for name in th:
            # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)

class autoencoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(autoencoder, self).__init__()

        self.encoder1 = nn.Conv2d(n_channels, 64, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.encoder4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        # self.encoder5 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)

        # self.decoder1 = nn.Conv2d(1024, 512, 3, stride=1, padding=2)  # b, 16, 5, 5
        # self.decoder2 = nn.Conv2d(512, 256, 3, stride=1, padding=2)  # b, 8, 15, 1
        self.decoder3 = nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(64, n_classes, 3, stride=1, padding=1)

        # self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))

        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))

        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        # out = self.soft(out)
        return out




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
        x = self.outc(x)
        return x

class unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(unet, self).__init__()

        self.encoder1 = nn.Conv2d(n_channels, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1, padding=1)  # b, 16, 5, 5
        self.decoder2 = nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(32, n_classes, 3, stride=1, padding=1)


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
    def __init__(self, n_channels, n_classes):
        super(kinetwithsk, self).__init__()

        self.encoder1 = nn.Conv2d(n_channels, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.encoder4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1, padding=2)  # b, 16, 5, 5
        self.decoder2 = nn.Conv2d(256, 128, 3, stride=1, padding=2)  # b, 8, 15, 1

        self.decoder3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(32, n_classes, 3, stride=1, padding=1)

        # self.decoderf1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.decoderf2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        # self.decoderf3 = nn.Conv2d(32, 2, 3, stride=1, padding=1)
        #
        # self.encoderf1 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        # self.encoderf2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        # self.encoderf3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)


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

        return out


class kinet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(kinet, self).__init__()

        self.encoder1 = nn.Conv2d(n_channels, 32, 15, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv2d(32, 64, 8, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv2d(64, 128, 5, stride=1, padding=1)

        self.encoder4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1, padding=2)  # b, 16, 5, 5
        self.decoder2 = nn.Conv2d(256, 128, 3, stride=1, padding=2)  # b, 8, 15, 1

        self.decoder3 = nn.Conv2d(128, 64, 5, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv2d(64, 32, 8, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(32, n_classes, 15, stride=1, padding=1)

        # self.decoderf1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.decoderf2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        # self.decoderf3 = nn.Conv2d(32, n_classes, 3, stride=1, padding=1)
        #
        # self.encoderf1 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        # self.encoderf2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        # self.encoderf3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)


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
    def __init__(self, in_channels, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                            bias=False)

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

        self.conv_original_size0 = convrelu(in_channels, 64, 3, 1)
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

    def __init__(self, n_channels, n_classes):
        super(kiunet, self).__init__()

        self.encoder1 = nn.Conv2d(n_channels, 16, 3, stride=1,
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

        self.encoderf1 = nn.Conv2d(n_channels, 16, 3, stride=1,
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
        # self.interd3_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.intd3_1bn = nn.BatchNorm2d(64)

        self.interd1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.intd1_2bn = nn.BatchNorm2d(32)
        self.interd2_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.intd2_2bn = nn.BatchNorm2d(16)
        # self.interd3_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.intd3_2bn = nn.BatchNorm2d(64)

        self.final = nn.Conv2d(8, n_classes, 1, stride=1, padding=0)

        # self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu(self.en1_bn(F.max_pool2d(self.encoder1(x), 2, 2)))  # U-Net branch
        out1 = F.relu(
            self.enf1_bn(F.interpolate(self.encoderf1(x), scale_factor=(2, 2), mode='bilinear', align_corners=True, recompute_scale_factor=True)))  # Ki-Net branch
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))), scale_factor=(0.25, 0.25),
                                           mode='bilinear', align_corners=True, recompute_scale_factor=True))  # CRFB
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(tmp))), scale_factor=(4, 4),
                                             mode='bilinear', align_corners=True, recompute_scale_factor=True))  # CRFB

        u1 = out  # skip conn
        o1 = out1  # skip conn

        out = F.relu(self.en2_bn(F.max_pool2d(self.encoder2(out), 2, 2)))
        out1 = F.relu(self.enf2_bn(F.interpolate(self.encoderf2(out1), scale_factor=(2, 2), mode='bilinear', align_corners=True, recompute_scale_factor=True)))
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))), scale_factor=(0.0625, 0.0625),
                                           mode='bilinear', align_corners=True, recompute_scale_factor=True))
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(tmp))), scale_factor=(16, 16),
                                             mode='bilinear', align_corners=True, recompute_scale_factor=True))

        u2 = out
        o2 = out1

        out = F.relu(self.en3_bn(F.max_pool2d(self.encoder3(out), 2, 2)))
        out1 = F.relu(self.enf3_bn(F.interpolate(self.encoderf3(out1), scale_factor=(2, 2), mode='bilinear', align_corners=True, recompute_scale_factor=True)))
        tmp = out
        out = torch.add(out,
                        F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))), scale_factor=(0.015625, 0.015625),
                                      mode='bilinear', align_corners=True, recompute_scale_factor=True))
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(tmp))), scale_factor=(64, 64),
                                             mode='bilinear', align_corners=True, recompute_scale_factor=True))

        ### End of encoder block

        ### Start Decoder

        out = F.relu(self.de1_bn(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear', align_corners=True, recompute_scale_factor=True)))  # U-NET
        out1 = F.relu(self.def1_bn(F.max_pool2d(self.decoderf1(out1), 2, 2)))  # Ki-NET
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))), scale_factor=(0.0625, 0.0625),
                                           mode='bilinear', align_corners=True, recompute_scale_factor=True))
        out1 = torch.add(out1, F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(tmp))), scale_factor=(16, 16),
                                             mode='bilinear', align_corners=True, recompute_scale_factor=True))

        out = torch.add(out, u2)  # skip conn
        out1 = torch.add(out1, o2)  # skip conn

        out = F.relu(self.de2_bn(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear', align_corners=True, recompute_scale_factor=True)))
        out1 = F.relu(self.def2_bn(F.max_pool2d(self.decoderf2(out1), 2, 2)))
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))), scale_factor=(0.25, 0.25),
                                           mode='bilinear', align_corners=True, recompute_scale_factor=True))
        out1 = torch.add(out1, F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(tmp))), scale_factor=(4, 4),
                                             mode='bilinear', align_corners=True, recompute_scale_factor=True))

        out = torch.add(out, u1)
        out1 = torch.add(out1, o1)

        out = F.relu(self.de3_bn(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear', align_corners=True, recompute_scale_factor=True)))
        out1 = F.relu(self.def3_bn(F.max_pool2d(self.decoderf3(out1), 2, 2)))

        out = torch.add(out, out1)  # fusion of both branches

        out = F.relu(self.final(out))  # 1*1 conv


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


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)

class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

# https://github.com/ZJUGiveLab/UNet-Version/blob/8ecd06740d18cf508e5c8b2cc2bf38a026aa0e9b/models/init_weights.py#L53
class UNet_2Plus(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_2Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale

        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)


        # upsampling
        self.up_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return final
        else:
            return final_4


class ZJUNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(ZJUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        #
        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        # # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        #
        self.outconv1 = nn.Conv2d(filters[0], 1, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        d1 = self.outconv1(up1)  # 256

        return d1

class UNet_3Plus(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return d1

# def testModel(model):
#     input = torch.rand((4, 5, 224, 224))
#     output = model(input)
#     print(output)
#     print(output.shape)
#
# def testBackward(model):
#     label = torch.rand((4, 1, 224, 224))
#     input = torch.rand((4, 5, 224, 224))
#     testEpoch = 3
#     for epoch in range(testEpoch):
#         output = model(input)
#         output = nn.Sigmoid()(output)
#         print(output.shape)
#         criterion = get_criterion('BCELoss')
#         optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
#         loss = criterion(output, label)
#         print(loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()



if __name__ == '__main__':
    # testModel(ResNetUNet(5, 1))
    # testModel(autoencoder(5, 1))
    # testBackward(kinetwithsk(5, 1))
    # testBackward(autoencoder(5, 1))
    testBackward(UNet_3Plus(5, 1))

