#!/usr/bin/env python  
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from Model import UNet
import math



class ILFFCNResNet101(nn.Module):
    def __init__(self, in_channels, n_classes, pretrained=False):
        super(ILFFCNResNet101, self).__init__()
        self.base_model = torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained, progress=False, num_classes=n_classes,
                                                     aux_loss=None)
        self.base_model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.base_model.backbone
        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    # nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    stdv = 1./ math.sqrt(m.weight.size(1))
                    nn.init.uniform_(m.weight,-stdv,stdv)
                    # nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.base_model(x)['out']

class ILFDLV3ResNet50(nn.Module):
    def __init__(self, in_channels, n_classes, pretrained=False):
        super(ILFDLV3ResNet50, self).__init__()
        self.base_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=False, num_classes=n_classes,
                                                     aux_loss=None)
        self.base_model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.base_model.backbone

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    # nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    stdv = 1./ math.sqrt(m.weight.size(1))
                    nn.init.uniform_(m.weight,-stdv,stdv)
                    # nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.base_model(x)['out']

class ILFDLV3ResNet101(nn.Module):
    def __init__(self, in_channels, n_classes, pretrained=False):
        super(ILFDLV3ResNet101, self).__init__()
        self.base_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=False, num_classes=n_classes,
                                                     aux_loss=None)
        self.base_model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.base_model.backbone

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    # nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    stdv = 1./ math.sqrt(m.weight.size(1))
                    nn.init.uniform_(m.weight,-stdv,stdv)
                    # nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.base_model(x)['out']

class ILFUNet(nn.Module):
    def __init__(self, in_channels, n_classes, pretrained=False):
        super(ILFUNet, self).__init__()
        self.base_model = UNet.UNet(n_channels=in_channels, n_classes=n_classes)
        # self.base_model.backbone

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    # nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    stdv = 1./ math.sqrt(m.weight.size(1))
                    nn.init.uniform_(m.weight,-stdv,stdv)
                    # nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.base_model(x)

class ILFModel(nn.Module):
    def __init__(self, model, in_channels, n_classes, pretrained=False):
        super(ILFModel, self).__init__()
        self.base_model = model(n_channels=in_channels, n_classes=n_classes)

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    # nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    stdv = 1./ math.sqrt(m.weight.size(1))
                    nn.init.uniform_(m.weight,-stdv,stdv)
                    # nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.base_model(x)


if __name__ == '__main__':
    model = ILFFCNResNet101(in_channels=5, n_classes=1)
    input1 = torch.rand((16,3,224,224))
    input2 = torch.rand((16, 2, 224, 224))
    print(model(input1, input2))
