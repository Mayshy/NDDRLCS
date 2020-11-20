#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
import torch.nn as nn
import torchvision
import math


class SourceModel(nn.Module):
    def __init__(self, in_channels, n_class, model_name='FCN_ResNet50', pretrained=False):
        super(SourceModel, self).__init__()
        if model_name == 'FCN_ResNet50':
            self.base = torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained, progress=False, num_classes=n_class)
        elif model_name == 'FCN_ResNet101':
            self.base = torchvision.models.segmentation.fcn_resnet101(pretrained=pretrained, progress=False,
                                                                      num_classes=n_class)
        elif model_name == 'DLV3__ResNet50':
            self.base = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=False, num_classes=n_class)
        elif model_name == 'DLV3__ResNet101':
            self.base = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=False, num_classes=n_class)
        else:
            raise NotImplementedError("Model {} is unsupported now.".format(model_name))
        if in_channels != 3:
            self.base.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                stdv = 1./ math.sqrt(m.weight.size(1))
                nn.init.uniform_(m.weight,-stdv,stdv)
                # nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.base(x)['out']