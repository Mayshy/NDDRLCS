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


# class _SimpleSegmentationModel(nn.Module):
#     __constants__ = ['aux_classifier']
#
#     def __init__(self, backbone, classifier, aux_classifier=None):
#         super(_SimpleSegmentationModel, self).__init__()
#         self.backbone = backbone
#         self.classifier = classifier
#         self.aux_classifier = aux_classifier
#
#     def forward(self, x):
#         input_shape = x.shape[-2:]
#         # contract: features is a dict of tensors
#         features = self.backbone(x)
#
#         result = OrderedDict()
#         x = features["out"]
#         x = self.classifier(x)
#         x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
#         result["out"] = x
#
#         if self.aux_classifier is not None:
#             x = features["aux"]
#             x = self.aux_classifier(x)
#             x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
#             result["aux"] = x
#
#         return result
#
#
# __all__ = ["FCN"]
#
#
# class FCN(_SimpleSegmentationModel):
#     """
#     Implements a Fully-Convolutional Network for semantic segmentation.
#     Arguments:
#         backbone (nn.Module): the network used to compute the features for the model.
#             The backbone should return an OrderedDict[Tensor], with the key being
#             "out" for the last feature map used, and "aux" if an auxiliary classifier
#             is used.
#         classifier (nn.Module): module that takes the "out" element returned from
#             the backbone and returns a dense prediction.
#         aux_classifier (nn.Module, optional): auxiliary classifier used during training
#     """
#     pass
#
#
# class FCNHead(nn.Sequential):
#     def __init__(self, in_channels, channels):
#         inter_channels = in_channels // 4
#         layers = [
#             nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Conv2d(inter_channels, channels, 1)
#         ]
#
#         super(FCNHead, self).__init__(*layers)

class FCNResNet101(nn.Module):
    def __init__(self, in_channels, n_classes, pretrained=False):
        super(FCNResNet101, self).__init__()
        self.base_model = torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained, progress=False, num_classes=n_classes,
                                                     aux_loss=None)
        self.base_model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.base_model.backbone

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.base_model(x)['out']

class ILFUNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ILFUNet, self).__init__()
        self.base_model = UNet.UNet(n_channels=in_channels, n_classes=n_classes)
        # self.base_model.backbone

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.base_model(x)

class ILFModel(nn.Module):
    def __init__(self, model, in_channels, n_classes):
        super(ILFModel, self).__init__()
        self.base_model = model(n_channels=in_channels, n_classes=n_classes)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.base_model(x)


if __name__ == '__main__':
    model = FCNResNet101(in_channels=5, n_classes=1)
    input1 = torch.rand((16,3,224,224))
    input2 = torch.rand((16, 2, 224, 224))
    print(model(input1, input2))
