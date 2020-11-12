# 提供以下特性：
# 1. 组合BackBone与Classifier
# 2. 提供统一的参数初始化接口，但可以允许某些指定的module自行初始化
# 提供的模型包括非U-Net系的single model、input fusion、和NDDR系的layer fusion
# TODO：新增功能， DecisionFusion
import collections
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
import math
import torch.hub as hub
from typing import Dict
from torchvision.models import resnet, densenet, vgg, inception
from Model.Backbone import DenseNet_BB, TwoInput_NDDRLSC_BB
from Model.Classifier import DeepLabV3Head, FCNHead
from Model._utils import IntermediateLayerGetter, get_criterion, extractDict, test2IBackward


class SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None):
        super(SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight)
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
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result

class TwoInputSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None):
        super(TwoInputSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if not name.startswith('nddr'):
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
    def forward(self, x, y):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x, y)

        result = OrderedDict()
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class DesicionFusion(nn.Module):
    def __init__(self, modelA, modelB, num_class, combining_points='late', fusion_function='conv'):
        super(DesicionFusion, self).__init__()
        if fusion_function not in ['conv', 'max', 'sum']:
            raise ValueError("Fusion Function not defined.")
        self.modelA = modelA
        self.modelB = modelB

        if fusion_function == 'conv':
            self.fusion = nn.Conv2d(num_class * 2, num_class, 1)
        self.fusion_function = fusion_function
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if not name.startswith('nddr'):
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

    def forward(self, x, y):
        x = self.modelA(x)
        y = self.modelB(y)
        x = extractDict(x)
        y = extractDict(y)
        if self.fusion_function == 'conv':
            out = torch.cat((x, y), dim=1)
            out = self.fusion(out)
        elif self.fusion_function == 'max':
            out = torch.max(x, y)
        elif self.fusion_function == 'sum':
            out = torch.add(x, y)
        return out









def testModel(model):
    input = torch.rand((4, 3, 224, 224))
    out = model(input)
    print(out['out'].shape)
    print(out)



def testBackward(model):
    label = torch.rand((4, 1, 224, 224))
    input = torch.rand((4, 3, 224, 224))
    testEpoch = 3
    for epoch in range(testEpoch):

        output = model(input)
        output = nn.Sigmoid()(output)
        print(output.shape)
        criterion = get_criterion('BCELoss')
        optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-4, eps=1e-8)
        loss = criterion(output, label)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    backbone = DenseNet_BB(3)
    classifier = DeepLabV3Head(backbone.out_channels, 1)
    modelA = SimpleSegmentationModel(backbone, classifier)
    # testModel(modelA)
    backboneB = DenseNet_BB(1)
    classifierB = DeepLabV3Head(backboneB.out_channels, 1)
    modelB = SimpleSegmentationModel(backboneB, classifierB)
    model = DesicionFusion(modelA, modelB, 1)
    test2IBackward(model)


# 原始torchvision给出 backbone-classfier模型：backbone 将input(c, h, w) 变为 （2048， h/8, w/8)， 二分类为(1, h/8, w/8), 最后再插值回去。
# 这样使得每个像素要对应之前的64个像素，显然是表达困难的... 需要重新实现！
# input output shapetorch.Size([8, 3, 224, 224])
# backbone output shapetorch.Size([8, 2048, 28, 28])
# classifier output shapetorch.Size([8, 1, 28, 28])
# result output shapetorch.Size([8, 1, 224, 224])
# input output shapetorch.Size([8, 3, 448, 448])
# backbone output shapetorch.Size([8, 2048, 56, 56])
# classifier output shapetorch.Size([8, 1, 56, 56])
# result output shapetorch.Size([8, 1, 448, 448])