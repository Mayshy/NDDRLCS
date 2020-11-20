# 提供以下特性：
# 1. 组合BackBone与Classifier
# 2. 提供统一的参数初始化接口，但可以允许某些指定的module自行初始化
# 提供的模型包括非U-Net系的single model、input fusion、和NDDR系的layer fusion
import collections
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
import math
from Model.Backbone import DenseNet_BB, TwoInput_NDDRLSC_BB, ResNet_BB
from Model.Classifier import DeepLabV3Head, FCNHead
from Model.PointRend import sampling_points, point_sample
from Model._utils import IntermediateLayerGetter, extractDict, test2IBackward, testModel, test2IModel


class PointRendUpsample(nn.Module):
    def __init__(self, in_c=514, num_classes=2, k=3, beta=0.75):
        super(PointRendUpsample, self).__init__()

        self.mlp = nn.Conv1d(in_c, num_classes, 1)
        self.k = k
        self.beta = beta

    def forward(self, origin_shape, fine_grained_feature, coarse_feature):
        """
        1. Fine-grained features are interpolated from res2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input()
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        """
        if not self.training:
            return self.inference(origin_shape, fine_grained_feature, coarse_feature)

        points = sampling_points(coarse_feature, origin_shape // 16, self.k, self.beta)

        coarse = point_sample(coarse_feature, points, align_corners=False)
        fine = point_sample(fine_grained_feature, points, align_corners=False)

        feature_representation = torch.cat([coarse, fine], dim=1)

        rend = self.mlp(feature_representation)

        return {"rend": rend, "points": points}

    @torch.no_grad()
    def inference(self, origin_shape, fine_grained_feature, coarse_feature):
        """
        During inference, subdivision uses N=8096
        (i.e., the number of points in the stride 16 map of a 1024×2048 image)
        """
        num_points = 8096

        while coarse_feature.shape[-1] != origin_shape:
            coarse_feature = F.interpolate(coarse_feature, scale_factor=2, mode="bilinear", align_corners=True)

            points_idx, points = sampling_points(coarse_feature, num_points, training=self.training)

            coarse = point_sample(coarse_feature, points, align_corners=False)
            fine = point_sample(fine_grained_feature, points, align_corners=False)

            feature_representation = torch.cat([coarse, fine], dim=1)

            rend = self.mlp(feature_representation)

            B, C, H, W = coarse_feature.shape
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
            coarse_feature = (coarse_feature.reshape(B, C, -1)
                              .scatter_(2, points_idx, rend)
                              .view(B, C, H, W))

        return {"out": coarse_feature}

class SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, if_extract_dict=True, if_point_rend_upsample=False):
        super(SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.if_extract_dict = if_extract_dict # only for deeplabV3+ do not extract dict
        self.if_point_rend_upsample = if_point_rend_upsample
        if if_point_rend_upsample:
            # limit num_classes = 2
            self.rend_upsample = PointRendUpsample(self.backbone.fine_grained_channels + 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                stdv = 1./ math.sqrt(m.weight.size(1))
                nn.init.uniform_(m.weight,-stdv,stdv)
                # nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_shape = x.shape[-1]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        if self.if_extract_dict:
            fine_grained = features['fine_grained']
            features = extractDict(features)
        if isinstance(features, tuple):
            x = self.classifier(*features)
        else:
            x = self.classifier(features)

        if self.if_point_rend_upsample:
            result = self.rend_upsample(input_shape, fine_grained, x)
            if not self.training:
                return result['out']
            else:
                # when training, calculating seg_loss from result['coarse'] after interpolating and point_loss from result['point'] and rend
                result['out'] = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
                return result
        else:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            return x



class TwoInputSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, if_extract_dict=True, if_point_rend_upsample=False):
        super(TwoInputSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.if_extract_dict = if_extract_dict  # only for deeplabV3+ do not extract dict
        self.if_point_rend_upsample = if_point_rend_upsample
        if if_point_rend_upsample:
            # limit num_classes = 2
            self.rend_upsample = PointRendUpsample(self.backbone.fine_grained_channels + 2)

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
        input_shape = x.shape[-1]
        # contract: features is a dict of tensors
        features = self.backbone(x, y)

        if self.if_extract_dict:
            fine_grained = features['fine_grained']
            features = extractDict(features)
        if isinstance(features, tuple):
            x = self.classifier(*features)
        else:
            x = self.classifier(features)

        if self.if_point_rend_upsample:
            result = self.rend_upsample(input_shape, fine_grained, x)
            if not self.training:
                return result['out']
            else:
                # when training, calculating seg_loss from result['coarse'] after interpolating and point_loss from result['point'] and rend
                result['out'] = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
                return result
        else:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            return x




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






if __name__ == '__main__':
    backbone = TwoInput_NDDRLSC_BB(3, 3, clf='PointRend')
    classifier = DeepLabV3Head(backbone.out_channels, 2)
    modelA = TwoInputSegmentationModel(backbone, classifier, if_point_rend_upsample=True)
    test2IModel(modelA, eval=True)
    # test2IBackward(modelA, eval=True)


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