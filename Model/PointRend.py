#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:shy
@license: Apache Licence 
@file: PointRend.py 
@time: 2020/11/04
@contact: justbeshy@outlook.com
@site:  
@software: PyCharm 

@description:

# Programs must be written for people to read.
# Good code is its own best documentation.
# Focus on your question, not your function.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import Bottleneck, resnet50, resnet101, ResNet
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def point_sample(input, point_coords, **kwargs):
    """
    From Detectron2, point_features.py#19
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


@torch.no_grad()
def sampling_points(mask, N, k=3, beta=0.75, training=True):
    """
    Follows 3.1. Point Selection for Inference and Training
    In Train:, `The sampling strategy selects N points on a feature map to train on.`
    In Inference, `then selects the N most uncertain points`
    Args:
        mask(Tensor): [B, C, H, W]
        N(int): `During training we sample as many points as there are on a stride 16 feature map of the input`
        k(int): Over generation multiplier
        beta(float): ratio of importance points
        training(bool): flag
    Return:
        selected_point(Tensor) : flattened indexing points [B, num_points, 2]
    """
    assert mask.dim() == 4, "Dim must be N(Batch)CHW"
    device = mask.device
    B, _, H, W = mask.shape
    mask, _ = mask.sort(1, descending=True)

    if not training:
        H_step, W_step = 1 / H, 1 / W
        N = min(H * W, N)
        uncertainty_map = -1 * (mask[:, 0] - mask[:, 1])
        _, idx = uncertainty_map.view(B, -1).topk(N, dim=1)

        points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        points[:, :, 0] = W_step / 2.0 + (idx  % W).to(torch.float) * W_step
        points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
        return idx, points

    # Official Comment : point_features.py#92
    # It is crucial to calculate uncertanty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to worse results. To illustrate the difference: a sampled point between two coarse predictions
    # with -1 and 1 logits has 0 logit prediction and therefore 0 uncertainty value, however, if one
    # calculates uncertainties for the coarse predictions first (-1 and -1) and sampe it for the
    # center point, they will get -1 unceratinty.

    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = point_sample(mask, over_generation, align_corners=False)

    uncertainty_map = -1 * (over_generation_map[:, 0] - over_generation_map[:, 1])
    _, idx = uncertainty_map.topk(int(beta * N), -1)

    shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)

    idx += shift[:, None]

    importance = over_generation.view(-1, 2)[idx.view(-1), :].view(B, int(beta * N), 2)
    coverage = torch.rand(B, N - int(beta * N), 2, device=device)
    return torch.cat([importance, coverage], 1).to(device)

class ResNetXX3(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__(block, layers, num_classes, zero_init_residual,
                         groups, width_per_group, replace_stride_with_dilation,
                         norm_layer)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')


def resnet53(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetXX3(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet103(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetXX3(Bottleneck, [3, 4, 23, 3], **kwargs)


class PointHead(nn.Module):
    def __init__(self, in_c=514, num_classes=2, k=3, beta=0.75):
        super().__init__()
        self.mlp = nn.Conv1d(in_c, num_classes, 1)
        self.k = k
        self.beta = beta

    def forward(self, x, res2, out):
        """
        1. Fine-grained features are interpolated from res2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input()
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        """
        if not self.training:
            return self.inference(x, res2, out)

        points = sampling_points(out, x.shape[-1] // 16, self.k, self.beta)

        coarse = point_sample(out, points, align_corners=False)
        fine = point_sample(res2, points, align_corners=False)

        feature_representation = torch.cat([coarse, fine], dim=1)

        rend = self.mlp(feature_representation)

        return {"rend": rend, "points": points}

    @torch.no_grad()
    def inference(self, x, res2, out):
        """
        During inference, subdivision uses N=8096
        (i.e., the number of points in the stride 16 map of a 1024×2048 image)
        """
        num_points = 8096
        while out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)

            points_idx, points = sampling_points(out, num_points, training=self.training)

            coarse = point_sample(out, points, align_corners=False)
            fine = point_sample(res2, points, align_corners=False)

            feature_representation = torch.cat([coarse, fine], dim=1)

            rend = self.mlp(feature_representation)

            B, C, H, W = out.shape
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)

            out = (out.reshape(B, C, -1)
                      .scatter_(2, points_idx, rend)
                      .view(B, C, H, W))

        return {"fine": out}


class SmallDeepLab(_SimpleSegmentationModel):
    def forward(self, input_):
        result = self.backbone(input_)
        result["coarse"] = self.classifier(result["out"])
        return result


def deeplabv3(pretrained=False, resnet="res103", head_in_ch=2048, num_classes=2):
    resnet = {
        "res53":  resnet53,
        "res103": resnet103,
        "res50":  resnet50,
        "res101": resnet101
    }[resnet]

    net = SmallDeepLab(
        backbone=IntermediateLayerGetter(
            resnet(pretrained=True, replace_stride_with_dilation=[False, True, True]),
            return_layers={'layer2': 'res2', 'layer4': 'out'}
        ),
        classifier=DeepLabHead(head_in_ch, num_classes)
    )
    return net

class PointRend(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        result = self.backbone(x)
        result.update(self.head(x, result["res2"], result["coarse"]))
        return result

if __name__ == "__main__":
    x = torch.randn(4, 3, 244, 244)
    # pred = F.interpolate(result["coarse"], x.shape[-2:], mode="bilinear", align_corners=True)
    # seg_loss = F.cross_entropy(pred, gt, ignore_index=255)
    net = PointRend(deeplabv3(False, num_classes=2), PointHead(num_classes=2))
    # net.eval()
    out = net(x)
    for k, v in out.items():
        print(k, v.shape)