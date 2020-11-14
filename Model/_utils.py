import collections

import torch.nn as nn
from typing import Dict
from collections import OrderedDict
import torch
from Loss import MTLLoss
import numpy as np
import random


def adjust_learning_rate(optimizer, epoch, base_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = base_lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_criterion(criterion):
    criterion = criterion.strip()
    if criterion == "TheCrossEntropy":
        return MTLLoss.TheCrossEntropy()
    elif criterion == "BCELoss":
        return nn.BCELoss()
    elif criterion == "DiceLoss":
        return MTLLoss.DiceLoss()
    elif criterion == "mIOULoss":
        return MTLLoss.mIoULoss()
    elif criterion == "IouLoss":
        return MTLLoss.IouLoss()
    elif criterion == "GDL":
        return MTLLoss.GWDL(weighting_mode='GDL')
    elif criterion == "GWDL":
        return MTLLoss.GWDL(weighting_mode='default')
    elif criterion == "LovaszSoftmax":
        return MTLLoss.LovaszSoftmax(per_image=True)
    elif criterion == "LovaszHinge":
        return MTLLoss.LovaszHinge(per_image=True)
    # if criterion == "Hausdorff":
    #     return MTLLoss.GeomLoss(loss="hausdorff")
    if criterion == "HDLoss":
        return MTLLoss.HDLoss()

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

def extractDict(x):
    if isinstance(x, collections.OrderedDict):
        return x['out']
    return x

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def testModel(model):
    input = torch.rand((4, 5, 224, 224))
    output = model(input)
    print(output)
    print(output.shape)

def testBackward(model):
    label = torch.rand((4, 1, 224, 224))
    input = torch.rand((4, 5, 224, 224))
    testEpoch = 3
    for epoch in range(testEpoch):
        output = model(input)
        output = nn.Sigmoid()(output)
        print(output.shape)
        criterion = get_criterion('BCELoss')
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        loss = criterion(output, label)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test2IModel(model):
    input0 = torch.rand((4, 3, 224, 224))
    input1 = torch.rand((4, 3, 224, 224))
    out = model(input0, input1)
    # print(out['out'].shape)
    print(out)
    print(out.shape)


def test2IBackward(model):
    label = torch.zeros((4, 1, 224, 224))
    input0 = torch.rand((4, 3, 224, 224))
    input1 = torch.rand((4, 3, 224, 224))
    testEpoch = 10
    for epoch in range(testEpoch):

        output = model(input0, input1)
        output = nn.Sigmoid()(output)
        print(output.shape)
        criterion = get_criterion('BCELoss')
        optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-4, eps=1e-8)
        loss = criterion(output, label)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()