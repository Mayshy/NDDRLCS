import collections

import torch.nn as nn
from typing import Dict
from collections import OrderedDict
import torch
from Loss import LossList
import numpy as np
import random
import logging

from Loss.LossList import get_criterion


def adjust_learning_rate(optimizer, epoch, base_lr, full_epoch=200):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = base_lr * (0.2 ** (epoch // (full_epoch/4)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



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

def extractDict(x, force=False):
    if isinstance(x, dict):
        if force or (len(x)==1):
            return x['out']
    return x

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def log_mean(epoch, array, name, isLog = False):
    array = np.array(array)
    mean = np.mean(array)
    if isLog:
        logging.info("Epoch {0} {2} MEAN {1}".format(epoch, mean, name))
        logging.info("Epoch {0} {2} STD {1}".format(epoch, np.std(array), name))
    return mean

def dict_sum(res, addend):
    if not res:
        res = addend
    else:
        for k in res:
            res[k] += addend[k]
    return res

def testModel(model, eval=False):
    if eval:
        model.eval()
    input = torch.rand((4, 3, 256, 256))
    output = model(input)
    if isinstance(output, dict) and len(output) > 1:
        for k, v in output.items():
            print(k, v.shape)

    else:
        output = extractDict(output)
        if isinstance(output, tuple):
            output = output[0]
        print(output.shape)

def testBackward(model):
    label = torch.rand((4, 2, 224, 224))
    input = torch.rand((4, 3, 224, 224))
    testEpoch = 3
    for epoch in range(testEpoch):
        output = model(input)
        output = extractDict(output, True)
        output = nn.Sigmoid()(output)
        criterion = get_criterion('BCELoss')
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        loss = criterion(output, label)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test2IModel(model, eval=False):
    if eval:
        model.eval()
    input0 = torch.rand((4, 3, 256, 256))
    input1 = torch.rand((4, 3, 256, 256))
    output = model(input0, input1)
    if isinstance(output, dict) and len(output) > 1:
        for k, v in output.items():
            print(k, v.shape)

    else:
        output = extractDict(output)
        if isinstance(output, tuple):
            output = output[0]
        print(output.shape)


def test2IBackward(model, eval=False):
    if eval:
        model.eval()
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