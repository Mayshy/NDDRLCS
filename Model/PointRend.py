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
import torchvision
from torch.nn import functional as F

class PointRendR50FPN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(PointRendR50FPN, self).__init__()

    def forward(self, x):
        return x