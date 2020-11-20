#!/usr/bin/env python  
#-*- coding:utf-8 _*-  

import numpy as np
import torch

# mixup 数据增强器，帮助提升小数据集下训练与测试的稳定性
def mixup_data(x, y0, y1, y2, device, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y0_a, y0_b = y0, y0[index]
    y1_a, y1_b = y1, y1[index]
    y2_a, y2_b = y2, y2[index]
    return mixed_x, y0_a, y0_b, y1_a, y1_b, y2_a, y2_b, lam

def mixup_data2(x0, x1, y0, y1, y2, device, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x0.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x0 = lam * x0 + (1 - lam) * x0[index, :]
    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    y0_a, y0_b = y0, y0[index]
    y1_a, y1_b = y1, y1[index]
    y2_a, y2_b = y2, y2[index]
    return mixed_x0, mixed_x1, y0_a, y0_b, y1_a, y1_b, y2_a, y2_b, lam

def mixup_criterion_type(the_criterion, pred, y_a, y_b, lam):
    return lam * the_criterion(pred, y_a) + (1 - lam) * the_criterion(pred, y_b)