#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import torch.nn as nn

bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],loss6.data[0]))

	return loss0, loss

class U2NetLoss(nn.Module):
    def __init__(self):
        super(U2NetLoss, self).__init__()

    def forward(self, pred, target):
        if isinstance(pred, tuple):
            d0, d1, d2, d3, d4, d5, d6 = pred
        else:
            raise TypeError("Type of pred must be tuple.")
        loss0, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, target)
        return loss