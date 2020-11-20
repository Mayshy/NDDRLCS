#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
from torch import optim

# 可选的LR策略：
# params = [{"params": net.backbone.backbone.parameters(),   "lr": C.train.lr},
#               {"params": net.head.parameters(),                "lr": C.train.lr},
#               {"params": net.backbone.classifier.parameters(), "lr": C.train.lr * 10}]

def get_optimizer(optimizer, model, LEARNING_RATE, MOMENTUM=0, WEIGHT_DECAY=0):
    if (optimizer == 'Adam'):
        # return optim.Adam([{'params':model.parameters()},{'params':multi_loss.parameters()}],lr=LEARNING_RATE)
        return optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    if (optimizer == 'RMSprop'):
        return optim.RMSprop(params=model.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-08, weight_decay=0,
                             momentum=MOMENTUM, centered=False)
    if (optimizer == 'SGD'):
        return optim.SGD(params=model.parameters(), lr=LEARNING_RATE, momentum = MOMENTUM)
    if (optimizer == 'AdamW'):
        return optim.AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    if (optimizer == 'AmsgradW'):
        return optim.AdamW(params=model.parameters(), lr=LEARNING_RATE,weight_decay = WEIGHT_DECAY, amsgrad=True)