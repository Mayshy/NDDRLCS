#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
import torch
import torchvision

from Loss.LossList import LovaszHinge, LovaszSoftmax, SymmetricLovasz
from Model._utils import extractDict, setup_seed

setup_seed(20)

def testLoss(Loss):
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=False, num_classes=2,
                                                         aux_loss=None)
    input = torch.rand((8, 3, 224, 224))
    label = torch.rand((8, 1, 224, 224))
    label[label >= 0.5] = 1
    label[label < 0.5] = 0
    # label = label.long()
    testEpoch = 100
    for epoch in range(testEpoch):
        output = model(input)
        output = extractDict(output)
        # output = torch.nn.Sigmoid()(output)
        print(output.shape)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        loss = Loss(output, label)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    # DEVICE = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

    # testLoss(model, HDLoss())  待有GPU时测试
    # testLoss(model, BoundaryLoss())
    # testLoss(model, nn.BCELoss())
    # loss = LovaszSoftmax(per_image=True)
    loss = LovaszSoftmax(classes='all')
    # SymmetricLovasz bug
    # loss = GWDL(weighting_mode='GDL')
    # testLoss(loss)
    testLoss(loss)