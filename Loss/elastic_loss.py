import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.fft

torch.backends.cudnn.deterministic = True
# 待GPU可用时测试

def odd_flip(H):
    '''
    generate frequency map
    when height or width of image is odd number,
    creat a array concol [0,1,...,int(H/2)+1,int(H/2),...,0]
    len(concol) = H
    '''
    m = int(H / 2)
    col = np.arange(0, m + 1)
    flipcol = col[m - 1::-1]
    concol = np.concatenate((col, flipcol), 0)
    return concol


def even_flip(H):
    '''
    generate frequency map
    when height or width of image is even number,
    creat a array concol [0,1,...,int(H/2),int(H/2),...,0]
    len(concol) = H
    '''
    m = int(H / 2)
    col = np.arange(0, m)
    flipcol = col[m::-1]
    concol = np.concatenate((col, flipcol), 0)
    return concol


def dist(target):
    '''
    sqrt(m^2 + n^2) in eq(8)
    '''

    _, _, H, W = target.shape

    if H % 2 == 1:
        concol = odd_flip(H)
    else:
        concol = even_flip(H)

    if W % 2 == 1:
        conrow = odd_flip(W)
    else:
        conrow = even_flip(W)

    m_col = concol[:, np.newaxis]
    m_row = conrow[np.newaxis, :]
    dist = np.sqrt(m_col * m_col + m_row * m_row)  # sqrt(m^2+n^2)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dist_ = torch.from_numpy(dist).float().cuda()
    else:
        dist_ = torch.from_numpy(dist).float()
    return dist_


class EnergyLoss(nn.Module):
    def __init__(self, cuda, alpha, sigma):
        super(EnergyLoss, self).__init__()
        self.energylossfunc = EnergylossFunc.apply
        self.alpha = alpha
        self.cuda = cuda
        self.sigma = sigma

    def forward(self, feat, label):
        return self.energylossfunc(self.cuda, feat, label, self.alpha, self.sigma)


class EnergylossFunc(Function):
    '''
    target: ground truth
    feat: Z -0.5. Z：prob of your target class(here is vessel) with shape[B,H,W].
    Z from softmax output of unet with shape [B,C,H,W]. C: number of classes
    alpha: default 0.35
    sigma: default 0.25
    '''

    @staticmethod
    def forward(ctx, cuda, feat_levelset, target, alpha, sigma, Gaussian=False):
        hardtanh = nn.Hardtanh(min_val=0, max_val=1, inplace=False)
        target = target.float()
        index_ = dist(target)
        dim_ = target.shape[1]
        target = torch.squeeze(target, 1)
        I1 = target + alpha * hardtanh(feat_levelset / sigma)  # G_t + alpha*H(phi) in eq(5)
        dmn = torch.fft.rfft(I1, 2, normalized=True, onesided=False)
        dmn_r = dmn[:, :, :, 0]  # dmn's real part
        dmn_i = dmn[:, :, :, 1]  # dmm's imagine part
        dmn2 = dmn_r * dmn_r + dmn_i * dmn_i  # dmn^2

        ctx.save_for_backward(feat_levelset, target, dmn, index_)

        print(index_.shape)
        print(dmn2.shape)
        print(feat_levelset.shape)

        F_energy = torch.sum(index_ * dmn2) / feat_levelset.shape[0] / feat_levelset.shape[1] / feat_levelset.shape[
            2]  # eq(8)

        return F_energy

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, dmn, index_ = ctx.saved_tensors
        index_ = torch.unsqueeze(index_, 0)
        index_ = torch.unsqueeze(index_, 3)
        F_diff = -0.5 * index_ * dmn  # eq(9)
        diff = torch.irfft(F_diff, 2, normalized=True, onesided=False) / feature.shape[0]  # eq
        return None, Variable(-grad_output * diff), None, None, None

def testLoss(model, Loss):
    input = torch.rand((4, 3, 224, 224))
    label = torch.rand((4, 1, 224, 224))
    testEpoch = 3
    for epoch in range(testEpoch):
        output = model(input)['out']
        output = nn.Sigmoid()(output)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        loss = Loss(output, label)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




if __name__ == '__main__':
    DEVICE = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=False, num_classes=1,
                                                         aux_loss=None)
    # default: alpha 0.35 sigma 0.25
    loss = EnergyLoss(cuda=False, alpha=0.35, sigma=0.25)
    testLoss(model, loss)

