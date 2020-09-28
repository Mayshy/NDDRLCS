import torch
from torch import nn
import torch.nn.functional as F
import metrics
import geomloss
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
"""
语义分割常用损失函数
https://blog.csdn.net/CaiDaoqing/article/details/90457197
"""


class MultiLossLayer(nn.Module):
    def __init__(self, list_length):
        super(MultiLossLayer, self).__init__()
        self._sigmas_sq = nn.ParameterList([nn.Parameter(torch.empty(())) for i in range(list_length)])
        for p in self.parameters():
            nn.init.uniform_(p,0.2,1)
            # 初始化采用和原论文一样的方法......可能需要调整
        

    def forward(self,  loss0, loss1):
        factor0 = torch.div(1.0,torch.mul(self._sigmas_sq[0], 2.0))
        loss = torch.add(torch.mul(factor0, loss0), 0.5*torch.log(self._sigmas_sq[0]))
        
        factor1 = torch.div(1.0,torch.mul(self._sigmas_sq[1], 2.0))
        loss = torch.add(loss, torch.add(torch.mul(factor1, loss1), 0.5*torch.log(self._sigmas_sq[1])))

        return loss


# class MultiLossLayer(nn.Module):
#     def __init__(self, list_length):
#         super(MultiLossLayer, self).__init__()
#         self._sigmas_sq = nn.ParameterList([nn.Parameter(torch.empty(())) for i in range(list_length)])
#         for p in self.parameters():
#             nn.init.uniform_(p,0.2,1.0)
#         
#     def forward(self, regression_loss, classifier_loss):
#         # regression loss
#         factor0 = torch.div(1.0,torch.mul(self._sigmas_sq[0], 2.0))
#         loss = torch.add(torch.mul(factor0, regression_loss), 0.5 * torch.log(self._sigmas_sq[0]))
#         # classification loss
#         factor1 = torch.div(1.0,torch.mul(self._sigmas_sq[1], 1.0))
#         loss = torch.add(loss, torch.add(torch.mul(factor1,classifier_loss), 0.5 * torch.log(self._sigmas_sq[1])))

#         return loss

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super(XTanhLoss,self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super(XSigmoidLoss,self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)

# https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/dice.html
# TODO:验证它的正确性
class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            inputs: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        
        # compute softmax over the classes axis
        # input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        # target_one_hot = target
        # 这里，target传进来时就是one-hot了
        # target_one_hot = metrics.one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

        # compute the actual dice score
 
        # dims = (1, 2, 3)
        # intersection = torch.sum(input * target, dims)
        # cardinality = torch.sum(input + target, dims)

        # dice_score = 2. * intersection / (cardinality + self.eps)
        # return torch.mean(1. - dice_score)
        score = inputs[:, 1, ...]
        target = target[:, 1, ...]
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + self.eps) / (z_sum + y_sum + self.eps)
        loss = 1 - loss
        return loss
'''

def generalised_dice_loss_2d_ein(Y_gt, Y_pred):
    Y_gt = tf.cast(Y_gt, 'float32')
    Y_pred = tf.cast(Y_pred, 'float32')
    w = tf.einsum("bwhc->bc", Y_gt)
    w = 1 / ((w + 1e-10) ** 2)
    intersection = w * tf.einsum("bwhc,bwhc->bc", Y_pred, Y_gt)
    union = w * (tf.einsum("bwhc->bc", Y_pred) + tf.einsum("bwhc->bc", Y_gt))

    divided = 1 - 2 * (tf.einsum("bc->b", intersection) + 1e-10) / (tf.einsum("bc->b", union) + 1e-10)

    loss = tf.reduce_mean(divided)
    return loss


def generalised_dice_loss_2d(Y_gt, Y_pred):
    smooth = 1e-5
    w = tf.reduce_sum(Y_gt, axis=[1, 2])
    w = 1 / (w ** 2 + smooth)

    numerator = Y_gt * Y_pred
    numerator = w * tf.reduce_sum(numerator, axis=[1, 2])
    numerator = tf.reduce_sum(numerator, axis=1)

    denominator = Y_pred + Y_gt
    denominator = w * tf.reduce_sum(denominator, axis=[1, 2])
    denominator = tf.reduce_sum(denominator, axis=1)

    gen_dice_coef = 2 * numerator / (denominator + smooth)
    loss = tf.reduce_mean(1 - gen_dice_coef)
    return loss
'''
class GDL(nn.Module):
    def __init__(self) -> None:
        super(GDL, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        w = torch.einsum("bwhc->bc", target)
        w = 1 / ((w + 1e-10) ** 2)
        intersection = w * torch.einsum("bwhc,bwhc->bc", input, target)
        union = w * (torch.einsum("bwhc->bc", input) + torch.einsum("bwhc->bc", target))
        divided = 1 - 2 * (torch.einsum("bc->b", intersection) + self.eps) / (torch.einsum("bc->b", union) + self.eps)
        loss = torch.mean(torch.sum(divided))
        return loss

# soft IOU https://discuss.pytorch.org/t/how-to-implement-soft-iou-loss/15152
class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_oneHot):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = inter / union

        ## Return average loss over classes and batch
        return - loss.mean()
        
class BoundaryLoss(nn.Module):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: softmax results,  shape=(b,2,x,y)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y)
    output: boundary_loss; sclar
    """
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, outputs_soft, gt_sdf):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        pc = outputs_soft[:,1,...]
        dc = gt_sdf[:,1,...]
        multipled = torch.einsum('bxy, bxy->bxy', pc, dc)
        bd_loss = multipled.mean()

        return bd_loss

class HDLoss(nn.Module):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,2,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,2,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """
    def __init__(self):
        super(HDLoss, self).__init__()

    def forward(self, outputs_soft, label_batch):
        label_batch = label_batch[:, 1, ...]
        with torch.no_grad():
            gt_dtm_npy = compute_dtm(label_batch.cpu().numpy(), outputs_soft.shape)
            gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(outputs_soft.device.index)
            seg_dtm_npy = compute_dtm(outputs_soft[:, 1, ...].cpu().numpy()>0.5, outputs_soft.shape)
            seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(outputs_soft.device.index)

        delta_s = (outputs_soft[:,1,...] - label_batch.float()) ** 2
        s_dtm = seg_dtm[:,1,...] ** 2
        g_dtm = gt_dtm[:,1,...] ** 2
        dtm = s_dtm + g_dtm
        multipled = torch.einsum('bxy, bxy->bxy', delta_s, dtm)
        hd_loss = multipled.mean()

        return hd_loss
    


def compute_dtm(img_gt, out_shape):
        """
        compute the distance transform map of foreground in binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the foreground Distance Map (SDM) 
        dtm(x) = 0; x in segmentation boundary
                inf|x-y|; x in segmentation
        """

        fg_dtm = np.zeros(out_shape)

        for b in range(out_shape[0]): # batch size
            for c in range(1, out_shape[1]):
                posmask = img_gt[b].astype(np.bool)
                if posmask.any():
                    posdis = distance(posmask)
                    fg_dtm[b][c] = posdis

        return fg_dtm

class TheCrossEntropy(nn.Module):
    def __init__(self):
        super(TheCrossEntropy, self).__init__()

    def forward(self, outputs, label_batch):
        target = label_batch[:, 1, :].long()
        return F.cross_entropy(outputs, target)


class RHD(nn.Module):
    '''
    set λ (for the next epoch) as the ratio of the mean of the HD-based loss term to
    the mean of the DSC loss term. 
    '''
    def __init__(self):
        super(RHD, self).__init__()
        self.HDLoss = HDLoss()
        self.DiceLoss = DiceLoss()
        self.CrossEntropy = TheCrossEntropy()
        self.lam = 1 #TODO

    def forward(self, outputs, label_batch):
        loss_hd = self.HDLoss(outputs, label_batch)
        loss_dice = self.DiceLoss(outputs, label_batch)
        loss_ce = self.CrossEntropy(outputs, label_batch)
        loss = self.lam *(loss_ce+loss_seg_dice) + (1 - alpha) * loss_hd

if __name__ == '__main__':
    x = 1
