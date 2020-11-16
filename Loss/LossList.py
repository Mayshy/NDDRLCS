import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import metrics
import geomloss
import numpy as np
from math import exp
from scipy.ndimage import distance_transform_edt as distance

from Loss.GeneralisedLoss import GeneralizedWassersteinDiceLoss
from Loss.lovasz_losses import lovasz_hinge, lovasz_softmax, symmetric_lovasz


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


def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b


class IouLoss(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IouLoss, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)

# https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/dice.html
class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            inputs: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        dims = (1, 2, 3)
        intersection = torch.sum(inputs * target, dims)
        cardinality = torch.sum(inputs + target, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)
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
    def __init__(self, weight=None, size_average=True, n_classes=1):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_oneHot):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        # inputs = F.softmax(inputs, dim=1)

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
        label_batch = label_batch[:, 0, ...]
        with torch.no_grad():
            gt_dtm_npy = compute_dtm(label_batch.cpu().numpy(), outputs_soft.shape)
            gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(outputs_soft.device.index)
            seg_dtm_npy = compute_dtm(outputs_soft[:, 0, ...].cpu().numpy()>0.5, outputs_soft.shape)
            seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(outputs_soft.device.index)

        delta_s = (outputs_soft[:,0,...] - label_batch.float()) ** 2
        s_dtm = seg_dtm[:,0,...] ** 2
        g_dtm = gt_dtm[:,0,...] ** 2
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


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    # weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible,
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)


class LovaszSoftmax(torch.nn.Module):
    def __init__(self, per_image=False):
        super(LovaszSoftmax, self).__init__()
        self.per_image = per_image

    # probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
    #               Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
    #       labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
    #       classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    def forward(self, outputs_soft, label_batch):
        label_batch = label_batch[:, 0, :]

        return lovasz_softmax(outputs_soft, label_batch, classes=[1], per_image=self.per_image)

class LovaszHinge(torch.nn.Module):
    def __init__(self, per_image=False):
        super(LovaszHinge, self).__init__()
        self.per_image = per_image

    def forward(self, outputs_soft, label_batch):
        return lovasz_hinge(outputs_soft, label_batch, per_image=self.per_image)


class SymmetricLovasz(torch.nn.Module):
    def __init__(self, per_image=False):
        super(SymmetricLovasz, self).__init__()
        self.per_image = per_image

    def forward(self, outputs_soft, label_batch):
        return symmetric_lovasz(outputs_soft, label_batch, per_image=self.per_image)


# https://www.zhihu.com/question/272988870/answer/562262315
# 在loss后面再加这个？
def focal_loss(output, target, alpha, gamma, OHEM_percent):
        output = output.contiguous().view(-1)
        target = target.contiguous().view(-1)

        max_val = (-output).clamp(min=0)
        loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-output * (target * 2 - 1))
        focal_loss = alpha * (invprobs * gamma).exp() * loss

        # Online Hard Example Mining: top x% losses (pixel-wise). Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
        OHEM, _ = focal_loss.topk(k=int(OHEM_percent * [*focal_loss.shape][0]))
        return OHEM.mean()

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha, gamma, OHEM_percent):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.OHEM_percent = OHEM_percent

    def forward(self, outputs_soft, label_batch):
        return focal_loss(outputs_soft, label_batch, alpha=self.alpha, gamma=self.gamma, OHEM_percent=self.OHEM_percent)

# https://github.com/LucasFidon/GeneralizedWassersteinDiceLoss
# 用于num_class=1时的二分类，其他时候请调整dist_mat和output
class GWDL(torch.nn.Module):
    # weighting_mode='default', GWDL
    # weighting_mode='GDL', GDL
    def __init__(self, weighting_mode='default', reduction='mean'):
        super(GWDL, self).__init__()
        dist_mat = np.array([
            [0., 1.],
            [1., 0.],
        ])
        self.gwdl = GeneralizedWassersteinDiceLoss(dist_mat, weighting_mode=weighting_mode, reduction=reduction)

    def forward(self, outputs_soft, label_batch):
        two_channel_output = torch.cat((outputs_soft, 1 - outputs_soft), dim=1)
        return self.gwdl(two_channel_output, label_batch)

# https://github.com/dmitrysarov/clDice
# https://github.com/LucasFidon/GeneralizedWassersteinDiceLoss
class CLDiceLoss(torch.nn.Module):
    def __init__(self, target_skeleton=None, combination=False):
        super(CLDiceLoss, self).__init__()
        self.target_skeleton = target_skeleton
        self.combination = combination
        if combination:
            dist_mat = np.array([
                [0., 1.],
                [1., 0.],
            ])
            self.soft_dice = GeneralizedWassersteinDiceLoss(dist_mat, weighting_mode='GDL')


    def forward(self, outputs_soft, label_batch):
        if self.combination:
            # 只支持二分类
            two_channel_output = torch.cat((outputs_soft, 1 - outputs_soft),dim=1)
            return 0.5 * torch.mean(CLDiceLoss.soft_cldice_loss(outputs_soft, label_batch, target_skeleton=self.target_skeleton)) + 0.5 * self.soft_dice(two_channel_output, label_batch)
        return torch.mean(CLDiceLoss.soft_cldice_loss(outputs_soft, label_batch, target_skeleton=self.target_skeleton))

    @staticmethod
    def soft_cldice_loss(pred, target, target_skeleton=None):
        '''
        inputs shape  (batch, channel, height, width).
        calculate clDice loss
        Because pred and target at moment of loss calculation will be a torch tensors
        it is preferable to calculate target_skeleton on the step of batch forming,
        when it will be in numpy array format by means of opencv
        '''
        cl_pred = CLDiceLoss.soft_skeletonize(pred)
        if target_skeleton is None:
            target_skeleton = CLDiceLoss.soft_skeletonize(target)
        iflat = CLDiceLoss.norm_intersection(cl_pred, target)
        tflat = CLDiceLoss.norm_intersection(target_skeleton, pred)
        intersection = iflat * tflat

        return -((2. * intersection) /
                 (iflat + tflat))
    @staticmethod
    def soft_skeletonize(x, thresh_width=10):
        '''
        Differenciable aproximation of morphological skelitonization operaton
        thresh_width - maximal expected width of vessel
        '''
        for i in range(thresh_width):
            min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
            contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
            x = torch.nn.functional.relu(x - contour)
        return x

    @staticmethod
    def norm_intersection(center_line, vessel):
        '''
        inputs shape  (batch, channel, height, width)
        intersection formalized by first ares
        x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
        '''
        smooth = 1.
        clf = center_line.view(*center_line.shape[:2], -1)
        vf = vessel.view(*vessel.shape[:2], -1)
        intersection = (clf * vf).sum(-1)
        return (intersection + smooth) / (clf.sum(-1) + smooth)
    @staticmethod
    def dice_loss(pred, target):
        '''
        inputs shape  (batch, channel, height, width).
        calculate dice loss per batch and channel of sample.
        E.g. if batch shape is [64, 1, 128, 128] -> [64, 1]
        '''
        smooth = 1.
        iflat = pred.view(*pred.shape[:2], -1)  # batch, channel, -1
        tflat = target.view(*target.shape[:2], -1)
        intersection = (iflat * tflat).sum(-1)
        return -((2. * intersection + smooth) /
                 (iflat.sum(-1) + tflat.sum(-1) + smooth))




def get_criterion(criterion):
    criterion = criterion.strip()
    if criterion == "TheCrossEntropy":
        return TheCrossEntropy()
    elif criterion == "BCELoss":
        return nn.BCEWithLogitsLoss()
    elif criterion == "DiceLoss":
        return DiceLoss()
    elif criterion == "mIOULoss":
        return mIoULoss()
    elif criterion == "IouLoss":
        return IouLoss()
    elif criterion == "GDL":
        # logits
        return GWDL(weighting_mode='GDL')
    elif criterion == "GWDL":
        # logits
        return GWDL(weighting_mode='default')
    elif criterion == "LovaszSoftmax":
        # logits
        return LovaszSoftmax(per_image=True)
    elif criterion == "LovaszHinge":
        # logits
        return LovaszHinge(per_image=True)
    # if criterion == "Hausdorff":
    #     return MTLLoss.GeomLoss(loss="hausdorff")
    if criterion == "HDLoss":
        return HDLoss()




