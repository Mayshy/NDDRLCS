import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils import data
import MTLDataset
import torchvision

# https://zhuanlan.zhihu.com/p/117435908
"""
二分类分割性能指标
输出 (batch_size, num_class, width, height) ， 输出可能未被归一化
标签 (batch_size, num_class, width, height) ， 独热编码,用[:, x: x+1, :]来找到第x个类

out shape torch.Size([16, 2, 224, 224])

"""


EPSILON = 1e-6

"""
DICE
两个体相交的面积占总面积的比值，范围是0~1.
dice = (2 * tp) / (2 * tp + fp + fn)
https://blog.csdn.net/baidu_36511315/article/details/105217674
使用方式：
1. pred[:, 0:1, :] pred[:, 1:2, :]
"""
def dice_loss(pred, gt,  activation='softmax2d'):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")
 
    pred = activation_fn(pred)
    smooth = EPSILON
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    intersection = (pred_flat * gt_flat).sum()
    return ((2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)).item()
    # tp = torch.sum(gt_flat * pred_flat, dim=1)
    # fp = torch.sum(pred_flat, dim=1) - tp
    # fn = torch.sum(gt_flat, dim=1) - tp
    # loss = (2 * tp + EPSILON) / (2 * tp + fp + fn + EPSILON)
    # return (loss.sum() / N).item()



# 也是错的
def sensitivity(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()
    # output = output > 0.5
    

    return (intersection + EPSILON) / (output.sum() + target.sum() + EPSILON)

"""
"""
def iou_score(output, target):

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + EPSILON) / (union + EPSILON)




def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label 2D tensor to a one-hot 3D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch siz. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor.

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> tgm.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 3:
        raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                         .format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)


# jaccard_index = make_weighted_metric(classwise_iou)
# f1_score = make_weighted_metric(classwise_f1)


if __name__ == '__main__':
    output, gt = torch.zeros(3, 2, 5, 5), torch.zeros(3, 5, 5).long()
    # print(classwise_iou(output, gt))
    pred = torch.Tensor([[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 1, 1, 0],
         [1, 0, 0, 1]]]])
    
    gt = torch.Tensor([[
            [[0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]],
            [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]],
            [[1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1]]]])

    pred2 = torch.Tensor([[
        [[0, 1, 0, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0]],
        [[1, 0, 1, 1],
         [0, 1, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 1]]],
        [
            [[0, 1, 0, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]],
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            [[1, 0, 1, 1],
             [0, 1, 1, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 1]]]
    ])
 
    gt2 = torch.Tensor([[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 1]]],
        [
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]],
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            [[1, 0, 0, 1],
             [0, 1, 1, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 1]]]
    ])
    print(pred.shape)
    print("diceLoss " + str(dice_loss(pred, gt)))
    print("iouReal? " + str(iou_score(pred, gt)))
    print("sensi " + str(sensitivity(pred, gt)))

    print("diceLoss " + str(dice_loss(pred2, gt2)))
    print("diceLoss " + str(dice_loss(pred2[:, 0:1, :], gt2[:, 0:1, :])))
    print("diceLoss " + str(dice_loss(pred2[:, 1:2, :], gt2[:, 1:2, :])))
    print("diceLoss " + str(dice_loss(pred2[:, 2:3, :], gt2[:, 2:3, :])))
    print("iouReal? " + str(iou_score(pred2, gt2)))
    print("sensi " + str(sensitivity(pred2, gt2)))

    data_root = "../ResearchData/UltraImageUSFullTest/UltraImageCropFull"
    seg_root = "../seg/"
    us_path = '../ResearchData/data_ultrasound_1.csv'
    NUM_CLASSES = 4
    BATCH_SIZE = 16
    rf_sort_list = ['SizeOfPlaqueLong', 'SizeOfPlaqueShort', 'DegreeOfCASWtihDiameter', 'Age', 'PSVOfCCA', 'PSVOfICA', 'DiameterOfCCA', 'DiameterOfICA', 'EDVOfICA', 'EDVOfCCA', 'RIOfCCA', 'RIOfICA', 'IMT', 'IMTOfICA', 'IMTOfCCA', 'Positio0fPlaque', 'Sex', 'IfAnabrosis', 'X0Or0']
    train_dataset = MTLDataset.SegDataset(
        str(data_root)+'TRAIN/', seg_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test = 'Train',screener=rf_sort_list,screen_num = 10)
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # target为一个batch的值
    target = iter(train_dataloader).next()
    DEVICE = torch.device('cuda:' + str(1) if torch.cuda.is_available() else 'cpu')
    img = target[1]
    seg_label = target[2].to(DEVICE)
    
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=2, aux_loss=None).to(DEVICE)

    output = model(img.to(DEVICE))['out']

    soft_output = nn.Softmax2d()(output)
    print(dice_loss(soft_output, seg_label))
    print(seg_label.shape)
    print(seg_label.sum())