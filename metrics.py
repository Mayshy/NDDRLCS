import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils import data
import MTLDataset
import torchvision
from hausdorff import hausdorff_distance
import SimpleITK as sitk
import seg_metrics.seg_metrics as sg
import numpy as np
import copy

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
def dice_index(pred, gt):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = EPSILON
    if torch.is_tensor(pred):
        output = pred.data.cpu().numpy()
    if torch.is_tensor(gt):
        target = gt.data.cpu().numpy()
    size = pred.shape[0]
    sum = 0.0
    for i in range(size):
        the_output = output[i, 0, ...]
        the_target = target[i, 0, ...]
        intersection = (the_output * the_target).sum()
        sum += ((2. * intersection + smooth) / (the_output.sum() + the_target.sum() + smooth))
    return sum / size








# Positive predictive value
def ppv(output, target):
    smooth = EPSILON
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()
    return (intersection + smooth) / (output.sum() + smooth)

# 默认为传进来的时候已经切片过了，(16,1,224,224)
def hausdorff_index(output, target, name="euclidean"):
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    assert output.shape == target.shape
    # assert len(output.shape) == 4
    size = output.shape[0]
    sum = 0
    for i in range(size):
        sum += hausdorff_distance(output[i, 0], target[i, 0], distance = name)
    return sum/size




# output (16,1,224,224)
# target (16,1,224,224)
# ['dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs','hd', 'hd95', 'msd', 'mdsd', 'stdsd']
def get_metrics(batch_output, batch_target, metrics_type):

    if torch.is_tensor(batch_output):
        batch_output = batch_output.data.cpu().numpy()
    if torch.is_tensor(batch_target):
        batch_target = batch_target.data.cpu().numpy()
    assert batch_output.shape == batch_target.shape
    assert len(batch_output.shape) == 4
    spacing = (1, 1)
    size = batch_output.shape[0]
    metrics = dict.fromkeys(metrics_type, 0)
    for i in range(size):
        output = batch_output[i, 0]
        target = batch_target[i, 0]
        labelPred = sitk.GetImageFromArray(output, isVector=False)
        labelPred.SetSpacing(spacing)
        labelTrue = sitk.GetImageFromArray(target, isVector=False)
        labelTrue.SetSpacing(spacing)  # spacing order (x, y, z)
        # voxel_metrics
        pred = output.astype(int)
        gdth = target.astype(int)
        fp_array = copy.deepcopy(pred)  # keep pred unchanged
        fn_array = copy.deepcopy(gdth)
        gdth_sum = np.sum(gdth)
        pred_sum = np.sum(pred)
        intersection = gdth & pred
        union = gdth | pred
        intersection_sum = np.count_nonzero(intersection)
        union_sum = np.count_nonzero(union)

        tp_array = intersection

        tmp = pred - gdth
        fp_array[tmp < 1] = 0

        tmp2 = gdth - pred
        fn_array[tmp2 < 1] = 0

        tn_array = np.ones(gdth.shape) - union

        tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

        smooth = EPSILON
        precision = (tp + smooth) / (pred_sum + smooth)
        recall = (tp + smooth) / (gdth_sum + smooth)

        false_positive_rate = (fp + smooth) / (fp + tn + smooth)
        false_negtive_rate = (fn + smooth) / (fn + tp + smooth)

        jaccard = (intersection_sum + smooth) / (union_sum + smooth)
        dice = (2 * intersection_sum + smooth) / (gdth_sum + pred_sum + smooth)

        dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
        dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)

        # distance_metrics
        signed_distance_map = sitk.SignedMaurerDistanceMap(labelTrue > 0.5, squaredDistance=False,
                                                          useImageSpacing=True)  # It need to be adapted.

        ref_distance_map = sitk.Abs(signed_distance_map)

        ref_surface = sitk.LabelContour(labelTrue > 0.5, fullyConnected=True)
        # ref_surface_array = sitk.GetArrayViewFromImage(ref_surface)

        statistics_image_filter = sitk.StatisticsImageFilter()
        statistics_image_filter.Execute(ref_surface > 0.5)

        num_ref_surface_pixels = int(statistics_image_filter.GetSum())

        signed_distance_map_pred = sitk.SignedMaurerDistanceMap(labelPred > 0.5, squaredDistance=False,
                                                                useImageSpacing=True)

        seg_distance_map = sitk.Abs(signed_distance_map_pred)

        seg_surface = sitk.LabelContour(labelPred > 0.5, fullyConnected=True)
        # seg_surface_array = sitk.GetArrayViewFromImage(seg_surface)

        seg2ref_distance_map = ref_distance_map * sitk.Cast(seg_surface, sitk.sitkFloat32)

        ref2seg_distance_map = seg_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)

        statistics_image_filter.Execute(seg_surface > 0.5)

        num_seg_surface_pixels = int(statistics_image_filter.GetSum())

        seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
        seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
        seg2ref_distances = seg2ref_distances + list(np.zeros(num_seg_surface_pixels - len(seg2ref_distances)))
        ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
        ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
        ref2seg_distances = ref2seg_distances + list(np.zeros(num_ref_surface_pixels - len(ref2seg_distances)))  #
        all_surface_distances = seg2ref_distances + ref2seg_distances

        metrics['dice'] += dice
        metrics['jaccard'] += jaccard
        metrics['precision'] += precision
        metrics['recall'] += recall
        metrics['fpr'] += false_positive_rate
        metrics['fnr'] += false_negtive_rate
        metrics['vs'] += dicecomputer.GetVolumeSimilarity()
        metrics["msd"] += np.mean(all_surface_distances)
        metrics["mdsd"] += np.median(all_surface_distances)
        metrics["stdsd"] += np.std(all_surface_distances)
        metrics["hd95"] += np.percentile(all_surface_distances, 95)
        metrics["hd"] += np.max(all_surface_distances)

    for key in metrics:
        metrics[key] /= size
    return metrics


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

def virtual_test():
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
        [[0, 0.6, 0.4, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]]
       ]])
    pred2[pred2 >= 0.5] = 1
    pred2[pred2 < 0.5] = 0
    gt2 = torch.Tensor([[
        [[0, 1, 0, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]]
    ]])
    print("dice " + str(dice_index(pred, gt)))
    # print("iouReal? " + str(iou_score(pred, gt)))
    print("sensi " + str(sensitivity(pred, gt)))

    print("dice " + str(dice_index(pred2, gt2)))
    print("dice0 " + str(dice_index(pred2[:, 0:1, :], gt2[:, 0:1, :])))
    print("dice1 " + str(dice_index(pred2[:, 1:2, :], gt2[:, 1:2, :])))
    print("dice2 " + str(dice_index(pred2[:, 2:3, :], gt2[:, 2:3, :])))
    print("ppv0 " + str(ppv(pred2[:, 0:1, :], gt2[:, 0:1, :])))
    print("ppv1 " + str(ppv(pred2[:, 1:2, :], gt2[:, 1:2, :])))
    print("ppv2 " + str(ppv(pred2[:, 2:3, :], gt2[:, 2:3, :])))
    # print("iouReal? " + str(iou_score(pred2, gt2)))
    print("sensi " + str(sensitivity(pred2, gt2)))
    theP = pred2[0, 0, :].detach().cpu().numpy()
    theT = gt2[0, 0, :].detach().cpu().numpy()
    print(hausdorff_distance(theP, theT, distance="haversine"))
    test_metrics = ['dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs', 'hd', 'hd95', 'msd', 'mdsd', 'stdsd']
    quality = sg.computeQualityMeasures(theP, theT, (1, 1), test_metrics)
    print(quality)

def data_test():
    data_root = "../ResearchData/UltraImageUSFullTest/UltraImageCropFull"
    seg_root = "../seg/"
    us_path = '../ResearchData/data_ultrasound_1.csv'
    NUM_CLASSES = 4
    BATCH_SIZE = 16
    rf_sort_list = ['SizeOfPlaqueLong', 'SizeOfPlaqueShort', 'DegreeOfCASWtihDiameter', 'Age', 'PSVOfCCA', 'PSVOfICA',
                    'DiameterOfCCA', 'DiameterOfICA', 'EDVOfICA', 'EDVOfCCA', 'RIOfCCA', 'RIOfICA', 'IMT', 'IMTOfICA',
                    'IMTOfCCA', 'Positio0fPlaque', 'Sex', 'IfAnabrosis', 'X0Or0']
    train_dataset = MTLDataset.SegDataset(
        str(data_root) + 'TRAIN/', seg_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test='Train',
        screener=rf_sort_list, screen_num=10)
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # target为一个batch的值
    target = iter(train_dataloader).next()
    DEVICE = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    img = target[1]
    seg_label = target[2].to(DEVICE)
    print(seg_label.shape)

    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=2,
                                                         aux_loss=None).to(DEVICE)

    output = model(img.to(DEVICE))['out']

    soft_output = nn.Softmax2d()(output)
    soft_output[soft_output >= 0.5] = 1
    soft_output[soft_output < 0.5] = 0

    print(soft_output[:, 1:2, :].shape)
    print("dice_index 1: {0}".format(dice_index(soft_output[:, 1:2, :], seg_label[:, 1:2, :])))
    print("ppv 1: {0}".format(ppv(soft_output[:, 1:2, :], seg_label[:, 1:2, :])))
    print("sensitivity 1: {0}".format(sensitivity(soft_output[:, 1:2, :], seg_label[:, 1:2, :])))
    print("Hausdorff 1: {0}".format(hausdorff_index(soft_output[:, 1:2, :], seg_label[:, 1:2, :])))

    test_metrics = ['dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs','hd', 'hd95', 'msd', 'mdsd', 'stdsd']

    print(get_metrics(soft_output[:, 1:2, :],seg_label[:, 1:2, :], test_metrics))

if __name__ == '__main__':
    data_test()
    # virtual_test()

