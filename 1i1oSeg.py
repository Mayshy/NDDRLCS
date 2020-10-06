# python的可读性真是太差了，这里约束一下：
# 每个函数前
import os
import numpy as np
import pandas as pd
import MTLDataset
import torchvision
import logging
import argparse
import torch
from torch import optim
from torch import nn
import MTLLoss
from MTLLoss import MultiLossLayer
from torch.utils import data
from torchvision import transforms
import sys
import datetime
import metrics
import random

"""
单输入单输出分割

seg_label 标签 torch.Size([16, 2, 224, 224])

要使用的性能指标:
1. pixel_accuray , 参考
2. 

# Q：灰度图是否归一化？？？
# Q: 未来要做p-value
"""

def log_mean(array, name, isLog = False):
    array = np.array(array)
    mean = np.mean(array)
    if isLog:
        logging.info("Epoch {0} {2} MEAN {1}".format(epoch, mean, name))
        logging.info("Epoch {0} {2} STD {1}".format(epoch, np.std(array), name))
    return mean

def dict_sum(res, addend):
    if not res:
        res = addend
    else:
        for k, v in res.items():
            res[k] += addend[k]
    return res

def get_model(model_name, pretrained = False):
    model_name = model_name.strip()
    if model_name == 'FCN_ResNet50':
        return torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained, progress=False, num_classes=2,
                                                            aux_loss=None).to(DEVICE)
    if model_name == 'FCN_ResNet101':
        return torchvision.models.segmentation.fcn_resnet101(pretrained=pretrained, progress=False, num_classes=2,
                                                            aux_loss=None).to(DEVICE)
    if model_name == 'DLV3__ResNet50':
        return torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=False, num_classes=2,
                                                             aux_loss=None).to(DEVICE)
    if model_name == 'DLV3__ResNet101':
        return torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=False, num_classes=2,
                                                             aux_loss=None).to(DEVICE)


# 配置分类损失函数
def get_criterion(criterion):
    criterion = criterion.strip()
    if criterion == "TheCrossEntropy":
        return MTLLoss.TheCrossEntropy()
    if criterion == "BCELoss":
        return nn.BCELoss()
    if criterion == "SSLoss":
        return
    if criterion == "DiceLoss":
        return MTLLoss.DiceLoss()
    if criterion == "IOULoss":
        return MTLLoss.mIoULoss()
    if criterion == "GDL":
        return MTLLoss.GDL()
    if criterion == "TverskyLoss":
        return
    # if criterion == "Hausdorff":
    #     return MTLLoss.GeomLoss(loss="hausdorff")
    if criterion == "HDLoss":
        return MTLLoss.HDLoss()

# 配置回归损失函数
def get_criterionUS(criterionUS):
    if criterionUS == "XTanh":
        return MTLLoss.XTanhLoss()
# 配置优化器
def get_optimizer(optimizer, multi_loss):
    if (optimizer == 'Adam'):
        return optim.Adam([{'params':model.parameters()},{'params':multi_loss.parameters()}],lr=LEARNING_RATE)
    elif (optimizer == 'SGD'):
        return optim.SGD([{'params':model.parameters()}, {'params':multi_loss.parameters()}], lr=LEARNING_RATE, momentum = MOMENTUM)
    elif (optimizer == 'AdamW'):
        return optim.AdamW([{'params':model.parameters()}, {'params':multi_loss.parameters()}], lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    elif (optimizer == 'AmsgradW'):
        return optim.AdamW([{'params':model.parameters()}, {'params':multi_loss.parameters()}], lr=LEARNING_RATE,weight_decay = WEIGHT_DECAY, amsgrad=True)
            
# mixup 数据增强器，帮助提升小数据集下训练与测试的稳定性
def mixup_data(x, y0, y1, y2, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y0_a, y0_b = y0, y0[index]
    y1_a, y1_b = y1, y1[index]
    y2_a, y2_b = y2, y2[index]
    return mixed_x, y0_a, y0_b, y1_a, y1_b, y2_a, y2_b, lam

def mixup_criterion_type(the_criterion, pred, y_a, y_b, lam):
    return lam * the_criterion(pred, y_a) + (1 - lam) * the_criterion(pred, y_b)

def mixup_criterion_numeric(the_criterion, pred, y_a, y_b, lam):
    return the_criterion(pred, lam * y_a + (1 - lam) * y_b)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 参数解析
def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, help="The Main Model", default='FCN_ResNet50')
    parser.add_argument("--pretrained", type=bool, help="if pretrained", default=False)
    parser.add_argument("--mode", type=str, help="Mode", default='NddrLSC')
    parser.add_argument("--optim", type=str, help="Optimizer", default='Adam')
    parser.add_argument("--criterion", type=str, help="criterion", default='TheCrossEntropy')
    parser.add_argument("--criterionUS", type=str, help="criterionUS", default='XTanh')
    parser.add_argument("--s_data_root", type=str, help="single data root", default='../ResearchData/UltraImageUSFullTest/UltraImageCropTransection')
    parser.add_argument("--seg_root", type=str, help="segmentation label root",
                        default='../seg/')
    parser.add_argument("--logging_level", type=int, help="logging level", default=20)
    parser.add_argument("--log_file_name", type=str, help="logging file name", default=str(datetime.date.today())+'.log')
    # parser.add_argument("--length_US", type=int, help="Length of US_x", default=32)
    parser.add_argument("--length_aux", type=int, help="Length of y", default=10)
    parser.add_argument("--n_class", type=int, help="number of classes", default=4)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.00002)
    parser.add_argument("--wd", type=float, help="weight decay", default=0.1)
    parser.add_argument("--momentum", type=float, help="momentum", default=0.1)
    parser.add_argument("--nddr_dr", type=float, help="nddr drop rate", default = 0)
    parser.add_argument("--epoch", type=int, help="number of epoch", default=900)
    parser.add_argument("--n_batch_size", type=int, help="mini batch size", default=16)
    parser.add_argument("--n_tarin_check_batch", type=int, help="mini num of check batch", default=1)
    parser.add_argument("--save_best_model", type=int, help="if saving best model", default=0)
    parser.add_argument("--save_optim", type=int, help="if saving optim", default=0)    
    parser.add_argument("--logdir", type=str, help="Please input the tensorboard logdir.", default=str(datetime.date.today()))
    parser.add_argument("--GPU", type=int, help="GPU ID", default=0)
    parser.add_argument("--alpha", type=int, help="If use mixup", default=1)
    return parser.parse_args(argv)


# 训练
# TODO 得想个办法把海量的loss都放一起处理，显得优雅 : 输出形式 log + 逐行写入CSV,待检验
def train(epoch):

    # 设置数据集
    train_dataset = MTLDataset.SegDataset(
        str(data_root) + 'TRAIN/', args.seg_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test='Train',
        screener=rf_sort_list, screen_num=10)
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    logging.info("Epoch " + str(epoch))
    train_loss_list = []
    model.train()
    # 开始一个epoch的迭代
    for i, (ID, img, seg_label, US_data, label4, label2) in enumerate(train_dataloader):
        # 数据分为两类， 算法的输入:img 算法的输出 seg_label ， （其他还没用到)
        img = img.to(DEVICE)
        seg_label = seg_label.to(DEVICE)
        US_data = US_data.to(DEVICE)
        US_label = US_data[:, :args.length_aux]
        label4 = label4.to(DEVICE)

        # mixup
        img, seg_label_a, seg_label_b, US_label_a, US_label_b, label4_a, label4_b, lam = mixup_data(img, seg_label,
                                                                                                    US_label, label4,
                                                                                                    args.alpha)

        # 执行模型，得到输出
        out = model(img)['out']
        # 可能需要做sigmoid？
        out = nn.Softmax2d()(out)

        # 取损失函数
        train_loss = mixup_criterion_type(criterion, out, seg_label_a, seg_label_b, lam)
        train_loss_list.append(train_loss.item())

        # 使用优化器执行反向传播
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    train_loss_mean = log_mean(train_loss_list, "train loss", isLog= True)
    all_train_loss_list.append(train_loss_mean)

def test(epoch):

    test_loss_list = []
    dice0 = []
    dice1 = []
    sen0 = []
    sen1 = []
    ppv0 = []
    ppv1 = []
    hausdorff = []
    # 设置数据集
    test_dataset = MTLDataset.SegDataset(
        str(data_root) + 'TEST/', args.seg_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test='Test',
        screener=rf_sort_list,
        screen_num=10)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    epoch_quality = {}
    batch_num = 0
    # 开始这轮的迭代
    with torch.no_grad():
        for i, (ID, img, seg_label, US_data, label4, label2) in enumerate(test_dataloader):
            batch_num += 1
            # 数据分为两类， 算法的输入:img 算法的输出 seg_label ， （其他还没用到)
            img = img.to(DEVICE)
            seg_label = seg_label.to(DEVICE)
            US_data = US_data.to(DEVICE)
            US_label = US_data[:, :args.length_aux]
            label4 = label4.to(DEVICE)

            # 输出
            output = model(img)['out']
            output = nn.Softmax2d()(output)
            # seg_label 标签
            loss = criterion(output, seg_label)

            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            seg_test = seg_label.long()
            # output0 = output[:, 0:1, :]
            output1 = output[:, 1:2, :]
            # seg_test0 = seg_test[:, 0:1, :]
            seg_test1 = seg_test[:, 1:2, :]

            # 记录Loss，计算性能指标
            # logging.info("Epoch {0} TestLoss {1}".format(epoch, loss.item()))
            test_loss_list.append(loss.item())
            dice1.append(metrics.dice_index(output1, seg_test1))
            sen1.append(metrics.sensitivity(output1, seg_test1))
            ppv1.append(metrics.ppv(output1, seg_test1))
            hausdorff.append(metrics.hausdorff_index(output1, seg_test1))
            quality = metrics.get_metrics(output1, seg_test1, count_metrics)




            if i == 0:
                output = output.cpu()
                for j in range(BATCH_SIZE):
                    save_img = transforms.ToPILImage()(output[j][1]).convert('L')
                    path = '../Log/' + args.logdir + '/V' + str(j) + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    save_img.save(path + 'E' + str(epoch) + '_' + ID[j] + '.jpg')

    test_loss_mean = log_mean(test_loss_list, "test loss", isLog= True)
    dice0_mean = log_mean(dice0, "dice0", isLog= True)
    dice1_mean = log_mean(dice1, "dice1", isLog= True)
    sen0_mean = log_mean(sen0, "sen0", isLog= True)
    sen1_mean = log_mean(sen1, "sen1", isLog= True)
    ppv0_mean = log_mean(ppv0, "ppv0", isLog= True)
    ppv1_mean = log_mean(ppv1, "ppv1", isLog= True)
    hausdorff_mean = log_mean(hausdorff, "hausdorff", isLog= True)
    for k, v in epoch_quality.items():
        epoch_quality[k] /= batch_num
    logging.info("Epoch {0} MEAN {1}".format(epoch, epoch_quality))
    all_test_loss_list.append(test_loss_mean)
    all_dice0.append(dice0_mean)
    all_dice1.append(dice1_mean)
    all_sen0.append(sen0_mean)
    all_sen1.append(sen1_mean)
    all_ppv0.append(ppv0_mean)
    all_ppv1.append(ppv1_mean)
    all_hausdorff.append(hausdorff_mean)

    for k, v in epoch_quality.items():
        all_quality[k].append(epoch_quality[k])

        


# 启动配置
# ------------------------------------------------------------------------------------------------------------------------------------------
# # Meter用于度量波动区间
# batch_train_loss_meter = meter.AverageValueMeter()
# train_loss_meter = meter.AverageValueMeter()
# batch_test_loss_meter = meter.AverageValueMeter()
# test_loss_meter = meter.AverageValueMeter()

# 配置特征排序（和引用的特征量）
setup_seed(20)
rf_sort_list = ['SizeOfPlaqueLong', 'SizeOfPlaqueShort', 'DegreeOfCASWtihDiameter', 'Age', 'PSVOfCCA', 'PSVOfICA', 'DiameterOfCCA', 'DiameterOfICA', 'EDVOfICA', 'EDVOfCCA', 'RIOfCCA', 'RIOfICA', 'IMT', 'IMTOfICA', 'IMTOfCCA', 'Positio0fPlaque', 'Sex', 'IfAnabrosis', 'X0Or0']
# 写入配置
args = parse_args(sys.argv[1:])
start_epoch = 1
DEVICE = torch.device('cuda:' + str(args.GPU) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.GPU)
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.wd
model = get_model(args.net, args.pretrained)
criterion = get_criterion(args.criterion)
criterionUS = get_criterionUS(args.criterionUS)
multi_loss = MultiLossLayer(2)
multi_loss.to(DEVICE)
optimizer = get_optimizer(args.optim, multi_loss)
data_root = args.s_data_root.strip()
NUM_CLASSES = 4
BATCH_SIZE = args.n_batch_size
NUM_TRAIN_CHECK_BATCHES = 4
us_path = '../ResearchData/data_ultrasound_1.csv'
count_metrics = ['dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs','hd', 'hd95', 'msd', 'mdsd', 'stdsd']
quality_keys = ['dice', 'jaccard', 'precision', 'recall', 'false_positive_rate','false_negtive_rate','volume_similarity', 'Hausdorff', '95_surface_distance', 'mean_surface_distance', 'median_surface_distance', 'std_surface_distance']

log_path = '../Log/' + str(args.logdir).strip()  +'/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
logging.basicConfig(level=args.logging_level,filename= log_path + str(args.log_file_name) ,
                    filemode='a', format='%(asctime)s   %(levelname)s   %(message)s')
logging.warning('Model: {}  Mode:{} Loss:{} Data:{}'.format(args.net, args.mode, args.criterion, args.s_data_root))

## metrics
# jaccard_index = metrics.make_weighted_metric(metrics.classwise_iou)
# f1_score = metrics.make_weighted_metric(metrics.classwise_f1)

all_train_loss_list = []
all_test_loss_list = []
all_dice0 = []
all_dice1 = []
all_sen0 = []
all_sen1 = []
all_ppv0 = []
all_ppv1 = []
all_hausdorff = []
all_quality = dict.fromkeys(quality_keys, [])

for epoch in range(start_epoch, args.epoch):
    batch_test_loss = []
    batch_pixel_accuracy = []
    batch_IOU = []
    batch_DICE = []
    batch_hausdorff = []
    train(epoch)
    test(epoch)

log_mean(all_train_loss_list, "all_train_loss_list", isLog=True)
log_mean(all_test_loss_list, "all_test_loss_list", isLog=True)
log_mean(all_dice0, "all_dice0", isLog=True)
log_mean(all_dice1, "all_dice1", isLog=True)
log_mean(all_sen0, "all_sen0", isLog=True)
log_mean(all_sen1, "all_sen1", isLog=True)
log_mean(all_ppv0, "all_ppv0", isLog=True)
log_mean(all_ppv0, "all_ppv0", isLog=True)
log_mean(all_hausdorff, "all_hausdorff", isLog=True)
all_quality['train_loss'] = all_train_loss_list
all_quality['test_loss'] = all_test_loss_list
df = pd.DataFrame(all_quality)
df.to_csv(log_path + str(args.log_file_name) + ".csv")
# res = []
# res.append(all_train_loss_list)
# res.append(all_test_loss_list)
# res.append(all_dice1)
# res.append(all_sen1)
# res.append(all_ppv1)
# res.append(all_hausdorff)
# res.append(all_train_loss_list)
# res.append(all_train_loss_list)

