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
import torch.nn.functional as F
from Loss import LossList
from Model import UNet
from torch.utils import data
from torchvision import transforms
import sys
import datetime
import metrics
import random
import collections

from Model._utils import get_criterion, adjust_learning_rate

"""
单输入单输出分割

seg_label 标签 torch.Size([16, 2, 224, 224])

训练方案：https://blog.csdn.net/qq_34914551/article/details/87699317
https://zhuanlan.zhihu.com/p/93624972
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
        for k in res:
            res[k] += addend[k]
    return res



def get_model(model_name, pretrained = False):
    model_name = model_name.strip()
    if model_name == 'FCN_ResNet50':
        return torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained, progress=False, num_classes=1,
                                                            aux_loss=None)
    if model_name == 'FCN_ResNet101':
        return torchvision.models.segmentation.fcn_resnet101(pretrained=pretrained, progress=False, num_classes=1,
                                                            aux_loss=None)
    if model_name == 'DLV3__ResNet50':
        return torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=False, num_classes=1,
                                                             aux_loss=None)
    if model_name == 'DLV3__ResNet101':
        return torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=False, num_classes=1,
                                                             aux_loss=None)
    if model_name == "unet":
        return UNet.UNet(n_channels=3, n_classes=1)
    if model_name == "kinet":
        return UNet.KiNet(n_channels=3, n_classes=1)
    if model_name == "autoencoder":
        return UNet.autoencoder()
    if model_name == "kiunet":
        return UNet.kiunet()
    if model_name == "kinetwithsk":
        return UNet.kinetwithsk()
    if model_name == "dunet":
        return UNet.DilatedUNet(n_channels=3, n_classes=1)
    # elif model_name == "pspnet":
    #     return psp.PSPNet(layers=5, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True,
    #                        pretrained=False).cuda()


# 配置分类损失函数


# 配置回归损失函数
def get_criterionUS(criterionUS):
    if criterionUS == "XTanh":
        return LossList.XTanhLoss()
# 配置优化器
def get_optimizer(optimizer):
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

def get_dataloader(dataset_type):
    assert dataset_type == 'train' or 'test'
    if dataset_type == 'train':
        if args.intersection == 1:
            train_dataset = MTLDataset.SegDataset(
                str(data_root) + 'TRAIN/', args.seg_root, fluid_root=args.fluid_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test='Train',
                screener=rf_sort_list, screen_num=10)
        else:
            train_dataset = MTLDataset.SegDataset(
                str(data_root) + 'TRAIN/', args.seg_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test='Train',
                screener=rf_sort_list, screen_num=10)
        return data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        if args.intersection == 1:
            test_dataset = MTLDataset.SegDataset(
                str(data_root) + 'TEST/', args.seg_root, fluid_root=args.fluid_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test='Test',
                screener=rf_sort_list, screen_num=10)

        else:
            test_dataset = MTLDataset.SegDataset(
                str(data_root) + 'TEST/', args.seg_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test='Test',
                screener=rf_sort_list, screen_num=10)
        return data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False)







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
    parser.add_argument("--net", type=str, help="The Main Model", default='unet')
    parser.add_argument("--pretrained", type=bool, help="if pretrained", default=False)
    parser.add_argument("--mode", type=str, help="Mode", default='NddrLSC')
    parser.add_argument("--optim", type=str, help="Optimizer", default='Adam')
    parser.add_argument("--criterion", type=str, help="criterion", default='DiceLoss')
    parser.add_argument("--criterionUS", type=str, help="criterionUS", default='XTanh')
    parser.add_argument("--s_data_root", type=str, help="single data root", default='../ResearchData/UltraImageUSFullTest/UltraImageCropWithoutMannualResize')
    parser.add_argument("--seg_root", type=str, help="segmentation label root",
                        default='../BlurBinaryLabel/')
    parser.add_argument("--fluid_root", type=str, help="fluid data root",
                        default='../flowImage/')
    parser.add_argument("--intersection", type=int, help="if use the intersection of fluid_root",
                        default=0)
    parser.add_argument("--logging_level", type=int, help="logging level", default=20)
    parser.add_argument("--log_file_name", type=str, help="logging file name", default=str(datetime.date.today())+'.log')
    # parser.add_argument("--length_US", type=int, help="Length of US_x", default=32)
    parser.add_argument("--length_aux", type=int, help="Length of y", default=10)
    parser.add_argument("--n_class", type=int, help="number of classes", default=4)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=0)
    parser.add_argument("--momentum", type=float, help="momentum", default=0)
    parser.add_argument("--nddr_dr", type=float, help="nddr drop rate", default = 0)
    parser.add_argument("--epoch", type=int, help="number of epoch", default=200)
    parser.add_argument("--n_batch_size", type=int, help="mini batch size", default=8)
    parser.add_argument("--n_tarin_check_batch", type=int, help="mini num of check batch", default=1)
    parser.add_argument("--save_best_model", type=int, help="if saving best model", default=0)
    parser.add_argument("--save_optim", type=int, help="if saving optim", default=0)    
    parser.add_argument("--logdir", type=str, help="Please input the tensorboard logdir.", default=str(datetime.date.today()))
    parser.add_argument("--GPU", type=int, help="GPU ID", default=0)
    parser.add_argument("--alpha", type=int, help="If use mixup", default=0)
    parser.add_argument("--ifDataParallel", type=int, help="If use mixup", default=0)

    return parser.parse_args(argv)



# 训练
def train(epoch):
    # 设置数据集
    train_dataloader = get_dataloader('train')

    logging.info("Epoch " + str(epoch))
    train_loss_list = []
    model.train()
    # 开始一个epoch的迭代

    for i, (ID, img, seg_label, US_data, label4, label2) in enumerate(train_dataloader):
        # 数据分为两类， 算法的输入:img 算法的输出 seg_label ， （其他还没用到)
        img = img.to(DEVICE)
        seg_label = seg_label.to(DEVICE)
        if args.criterion.strip() == 'BCELoss':
            seg_label = seg_label.float()
        US_data = US_data.to(DEVICE)
        US_label = US_data[:, :args.length_aux]
        label4 = label4.to(DEVICE)

        # mixup
        img, seg_label_a, seg_label_b, US_label_a, US_label_b, label4_a, label4_b, lam = mixup_data(img, seg_label,
                                                                                                    US_label, label4,
                                                                                                    args.alpha)
        # 执行模型，得到输出
        out = model(img)['out']
        # out = model(img)
        # out = nn.Sigmoid()(out)
        # out = F.interpolate(out, size= , mode='bilinear')

        # 取损失函数
        # train_loss = criterion(out, seg_label)
        # out = (out >= 0.5).float().requires_grad_()
        train_loss = mixup_criterion_type(criterion, out, seg_label_a, seg_label_b, lam)
        train_loss_list.append(train_loss.item())

        # 使用优化器执行反向传播
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    train_loss_mean = log_mean(train_loss_list, "train loss", isLog= True)
    all_quality['train_loss'].append(train_loss_mean)
    return train_loss_mean

def test(epoch):
    test_loss_list = []
    dice2_list = []
    # 设置数据集
    test_dataloader = get_dataloader('test')
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
            if args.criterion.strip() == 'BCELoss':
                seg_label = seg_label.float()
            US_data = US_data.to(DEVICE)
            US_label = US_data[:, :args.length_aux]
            label4 = label4.to(DEVICE)

            # 输出
            output = model(img)['out']
            loss = criterion(output, seg_label)
            output = nn.Sigmoid()(output)

            # seg_label 标签

            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            output = F.interpolate(output, size=512, mode='bilinear', align_corners=True)
            seg_test = seg_label.long()



            # 记录Loss，计算性能指标
            # logging.info("Epoch {0} TestLoss {1}".format(epoch, loss.item()))
            test_loss_list.append(loss.item())
            dice2 = metrics.dice_index(output, seg_test)
            dice2_list.append(dice2)
            quality = metrics.get_sum_metrics(output, seg_test, count_metrics)
            epoch_quality = dict_sum(epoch_quality, quality)
            if i == 0:
                output = output.cpu()
                for j in range(BATCH_SIZE):
                    save_img = transforms.ToPILImage()(output[j][0]).convert('L')
                    path = '../Log/' + args.logdir + '/V' + str(j) + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    save_img.save(path + 'E' + str(epoch) + '_' + ID[j] + '.jpg')

    test_loss_mean = log_mean(test_loss_list, "test loss", isLog= True)
    dice2_mean = log_mean(dice2_list, "dice2", isLog= True)
    for k in epoch_quality:
        epoch_quality[k] /= len(test_dataloader.dataset)
    logging.info("Epoch {0} MEAN {1}".format(epoch, epoch_quality))

    for k in epoch_quality:
        all_quality[k].append(epoch_quality[k])
    all_quality['test_loss'].append(test_loss_mean)
    all_quality['dice2'].append(dice2_mean)

# 启动配置
# ------------------------------------------------------------------------------------------------------------------------------------------
# # Meter用于度量波动区间


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
MOMENTUM = args.momentum
# model = get_model(args.net, args.pretrained).to(DEVICE)
if args.ifDataParallel:
    model = torch.nn.DataParallel(get_model(args.net, args.pretrained), device_ids=[0, 1]).cuda()
else:
    model = get_model(args.net, args.pretrained).to(DEVICE)
criterion = get_criterion(args.criterion).to(DEVICE)

optimizer = get_optimizer(args.optim)
data_root = args.s_data_root.strip()
NUM_CLASSES = 4
BATCH_SIZE = args.n_batch_size
NUM_TRAIN_CHECK_BATCHES = 4
us_path = '../ResearchData/data_ultrasound_1.csv'
count_metrics = ['dice', 'jaccard', 'ppv', 'precision', 'recall', 'fpr', 'fnr', 'vs','hd', 'hd95', 'msd', 'mdsd', 'stdsd']
all_metrics = ['train_loss', 'test_loss', 'ppv', 'dice', 'dice2', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs','hd', 'hd95', 'msd', 'mdsd', 'stdsd']

log_path = '../Log/' + str(args.logdir).strip()  +'/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
logging.basicConfig(level=args.logging_level,filename= log_path + str(args.log_file_name) ,
                    filemode='a', format='%(asctime)s   %(levelname)s   %(message)s')
logging.warning('Model: {}  Mode:{} LR:{} Loss:{} Data:{}'.format(args.net, args.mode, args.lr, args.criterion, args.s_data_root))

all_quality = collections.defaultdict(list)
last_train_loss = float('inf')
count_loss_improve = 0

for epoch in range(start_epoch, args.epoch):
    adjust_learning_rate(optimizer, epoch, args.lr)
    cur_train_loss = train(epoch)
    if (cur_train_loss >= last_train_loss):
        count_loss_improve += 1
    else:
        count_loss_improve = 0
    last_train_loss = cur_train_loss
    # if (count_loss_improve != 0 & count_loss_improve % 3 == 0):
    #     optimizer.param_groups[0]['lr'] *= 0.8
    test(epoch)
    if (count_loss_improve == 8):
        break


df = pd.DataFrame(all_quality)
df.to_csv(log_path + str(args.log_file_name) + ".csv")


