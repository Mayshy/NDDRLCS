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
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from torchvision.utils import save_image
from torchvision import transforms
import sys
import visdom
from torchnet import meter
import datetime
import metrics
from collections import  Counter


"""
单输入单输出分割

seg_label 标签 torch.Size([16, 2, 224, 224])
"""
# Q：灰度图是否归一化？？？
# Q: 未来要做p-value

# 配置分类损失函数
def get_criterion(criterion):
    if criterion == "CrossEntropy":
        return nn.CrossEntropyLoss()
    if criterion == "BCELoss":
        return nn.BCELoss()
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

# 参数解析
def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, help="The Main Classifier", default='NddrCrossDense')
    parser.add_argument("--mode", type=str, help="Mode", default='NddrLSC')
    parser.add_argument("--optim", type=str, help="Optimizer", default='Adam')
    parser.add_argument("--criterion", type=str, help="criterion", default='BCELoss')
    parser.add_argument("--criterionUS", type=str, help="criterionUS", default='XTanh')
    parser.add_argument("--s_data_root", type=str, help="single data root", default='../ResearchData/UltraImageUSFullTest/UltraImageCropFull')
    parser.add_argument("--seg_root", type=str, help="segmentation label root",
                        default='../seg/')
    parser.add_argument("--logging_level", type=int, help="logging level", default=20)
    parser.add_argument("--log_file_name", type=str, help="logging file name", default="../Log/" +str(datetime.date.today())+'.log')
    # parser.add_argument("--length_US", type=int, help="Length of US_x", default=32)
    parser.add_argument("--length_aux", type=int, help="Length of y", default=10)
    parser.add_argument("--n_class", type=int, help="number of classes", default=4)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.0001)
    parser.add_argument("--wd", type=float, help="weight decay", default=0.1)
    parser.add_argument("--momentum", type=float, help="momentum", default=0.1)
    parser.add_argument("--nddr_dr", type=float, help="nddr drop rate", default = 0)
    parser.add_argument("--epoch", type=int, help="number of epoch", default=900)
    parser.add_argument("--n_batch_size", type=int, help="mini batch size", default=16)
    parser.add_argument("--n_tarin_check_batch", type=int, help="mini num of check batch", default=1)
    parser.add_argument("--save_best_model", type=int, help="if saving best model", default=0)
    parser.add_argument("--save_optim", type=int, help="if saving optim", default=0)    
    parser.add_argument("--logdir", type=str, help="Please input the tensorboard logdir.", default='protoType')
    parser.add_argument("--GPU", type=int, help="GPU ID", default=0)
    parser.add_argument("--alpha", type=int, help="If use mixup", default=1)
    return parser.parse_args(argv)

# 训练
# TODO 得想个办法把海量的loss都放一起处理，显得优雅 : 输出形式 log + 逐行写入CSV
def train(epoch):
    all_train_iter_loss = []
    # 设置数据集
    train_dataset = MTLDataset.SegDataset(
        str(data_root)+'TRAIN/', args.seg_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test = 'Train',screener=rf_sort_list,screen_num = 10)
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
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
        img, seg_label_a, seg_label_b, US_label_a, US_label_b, label4_a, label4_b, lam = mixup_data(img, seg_label, US_label, label4, args.alpha)

        # 执行模型，得到输出
        out = model(img)['out']
        # 可能需要做sigmoid？
        out = torch.sigmoid(out)

        # 取损失函数
        train_loss = mixup_criterion_type(criterion, out, seg_label_a, seg_label_b, lam)
        train_loss_item = train_loss.data.item()
        train_loss_meter.add(train_loss_item)
        batch_train_loss_meter.add(train_loss_item)
        all_train_iter_loss.append(train_loss_item)

        # 使用优化器执行反向传播
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # TODO 输出每NUM_TRAIN_CHECK_BATCHES个batch的train_loss
        if i % NUM_TRAIN_CHECK_BATCHES == 0:
            logging.debug('Train {}:{}'.format(i * BATCH_SIZE, batch_train_loss_meter.value()))
            batch_train_loss_meter.reset()
        
            # for j in range(BATCH_SIZE):
            #     save_img = transforms.ToPILImage()(out[j]).convert('L')
            #     path = '../Log/' + args.logdir + '/T' + str(j) + '/'
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #     save_img.save(path + 'E' + str(epoch) + '_' + ID[j] + '.jpg')


    logging.info('Epoch Average Train Loss:{}'.format(train_loss_meter.value()))
    writer.add_scalar('loss/train', train_loss_meter.value()[0], epoch)
    train_loss_meter.reset()
    # 测试阶段要监控的指标很简单：loss，loss的单位有 单batch loss，多batch loss，总平均loss
    # 方案： 存为csv的是 单batch loss， 输出出来的是 多batch loss

def test(epoch):
    all_test_iter_loss = []
    logging.info("Epoch " + str(epoch))

    # 设置数据集
    test_dataset = MTLDataset.SegDataset(
        str(data_root) + 'TEST/', args.seg_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test='Test', screener=rf_sort_list,
        screen_num=10)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    test_loss_meter.reset()

    # 开始这轮的迭代
    with torch.no_grad():
        for i, (ID, img, seg_label, US_data, label4, label2) in enumerate(test_dataloader):
            # 数据分为两类， 算法的输入:img 算法的输出 seg_label ， （其他还没用到)
            img = img.to(DEVICE)
            seg_label = seg_label.to(DEVICE)
            US_data = US_data.to(DEVICE)
            US_label = US_data[:, :args.length_aux]
            label4 = label4.to(DEVICE)

            # 输出
            output = model(img)['out']
            # output = torch.sigmoid(output)
            # seg_label 标签
            # 

            # seg_test = torch.squeeze(seg_label, 1).long()
            seg_test = seg_label.long()
            

            # 记录Loss，计算性能指标
            # print("F1Raw" + str(metrics.classwise_f1(output, seg_test)))
            print("diceLoss " + str(metrics.dice_loss(output, seg_test)))
            print("iouReal? " + str(metrics.iou_score(output, seg_test)))
            print("sensi " + str(metrics.sensitivity(output, seg_test)))
            # print("iouRaw" + str(metrics.classwise_iou(output, seg_test)))
            


            loss = criterion(output, seg_label)
            iter_loss = loss.item()
            all_test_iter_loss.append(iter_loss)
            batch_test_loss_meter.add(iter_loss)
            test_loss_meter.add(iter_loss)

            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmin(output_np, axis=1)
            seg_label_np = seg_label.cpu().detach().numpy().copy()
            seg_label_np = np.argmin(seg_label_np, axis=1)

            if i % NUM_TRAIN_CHECK_BATCHES == 0:
                logging.debug('Test {}:{}'.format(i * BATCH_SIZE, batch_test_loss_meter.value()))
                batch_test_loss_meter.reset()

            # 目前每个epoch只输出第一个batch的图片
            # 利用ones_like和torch.where来
            if i == 0:
                one = torch.ones_like(output)
                zero = torch.zeros_like(output)
                output = torch.where(output >= 0.5, one, output)
                output = torch.where(output < 0.5, zero, output).cpu()
                for j in range(BATCH_SIZE):
                    save_img = transforms.ToPILImage()(output[j]).convert('L')
                    path = '../Log/' + args.logdir + '/V' + str(j) + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    save_img.save(path +'E'+ str(epoch) +'_' + ID[j] + '.jpg')
        logging.info('Epoch Average Test Loss:{}'.format(test_loss_meter.value()))
        writer.add_scalar('loss/test', test_loss_meter.value()[0], epoch)
        test_loss_meter.reset()


# 启动配置
# ------------------------------------------------------------------------------------------------------------------------------------------
# Meter用于度量波动区间
batch_train_loss_meter = meter.AverageValueMeter()
train_loss_meter = meter.AverageValueMeter()
batch_test_loss_meter = meter.AverageValueMeter()
test_loss_meter = meter.AverageValueMeter()

# 配置特征排序（和引用的特征量）
rf_sort_list = ['SizeOfPlaqueLong', 'SizeOfPlaqueShort', 'DegreeOfCASWtihDiameter', 'Age', 'PSVOfCCA', 'PSVOfICA', 'DiameterOfCCA', 'DiameterOfICA', 'EDVOfICA', 'EDVOfCCA', 'RIOfCCA', 'RIOfICA', 'IMT', 'IMTOfICA', 'IMTOfCCA', 'Positio0fPlaque', 'Sex', 'IfAnabrosis', 'X0Or0']
# 写入配置
args = parse_args(sys.argv[1:])
start_epoch = 1
DEVICE = torch.device('cuda:' + str(args.GPU) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.GPU)
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.wd
model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=2, aux_loss=None).to(DEVICE)
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

writer = SummaryWriter('../Log/'+str(args.logdir)+'/' + str(args.s_data_root[-10:]) +'/'+ str(args.net) +'_'+ str(args.mode) )
logging.basicConfig(level=args.logging_level,filename=args.log_file_name,
                    filemode='a', format='%(asctime)s   %(levelname)s   %(message)s')
logging.warning('Model: {}  Mode:{}'.format(args.net, args.mode))

## metrics
# jaccard_index = metrics.make_weighted_metric(metrics.classwise_iou)
# f1_score = metrics.make_weighted_metric(metrics.classwise_f1)

for epoch in range(start_epoch, args.epoch):
    train(epoch)
    test(epoch)