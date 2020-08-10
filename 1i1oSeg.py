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

from collections import  Counter

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
    parser.add_argument("--GPU", type=int, help="GPU ID", default=1)
    parser.add_argument("--alpha", type=int, help="If use mixup", default=1)

    return parser.parse_args(argv)


def train(epoch):
    all_train_iter_loss = []
    train_dataset = MTLDataset.SegDataset(
        str(data_root)+'TRAIN/', args.seg_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test = 'Train',screener=rf_sort_list,screen_num = 10)
  
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()

    for i, (ID, img, seg_label, US_data, label4, label2) in enumerate(train_dataloader):
        img = img.to(DEVICE)
        # label
        seg_label = seg_label.to(DEVICE)
        US_data = US_data.to(DEVICE)
        US_label = US_data[:, :args.length_aux]
        label4 = label4.to(DEVICE)
        # mixup
        img, seg_label_a, seg_label_b, US_label_a, US_label_b, label4_a, label4_b, lam = mixup_data(img, seg_label, US_label, label4, args.alpha)

        out = model(img)['out']
        out = torch.sigmoid(out)

        



        train_loss = mixup_criterion_type(criterion, out, seg_label_a, seg_label_b, lam)
        train_loss_item = train_loss.data.item()
        train_loss_meter.add(train_loss_item)
        batch_train_loss_meter.add(train_loss_item)
        all_train_iter_loss.append(train_loss_item)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if i % NUM_TRAIN_CHECK_BATCHES == 0:
            logging.debug('Train {}:{}'.format(i * BATCH_SIZE, batch_train_loss_meter.value()))
            batch_train_loss_meter.reset()
            

        if i == 0:
            one = torch.ones_like(out)
            zero = torch.zeros_like(out)
            out = torch.where(out >= 0.5, one, out)
            out = torch.where(out < 0.5, zero, out).cpu()
            for j in range(BATCH_SIZE):
                save_img = transforms.ToPILImage()(out[j]).convert('L')
                path = '../Log/' + args.logdir + '/T' + str(j) + '/'
                if not os.path.exists(path):
                    os.makedirs(path)
                save_img.save(path + 'E' + str(epoch) + '_' + ID[j] + '.jpg')


            # vis.close()
            # vis.images(out_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction'))
            # vis.images(seg_label_np[:, None, :, :], win='train_label', opts=dict(title='label'))
            # vis.line(all_train_iter_loss, win='train_iter_loss', opts=dict(title='train iter loss'))

    logging.info('Epoch Average Train Loss:{}'.format(train_loss_meter.value()))
    writer.add_scalar('loss/train', train_loss_meter.value()[0], epoch)
    train_loss_meter.reset()

def test(epoch):
    all_test_iter_loss = []
    logging.info("Epoch " + str(epoch))
    test_dataset = MTLDataset.SegDataset(
        str(data_root) + 'TEST/', args.seg_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test='Test', screener=rf_sort_list,
        screen_num=10)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    test_loss_meter.reset()

    with torch.no_grad():
        for i, (ID, img, seg_label, US_data, label4, label2) in enumerate(test_dataloader):
            img = img.to(DEVICE)
            seg_label = seg_label.to(DEVICE)
            US_data = US_data.to(DEVICE)
            US_label = US_data[:, :args.length_aux]
            label4 = label4.to(DEVICE)

            # 输出
            output = model(img)['out']
            output = torch.sigmoid(output)
            # 记录Loss，计算性能指标
            loss = criterion(output, seg_label)
            iter_loss = loss.item()
            all_test_iter_loss.append(iter_loss)
            batch_test_loss_meter.add(iter_loss)
            test_loss_meter.add(iter_loss)

            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmin(output_np, axis=1)
            seg_label_np = seg_label.cpu().detach().numpy().copy()
            seg_label_np = np.argmin(seg_label_np, axis=1)

            # if (i == 0):
            #     epoch_output = output
            #     epoch_label4 = label4
            #     epoch_output_US = output_US
            #     epoch_US_label = US_label
            #
            # else:
            #     epoch_output = torch.cat((epoch_output, output))
            #     epoch_label4 = torch.cat((epoch_label4, label4))
            #     epoch_output_US = torch.cat((epoch_output_US, output_US))
            #     epoch_US_label = torch.cat((epoch_US_label, US_label))

            if i % NUM_TRAIN_CHECK_BATCHES == 0:
                logging.debug('Test {}:{}'.format(i * BATCH_SIZE, batch_test_loss_meter.value()))
                # vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction'))
                # vis.images(seg_label_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                # vis.line(all_test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss'))
                batch_test_loss_meter.reset()

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
model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=1, aux_loss=None).to(DEVICE)
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
# vis = visdom.Visdom()
writer = SummaryWriter('../Log/'+str(args.logdir)+'/' + str(args.s_data_root[-10:]) +'/'+ str(args.net) +'_'+ str(args.mode) )
logging.basicConfig(level=args.logging_level,filename=args.log_file_name,
                    filemode='a', format='%(asctime)s   %(levelname)s   %(message)s')
logging.warning('Model: {}  Mode:{}'.format(args.net, args.mode))

for epoch in range(start_epoch, args.epoch):
    train(epoch)
    test(epoch)