# python的可读性真是太差了，这里约束一下：
# 每个函数前
import os
import pandas as pd
import MTLDataset
import logging
import torch
from torch import nn
import torch.nn.functional as F

from Loss.LossList import get_criterion
from Loss.OptimList import get_optimizer
from torch.utils import data
from torchvision import transforms
import sys
import metrics
import collections

from Model.ModelList import get_model
from Model._utils import adjust_learning_rate, setup_seed, log_mean, dict_sum, extractDict
from Model.mixup import mixup_criterion_type, mixup_data
from parse_args import parse_args


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
                                                                                                    device=DEVICE, alpha=args.alpha)
        # 执行模型，得到输出
        out = model(img)
        out = extractDict(out)


        # 取损失函数
        train_loss = mixup_criterion_type(criterion, out, seg_label_a, seg_label_b, lam)
        # train_loss = criterion(out, seg_label)

        train_loss_list.append(train_loss.item())

        # 使用优化器执行反向传播
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    train_loss_mean = log_mean(epoch, train_loss_list, "train loss", isLog= True)
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
            seg_label = seg_label.long().to(DEVICE)
            if args.criterion.strip() == 'BCELoss':
                seg_label = seg_label.float()
            US_data = US_data.to(DEVICE)
            US_label = US_data[:, :args.length_aux]
            label4 = label4.to(DEVICE)

            # 输出
            output = model(img)
            output = extractDict(output)
            output = nn.Sigmoid()(output)

            # seg_label 标签
            if args.criterion in ['GDL', 'GWDL']:
                output[output >= 0.5] = 0
                output[output < 0.5] = 1
            else:
                output[output >= 0.5] = 1
                output[output < 0.5] = 0
            if output.shape[3] != 512:
                output = F.interpolate(output, size=512, mode='bilinear', align_corners=True)
            seg_test = seg_label.long()



            # 记录Loss，计算性能指标
            # logging.info("Epoch {0} TestLoss {1}".format(epoch, loss.item()))
            loss = criterion(output, seg_label)
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

    test_loss_mean = log_mean(epoch, test_loss_list, "test loss", isLog= True)
    dice2_mean = log_mean(epoch, dice2_list, "dice2", isLog= True)
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
if args.ifDataParallel:
    model = torch.nn.DataParallel(get_model(args.net), device_ids=[0, 1]).cuda()
else:
    model = get_model(args.net).to(DEVICE)
criterion = get_criterion(args.criterion).to(DEVICE)

optimizer = get_optimizer(args.optim, model, LEARNING_RATE=LEARNING_RATE, MOMENTUM=MOMENTUM, WEIGHT_DECAY=WEIGHT_DECAY)
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
    # train(epoch)
    # test(epoch)
    adjust_learning_rate(optimizer, epoch, args.lr, args.epoch)
    cur_train_loss = train(epoch)
    if (cur_train_loss >= last_train_loss):
        count_loss_improve += 1
    else:
        count_loss_improve = 0
    last_train_loss = cur_train_loss
    test(epoch)
    if (count_loss_improve == 4):
        break


df = pd.DataFrame(all_quality)
df.to_csv(log_path + str(args.log_file_name) + ".csv")


