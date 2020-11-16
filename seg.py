from sklearn.metrics import roc_curve, auc,precision_recall_curve, average_precision_score,r2_score,mean_squared_error,mean_absolute_error
from torchnet import meter
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch import optim 
from torch.utils import data 
import sys
import logging
import argparse
import datetime
from Model import MTLModel
import MTLDataset
from Loss import LossList
import numpy as np
from Loss.LossList import MultiLossLayer




# 配置分类损失函数
def get_criterion(criterion):
    if criterion == "CrossEntropy":
        return nn.CrossEntropyLoss()
# 配置回归损失函数
def get_criterionUS(criterionUS):
    if criterionUS == "XTanh":
        return LossList.XTanhLoss()
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
            


# def checkpoint(acc, epoch):
#     # Save checkpoint.
#     print('Saving..')
#     state = {
#         'net': net,
#         'acc': acc,
#         'epoch': epoch,
#         'rng_state': torch.get_rng_state()
#     }
#     if not os.path.isdir('checkpoint'):
#         os.mkdir('checkpoint')
#     torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
#                + str(args.seed))


# 调用mixup
    # 注意要控制Y作为类型结果时，要是独热编码，这里要做一些测试
    # 然后可能要需要加入mixup的criterion
def mixup_data(x, y1, y2, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y1_a, y1_b = y1, y1[index]
    y2_a, y2_b = y2, y2[index]
    return mixed_x, y1_a, y1_b, y2_a, y2_b, lam

def mixup_criterion_type(the_criterion, pred, y_a, y_b, lam):
    return lam * the_criterion(pred, y_a) + (1 - lam) * the_criterion(pred, y_b)

def mixup_criterion_numeric(the_criterion, pred, y_a, y_b, lam):
    return the_criterion(pred, lam * y_a + (1 - lam) * y_b)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, help="The Main Classifier", default='NddrCrossDense')
    parser.add_argument("--mode", type=str, help="Mode", default='NddrLSC')
    parser.add_argument("--optim", type=str, help="Optimizer", default='Adam')
    parser.add_argument("--criterion", type=str, help="criterion", default='CrossEntropy')
    parser.add_argument("--criterionUS", type=str, help="criterionUS", default='XTanh')
    parser.add_argument("--s_data_root", type=str, help="single data root", default='../ResearchData/UltraImageUSFullTest/UltraImageCropFull')
    parser.add_argument("--logging_level", type=int, help="logging level", default=20)
    parser.add_argument("--log_file_name", type=str, help="logging file name", default="./Log/" +str(datetime.date.today())+'.log')
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
    parser.add_argument("--logdir", type=str, help="Please input the tensorboard logdir.", default='testMix')
    parser.add_argument("--GPU", type=int, help="GPU ID", default=1)
    parser.add_argument("--alpha", type=int, help="If use mixup", default=1)
    # parser.add_argument("--pkl_name", type=str, help="pkl name", default=0)
    # parser.add_argument("--reg_lambda", type=float, help="L2 regularization lambda", default=1e-5)
    # parser.add_argument("--keep_prob", type=float, help="Dropout keep probability", default=0.8)
    # parser.add_argument("--cross_stitch_enabled", type=bool, help="Use Cross Stitch or not", default=True)

    return parser.parse_args(argv)


def train(epoch):
    # dataloader setting
    train_dataset = MTLDataset.MTLDataset(
        str(data_root)+'TRAIN/', us_path=us_path, num_classes=NUM_CLASSES, train_or_test = 'Train',screener=rf_sort_list,screen_num = 10)
  
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
   
    model.train()
    for i, (img, US_data, label4, label2) in enumerate(train_dataloader):
        #input
        img = img.to(DEVICE)
        #label
        US_data = US_data.to(DEVICE)
        US_label = US_data[:,:args.length_aux]
        label4 = label4.to(DEVICE)
        # mixup
        img, US_label_a, US_label_b, label4_a, label4_b, lam = mixup_data(img, US_label, label4, args.alpha)
       

      
        label2 = label4%3>0
        
        if args.mode is not None:
            out, out_US = model(img)
        else:
            out = model(img)
    

        # Show the result
        # train_loss = criterion(out, label4)
        train_loss = mixup_criterion_type(criterion, out, label4_a, label4_b, lam)
        # train_loss_US = criterionUS(out_US, US_label)
        train_loss_US = mixup_criterion_numeric(criterionUS, out_US, US_label_a, US_label_b, lam)
        train_loss_multi = multi_loss(train_loss, train_loss_US)
        train_loss_meter_multi.add(train_loss_multi.data.item())
        optimizer.zero_grad()
        train_loss_multi.backward()
        optimizer.step()



        train_loss_item = train_loss.data.item()
        batch_train_loss_meter.add(train_loss_item)
        train_loss_meter.add(train_loss_item)
        train_loss_item_US = train_loss_US.data.item()
        batch_train_loss_meter_US.add(train_loss_item_US)
        train_loss_meter_US.add(train_loss_item_US)
        train_loss_meter_multi.add(train_loss_multi.data.item())

        if i % NUM_TRAIN_CHECK_BATCHES == 0:
            logging.debug('Main {}:{}'.format(i * BATCH_SIZE, batch_train_loss_meter.value()))
            logging.debug('US {}:{}'.format(i * BATCH_SIZE, batch_train_loss_meter_US.value()))
            batch_train_loss_meter.reset()
            batch_train_loss_meter_US.reset()

    logging.info('Epoch Average Train Loss:{}'.format(train_loss_meter.value()))
    writer.add_scalar('loss/train', train_loss_meter.value()[0], epoch)
    train_loss_meter.reset()
    logging.info('Epoch Average Train US Loss:{}'.format(train_loss_meter_US.value()))
    writer.add_scalar('loss/train_US', train_loss_meter_US.value()[0], epoch)
    train_loss_meter_US.reset()
    logging.info('Epoch Average Train Multi Loss:{}'.format(train_loss_meter_multi.value()))
    writer.add_scalar('loss/train_multi', train_loss_meter_multi.value()[0], epoch)
    train_loss_meter_multi.reset()


def test(epoch):
    logging.info("Epoch " + str(epoch))
    test_dataset = MTLDataset.MTLDataset(
        str(data_root) + 'TEST/', us_path=us_path, num_classes=NUM_CLASSES, train_or_test='Test', screener=rf_sort_list, screen_num=10)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    test_loss = 0
    test_loss_US = 0
    test_loss_multi = 0
    correct = 0
    correct_aux = 0
    test_loss_meter.reset()
    test_loss_meter_US.reset()
    test_loss_meter_multi.reset()
    confusion_matrix.reset()
    confusion_matrix_aux.reset()

    with torch.no_grad():
        for i ,(img, US_data, label4, label2) in enumerate(test_dataloader):
            
             
            img = img.to(DEVICE)
            US_data = US_data.to(DEVICE)
            US_label = US_data[:,:args.length_aux]
            label4 = label4.to(DEVICE)
            label2 = label4%3>0

            output, output_US = model(img)
                            
            if(i == 0):
                epoch_output = output
                epoch_label4 = label4
                epoch_output_US = output_US
                epoch_US_label = US_label

            else:
                epoch_output = torch.cat((epoch_output, output))
                epoch_label4 = torch.cat((epoch_label4, label4))
                epoch_output_US = torch.cat((epoch_output_US,output_US))
                epoch_US_label = torch.cat((epoch_US_label,US_label))
                
            

            test_loss = criterion(output, label4)
            test_loss_US = criterionUS(output_US, US_label)
            test_loss_multi = multi_loss(test_loss, test_loss_US)
            

            test_loss_item = test_loss.data.item()
            test_loss_item_US = test_loss_US.data.item()
            test_loss_item_multi = test_loss_US.data.item()
            test_loss_meter.add(test_loss_item)
            test_loss_meter_US.add(test_loss_item_US)
            test_loss_meter_multi.add(test_loss_item_multi)
            pred = output.argmax(dim=1)
            pred_aux = pred%3>0

            confusion_matrix.add(pred, label4)
            confusion_matrix_aux.add(pred_aux,label2)
            correct += pred.eq(label4.view_as(pred)).sum().item()
            correct_aux += pred_aux.eq(label2.view_as(pred_aux)).sum().item()

        confusion = confusion_matrix.value()
        confusion_aux = confusion_matrix_aux.value()

        logging.info('Precision  Recall    F1Score    Specificity')
        for i in range(4):
            rowsum, colsum, diagonalsum = sum(confusion[i]), sum(confusion[r][i] for r in range(4)), sum(confusion[i][i] for i in range(4))

            precision = confusion[i][i] / float(colsum)
            recall = confusion[i][i] / float(rowsum)
            f1score = 2*precision * recall / (precision + recall)
            specificity = (diagonalsum - confusion[i][i]) / (diagonalsum + float(rowsum) - 2 * confusion[i][i])

            cpu_epoch_output = epoch_output.cpu()
            cpu_epoch_label4 = epoch_label4.cpu()

            fpr, tpr, _ = roc_curve(cpu_epoch_label4, cpu_epoch_output[:, i], pos_label=i)
            roc_auc = auc(fpr, tpr)
            logging.info('AUC{}:{:.4f}'.format(i, roc_auc))
            
            AP = average_precision_score(cpu_epoch_label4==i, cpu_epoch_output[:, i])
            logging.info('AP{}:{:.4f}'.format(i,AP))

            if (epoch >= EPOCH - 5):
                curve_precision, curve_recall, _ = precision_recall_curve(cpu_epoch_label4, cpu_epoch_output[:, i], pos_label=i)
                for j in range(len(fpr)):
                    writer.add_scalar('ROC/Epoch' + str(epoch) + '/test' + str(i), 100 * tpr[j], 100 * fpr[j])
                for j in range(len(curve_recall)):
                    writer.add_scalar('PRC/Epoch'+str(epoch)+'/test'+str(i), 100*curve_precision[j], 100*curve_recall[j])
            
            writer.add_scalar('AUC/test' + str(i), roc_auc, epoch)
            writer.add_scalar('AP/test' + str(i), AP, epoch)   
                
            writer.add_scalar('precision/test'+str(i), precision, epoch)
            writer.add_scalar('recall/test'+str(i), recall, epoch)
            writer.add_scalar('f1score/test'+str(i), f1score, epoch)
            writer.add_scalar('specificity/test'+str(i), specificity, epoch)
            logging.info('{:.4f}     {:.4f}     {:.4f}      {:.4f}'.format(precision,recall,f1score,specificity))


        logging.info('Precision_aux  Recall_aux    F1Score_aux    Specificity_aux')
        for i in range(2):
            rowsum_aux, colsum_aux, diagonalsum_aux = sum(confusion_aux[i]), sum(confusion_aux[r][i] for r in range(2)), sum(confusion_aux[i][i] for i in range(2))

            precision_aux = confusion_aux[i][i] / float(colsum_aux)
            recall_aux = confusion_aux[i][i] / float(rowsum_aux)
            f1score_aux = 2*precision_aux * recall_aux / (precision_aux + recall_aux)
            specificity_aux = (diagonalsum_aux - confusion_aux[i][i]) / (diagonalsum_aux + float(rowsum_aux) - 2 * confusion_aux[i][i])


            writer.add_scalar('precision/test_aux'+str(i), precision_aux, epoch)
            writer.add_scalar('recall/test_aux'+str(i), recall_aux, epoch)
            writer.add_scalar('f1score/test_aux'+str(i), f1score_aux, epoch)
            writer.add_scalar('specificity/test_aux'+str(i), specificity_aux, epoch)
            logging.info('{:.4f}     {:.4f}     {:.4f}      {:.4f}'.format(precision_aux,recall_aux,f1score_aux,specificity_aux))

    
        full_accuracy = 100. * correct / len(test_dataloader.dataset)
        logging.info('Confusion_Matrix:\n {}'.format(confusion))
        logging.info('Test set: Average loss: {:.4f}~{:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss_meter.value()[0], test_loss_meter.value()[1], correct, len(test_dataloader.dataset),
            full_accuracy))
        writer.add_scalar('loss/test', test_loss_meter.value()[0], epoch)
        writer.add_scalar('accu/test', full_accuracy, epoch)
        
        full_accuracy_aux = 100. * correct_aux / len(test_dataloader.dataset)
        logging.info('Confusion_Matrix_aux:\n {}'.format(confusion_aux))
        logging.info('Test set _US: Average loss: {:.4f}~{:.4f}\n'.format(
            test_loss_meter_US.value()[0], test_loss_meter_US.value()[1]))
        logging.info(' Accuracy_aux: {}/{} ({:.0f}%)\n'.format(
            correct_aux, len(test_dataloader.dataset),
            full_accuracy_aux))
        writer.add_scalar('loss/test_US', test_loss_meter_US.value()[0], epoch)
        writer.add_scalar('accu/test_aux', full_accuracy_aux, epoch)
        writer.add_scalar('loss/test_multi', test_loss_meter_multi.value()[0], epoch)

        cpu_epoch_US_label = epoch_US_label.cpu()
        cpu_epoch_output_US = epoch_output_US.cpu()
        US_r2 = r2_score(cpu_epoch_US_label, cpu_epoch_output_US)
        logging.info('US----R2_score:{}'.format(US_r2))
        writer.add_scalar('R2/test_US', US_r2, epoch)
        US_MSE = mean_squared_error(cpu_epoch_US_label, cpu_epoch_output_US)
        logging.info('US----MSE:{}'.format(US_MSE))
        writer.add_scalar('MSE/test_US', US_MSE, epoch)
        US_MAE = mean_absolute_error(cpu_epoch_US_label, cpu_epoch_output_US)
        logging.info('US----MAE:{}'.format(US_MAE))
        writer.add_scalar('MAE/test_US', US_MAE, epoch)
        cpu_epoch_US_label0 = cpu_epoch_US_label[:, 0]
        cpu_epoch_US_label1 = cpu_epoch_US_label[:, 1]
        cpu_epoch_output_US0 = cpu_epoch_output_US[:, 0]
        cpu_epoch_output_US1 = cpu_epoch_output_US[:, 1]
        US_r2_0 = r2_score(cpu_epoch_US_label0, cpu_epoch_output_US0)
        logging.info('US0----R2_score:{}'.format(US_r2_0))
        writer.add_scalar('R2/test_US0', US_r2_0, epoch)
        US_r2_1 = r2_score(cpu_epoch_US_label1, cpu_epoch_output_US1)
        logging.info('US1----R2_score:{}'.format(US_r2_0))
        writer.add_scalar('R2/test_US1', US_r2_1, epoch)
        US_MSE0 = mean_squared_error(cpu_epoch_US_label0, cpu_epoch_output_US0)
        logging.info('US----MSE0:{}'.format(US_MSE0))
        writer.add_scalar('MSE0/test_US', US_MSE0, epoch)
        US_MAE0 = mean_absolute_error(cpu_epoch_US_label0, cpu_epoch_output_US0)
        logging.info('US----MAE0:{}'.format(US_MAE0))
        writer.add_scalar('MAE0/test_US', US_MAE0, epoch)
        US_MSE1 = mean_squared_error(cpu_epoch_US_label1, cpu_epoch_output_US1)
        logging.info('US----MSE1:{}'.format(US_MSE1))
        writer.add_scalar('MSE1/test_US', US_MSE1, epoch)
        US_MAE1 = mean_absolute_error(cpu_epoch_US_label1, cpu_epoch_output_US1)
        logging.info('US----MAE1:{}'.format(US_MAE1))
        writer.add_scalar('MAE1/test_US', US_MAE1, epoch)


# Meter用于度量波动区间
batch_train_loss_meter = meter.AverageValueMeter()
batch_train_loss_meter_US = meter.AverageValueMeter()
train_loss_meter = meter.AverageValueMeter()
train_loss_meter_US = meter.AverageValueMeter()
train_loss_meter_multi = meter.AverageValueMeter()
test_loss_meter = meter.AverageValueMeter()
test_loss_meter_US = meter.AverageValueMeter()
test_loss_meter_multi = meter.AverageValueMeter()
confusion_matrix = meter.ConfusionMeter(4)
confusion_matrix_aux = meter.ConfusionMeter(2)

# 配置特征排序（和引用的特征量）
rf_sort_list = ['SizeOfPlaqueLong', 'SizeOfPlaqueShort', 'DegreeOfCASWtihDiameter', 'Age', 'PSVOfCCA', 'PSVOfICA', 'DiameterOfCCA', 'DiameterOfICA', 'EDVOfICA', 'EDVOfCCA', 'RIOfCCA', 'RIOfICA', 'IMT', 'IMTOfICA', 'IMTOfCCA', 'Positio0fPlaque', 'Sex', 'IfAnabrosis', 'X0Or0']
# 写入配置
args = parse_args(sys.argv[1:])
start_epoch = 1
DEVICE = torch.device('cuda:' + str(args.GPU) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.GPU)
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.wd
model = MTLModel.NddrDenseNet(mode=args.mode, memory_efficient=False, nddr_drop_rate=args.nddr_dr, length_aux= args.length_aux).to(DEVICE)
criterion = get_criterion(args.criterion)
criterionUS = get_criterionUS(args.criterionUS)
multi_loss = MultiLossLayer(2)
multi_loss.to(DEVICE)
optimizer = get_optimizer(args.optim, multi_loss)
data_root = args.s_data_root.strip()
NUM_CLASSES = 4
BATCH_SIZE = args.n_batch_size
NUM_TRAIN_CHECK_BATCHES = 4


EPOCH = args.epoch
MOMENTUM = args.momentum
us_path = '../ResearchData/data_ultrasound_1.csv'
writer = SummaryWriter('./'+str(args.logdir)+'/' + str(args.s_data_root[-10:]) +'/'+ str(args.net) +'_'+ str(args.mode) )
logging.basicConfig(level=args.logging_level,filename=args.log_file_name,
                    filemode='a', format='%(asctime)s   %(levelname)s   %(message)s')
logging.warning('Model: {}  Mode:{}'.format(args.net, args.mode))

best_acc = 0.75

for epoch in range(start_epoch, args.epoch):
    train(epoch)
    test(epoch)
    # 考虑精细调整LR
    # adjust_learning_rate(optimizer, epoch)

    