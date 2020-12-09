#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
import argparse
import datetime

# 参数解析


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, help="The Main Model", default='FCN_ResNet101') #'NDDR_DLV3P_PointRend'
    parser.add_argument("--net_input_num", type=int, help="Use 1I model or 2I model", default=1)
    parser.add_argument("--pretrained", type=bool, help="if pretrained", default=False)
    parser.add_argument("--mode", type=str, help="Mode", default='NddrLSC')
    parser.add_argument("--optim", type=str, help="Optimizer", default='Adam')
    parser.add_argument("--criterion", type=str, help="criterion", default='BCELoss')
    parser.add_argument("--criterionUS", type=str, help="criterionUS", default='XTanh')
    parser.add_argument("--s_data_root", type=str, help="single data root", default='../ResearchData/UltraImageUSFullTest/UltraImageCropFullResize')
    parser.add_argument("--seg_root", type=str, help="segmentation label root",
                        default='../BlurBinaryLabel/')
    parser.add_argument("--fluid_root", type=str, help="fluid data root",
                        default='../flowImage/')  # or '../BinaryFlowImage/'
    parser.add_argument("--binary_fluid", type=int, help="fluid data root",
                        default=0)  # or '../BinaryFlowImage/'
    parser.add_argument("--intersection", type=int, help="if intersection",
                        default=0)
    parser.add_argument("--logging_level", type=int, help="logging level", default=20)
    parser.add_argument("--log_file_name", type=str, help="logging file name", default=str(datetime.date.today())+'.log')
    # parser.add_argument("--length_US", type=int, help="Length of US_x", default=32)
    parser.add_argument("--length_aux", type=int, help="Length of y", default=10)
    parser.add_argument("--n_class", type=int, help="number of classes", default=1)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=0)
    parser.add_argument("--momentum", type=float, help="momentum", default=0.9)
    parser.add_argument("--nddr_dr", type=float, help="nddr drop rate", default = 0)
    parser.add_argument("--epoch", type=int, help="number of epoch", default=200)
    parser.add_argument("--n_batch_size", type=int, help="mini batch size", default=8)
    parser.add_argument("--n_tarin_check_batch", type=int, help="mini num of check batch", default=1)
    parser.add_argument("--save_best_model", type=int, help="if saving best model", default=0)
    parser.add_argument("--save_optim", type=int, help="if saving optim", default=0)
    parser.add_argument("--logdir", type=str, help="Please input the tensorboard logdir.", default=str(datetime.date.today()))
    parser.add_argument("--GPU", type=int, help="GPU ID", default=1)
    parser.add_argument("--alpha", type=int, help="If use mixup", default=0)
    parser.add_argument("--ifDataParallel", type=int, help="If use mixup", default=0)
    parser.add_argument("--ifPointRend", type=int, help="If pointrend", default=0)
    return parser.parse_args(argv)