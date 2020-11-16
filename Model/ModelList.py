#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import torchvision

from Model import InputLevelFusion, UNet
from Model.Backbone import Inception_BB, VGG_BB, ResNet_BB, DenseNet_BB, DUCFCN_BB_ForUNet, Dense_BB_ForUNet
from Model.Classifier import FCNHead, DeepLabV3Head, SegNet, UNet_Classifier
from Model.DeepMadSeg import DeepMadSeg
from Model.DenseAPP import DenseASPP
from Model.FCDenseNet import FCDenseNet
from Model.HyperDenseNet import HyperDenseNet_2Mod
from Model.IVDNet import IVD_Net_asym_2M
from Model.PointRend import PointRend, deeplabv3, PointHead
from Model.SegTron import SimpleSegmentationModel
from Model.U2Net import U2NET, U2NETP, BASNet
from Model.UNet import autoencoder, UNet, UNet_2Plus, UNet_3Plus, ResNetUNet, DilatedUNet, ZJUNet, kiunet
from Model.UNetNest import R2U_Net, AttU_Net, R2AttU_Net, NestedUNet
from Model._utils import testModel
from Model.resnest_ import ResNestBase




def get_model(model_name, in_channelsX=3, in_channelsY=3, n_class=1):
    model_name = model_name.strip()
    # Pure Single Or InputLevel
    if model_name == 'FCN_ResNet50':
        return torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=False, num_classes=1,
                                                            aux_loss=None)
    if model_name == 'FCN_ResNet101':
        return torchvision.models.segmentation.fcn_resnet101(pretrained=False, progress=False, num_classes=1,
                                                             aux_loss=None)
    if model_name == 'DLV3__ResNet50':
        return torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=False, num_classes=1,
                                                                  aux_loss=None)
    if model_name == 'DLV3__ResNet101':
        return torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False, num_classes=1,
                                                                   aux_loss=None)
    # 单个
    if model_name == 'DeepMadSeg':
        return DeepMadSeg(in_channelsX, n_class)
    if model_name == 'DenseASPP':
        return DenseASPP(in_channels=in_channelsX, n_class=n_class)
    if model_name == 'FCDenseNet103':
        return FCDenseNet(
        in_channels=in_channelsX, down_blocks=(4, 5, 7, 10, 12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_class)
    if model_name == 'PointRend':
        # res["coarse"] is the pred
        return PointRend(deeplabv3(False), PointHead())
    if model_name == 'U2NET':
        return U2NET(in_ch=in_channelsX, out_ch=n_class)
    if model_name == 'U2NETP':
        return U2NETP(in_ch=in_channelsX, out_ch=n_class)
    if model_name == 'BASNet':
    #     must match muti_bce_loss_fusion
        return BASNet(n_channels=in_channelsX, n_classes=n_class)
    if model_name == 'autoencoder':
        return autoencoder(in_channelsX, n_class)
    if model_name == 'UNet':
        return UNet(in_channelsX, n_class)
    if model_name == 'UNet_2Plus':
        return UNet_2Plus(in_channels=in_channelsX, n_classes=n_class)
    if model_name == 'UNet_3Plus':
        return UNet_3Plus(in_channels=in_channelsX, n_classes=n_class)
    if model_name == 'ResNetUNet':
        return ResNetUNet(in_channels=in_channelsX, n_class=n_class)
    if model_name == 'DilatedUNet':
        return DilatedUNet(n_channels=in_channelsX, n_classes=n_class)
    if model_name == 'kiunet':
        return kiunet(n_channels=in_channelsX, n_classes=n_class)
    if model_name == 'R2U_Net':
        return R2U_Net(img_ch=in_channelsX, output_ch=n_class)
    if model_name == 'AttU_Net':
        return AttU_Net(img_ch=in_channelsX, output_ch=n_class)
    if model_name == 'R2AttU_Net':
        return R2AttU_Net(in_ch=in_channelsX, out_ch=n_class)
    if model_name == 'NestedUNet':
        return NestedUNet(in_ch=in_channelsX, out_ch=n_class)


    # layer fusion
    if model_name == 'HyperDenseNet':
        return HyperDenseNet_2Mod(n_class=n_class, mod0_channels=in_channelsX, mod1_channels=in_channelsY)
    if model_name == 'IVDNet':
        return IVD_Net_asym_2M(input0_nc=in_channelsX, input1_nc=in_channelsY, output_nc=n_class, ngf=32)



    else:
        raise NotImplementedError('model {} is not supported as of now'.format(model_name))









def get_composed_model(backbone_name, classifier_name, in_channels=3, n_class=1):
    # in_channelsX = 3, in_channelsY = 3,
    '''

    Args:
        backbone_name: find the backbone in the list
            SingleInput ['Inception_BB', 'VGG_BB', 'ResNet_BB', 'DenseNet_BB']
                For U-Net ['DUCFCN_BB_ForUNet', 'Dense_BB_ForUNet']
            MultiInput ['TwoInput_NDDRLSC_BB', ]
                For U-Net ['NDDRLSC_BB_ForUNet']
        classifier_name: find the classifier in the list:
            SingleInput ['FCNHead', 'SegNet', 'DeepLabV3Head']
                U-Net ['UNet_Classifier']


    Returns:
        return a model composed of backbone and corresponding classifier.

    '''
    backbone_name = backbone_name.strip()
    classifier_name = classifier_name.strip()
    if backbone_name == 'ResNet':
        backbone = ResNet_BB(in_channels, version='resnet101')
    elif backbone_name == 'VGG':
        backbone = VGG_BB(in_channels, version='vgg19_bn')
    elif backbone_name == 'Inception':
        backbone = Inception_BB(in_channels, version='inception_v3')
    elif backbone_name == 'DenseNet':
        backbone = DenseNet_BB(in_channels, version='densenet161')
    elif backbone_name == 'DUCFCN':
        backbone = DUCFCN_BB_ForUNet(in_channels, n_class)
    elif backbone_name == 'DenseForUNet':
        backbone = Dense_BB_ForUNet(in_channels)

    elif backbone_name == 'TwoInput_NDDRLSC_BB':
        backbone = TwoInput_NDDRLSC_BB(in_channels)

    else:
        raise NotImplementedError('backbone {} is not supported as of now'.format(backbone_name))

    if classifier_name == 'FCN':
        classifier = FCNHead(backbone.out_channels, n_class)
    elif classifier_name == 'SegNet':
        classifier = SegNet(backbone.out_channels, n_class)
    elif classifier_name == 'DeepLabV3':
        classifier = DeepLabV3Head(backbone.out_channels, n_class)
    elif classifier_name == 'UNet_Classifier':
        # Maybe need to check the inchannels list
        classifier = UNet_Classifier(num_classes=n_class)
    else:
        raise NotImplementedError('classifier {} is not supported as of now'.format(classifier_name))

    return SimpleSegmentationModel(backbone, classifier)


def get_decison_fusion_model(model0_name, model1_name, in_channelsX=3, in_channelsY=3, n_class=1)

if __name__ == '__main__':
    testModel()