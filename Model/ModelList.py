#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import torchvision

from Model import UNet, SourceRaw
from Model.Backbone import Inception_BB, VGG_BB, ResNet_BB, DenseNet_BB, Dense_BB_ForUNet, \
    TwoInput_NDDRLSC_BB, NDDRLSC_BB_ForUNet, BestPractice2I
from Model.Classifier import FCNHead, DeepLabV3Head, UNet_Classifier, DeepLabHeadV3Plus
from Model.DeepMadSeg import DeepMadSeg
from Model.DenseAPP import DenseASPP
from Model.FCDenseNet import FCDenseNet
from Model.HyperDenseNet import HyperDenseNet_2Mod
from Model.IVDNet import IVD_Net_asym_2M
from Model.LinkNet import LinkNet
from Model.PointRend import PointRend, deeplabv3, PointHead
from Model.SegTron import SimpleSegmentationModel, DesicionFusion, TwoInputSegmentationModel
from Model.U2Net import U2NET, U2NETP, BASNet
from Model.UNet import autoencoder, UNet, UNet_2Plus, UNet_3Plus, ResNetUNet, DilatedUNet, kiunet, SegNet, DUCFCN
from Model.UNetNest import R2U_Net, AttU_Net, R2AttU_Net, NestedUNet




def get_model(model_name, in_channelsX=3, in_channelsY=3, n_class=1, ifPointRend=False):
    model_name = model_name.strip()
    # Pure Single Or InputLevel
    if model_name == 'FCN_ResNet50':
        return SourceRaw.SourceModel(in_channelsX, n_class, model_name='FCN_ResNet50')
    if model_name == 'FCN_ResNet101':
        return SourceRaw.SourceModel(in_channelsX, n_class, model_name='FCN_ResNet101')
    if model_name == 'DLV3__ResNet50':
        return SourceRaw.SourceModel(in_channelsX, n_class, model_name='DLV3__ResNet50')
    if model_name == 'DLV3__ResNet101':
        return SourceRaw.SourceModel(in_channelsX, n_class, model_name='DLV3__ResNet101')
    # 单个
    if model_name == 'SegNet':
        return SegNet(in_channelsX, n_class)
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
    if model_name == 'LinkNet':
        return LinkNet(in_channels=in_channelsX, n_class=n_class)
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
    if model_name == 'DUCFCN':
        return DUCFCN(in_channelsX, n_class)
    # layer fusion
    if model_name == 'HyperDenseNet':
        return HyperDenseNet_2Mod(n_class=n_class, mod0_channels=in_channelsX, mod1_channels=in_channelsY)
    if model_name == 'IVDNet':
        return IVD_Net_asym_2M(input0_nc=in_channelsX, input1_nc=in_channelsY, output_nc=n_class, ngf=32)
    if model_name == 'NDDR_DLV3P_PointRend':
        backbone = BestPractice2I(3, 3 )
        classifier = DeepLabHeadV3Plus(backbone.out_channels, backbone.fine_grained_channels, 2)
        return TwoInputSegmentationModel(backbone, classifier, if_point_rend_upsample=ifPointRend)

    raise NotImplementedError('model {} is not supported as of now'.format(model_name))









def get_composed_model(backbone_name, classifier_name, in_channelsX=3, in_channelsY=3, n_class=1):
    '''

    Args:
        backbone_name: find the backbone in the list
            SingleInput ['Inception_BB', 'VGG_BB', 'ResNet_BB', 'DenseNet_BB']
                For U-Net ['DUCFCN_BB_ForUNet', 'DenseForUNet']
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
        backbone = ResNet_BB(in_channelsX, version='resnet101')
    elif backbone_name == 'VGG':
        backbone = VGG_BB(in_channelsX, version='vgg19_bn')
    elif backbone_name == 'Inception':
        backbone = Inception_BB(in_channelsX, version='inception_v3')
    elif backbone_name == 'DenseNet':
        backbone = DenseNet_BB(in_channelsX, version='densenet161')

    elif backbone_name == 'DenseForUNet':
        backbone = Dense_BB_ForUNet(in_channelsX)

    elif backbone_name == 'TwoInput_NDDRLSC_BB':
        backbone = TwoInput_NDDRLSC_BB(in_channelsX, in_channelsY, num_classes=1)
    elif backbone_name == 'NDDRLSC_BB_ForUNet':
        backbone = NDDRLSC_BB_ForUNet(in_channelsX, in_channelsY, num_classes=n_class)

    else:
        raise NotImplementedError('backbone {} is not supported as of now'.format(backbone_name))

    if hasattr(backbone, 'out_channels'):
        backbone_out_channels = backbone.out_channels
    else:
        backbone_out_channels = None
    classifier = get_clasifier(classifier_name, backbone_out_channels=backbone_out_channels, n_class=n_class)
    return SimpleSegmentationModel(backbone, classifier)

def get_2i_composed_model(backbone_name, classifier_name, in_channelsX=3, in_channelsY=3, n_class=1):
    if backbone_name == 'TwoInput_NDDRLSC_BB':
        backbone = TwoInput_NDDRLSC_BB(in_channelsX, in_channelsY, num_classes=1)
    elif backbone_name == 'NDDRLSC_BB_ForUNet':
        backbone = NDDRLSC_BB_ForUNet(in_channelsX, in_channelsY, num_classes=n_class)
    else:
        raise NotImplementedError('backbone {} is not supported as of now'.format(backbone_name))
    if hasattr(backbone, 'out_channels'):
        backbone_out_channels = backbone.out_channels
    else:
        backbone_out_channels = None
    classifier = get_clasifier(classifier_name, backbone_out_channels=backbone_out_channels, n_class=n_class)
    return TwoInputSegmentationModel(backbone, classifier)

def get_decison_fusion_model(model0, model1, n_class=1, fusion_function='conv'):
    return DesicionFusion(model0, model1, num_class=n_class, fusion_function=fusion_function)

def get_clasifier(classifier_name, backbone_out_channels=None, n_class=1):
    if classifier_name == 'FCN':
        return FCNHead(backbone_out_channels, n_class)
    elif classifier_name == 'DeepLabV3':
        return DeepLabV3Head(backbone_out_channels, n_class)
    elif classifier_name == 'UNet_Classifier':
        # Maybe need to check the inchannels list
        return UNet_Classifier(num_classes=n_class)
    else:
        raise NotImplementedError('classifier {} is not supported as of now'.format(classifier_name))

def get_single_input_model(model_name, model_clf_name=None,composed=False):
    if composed:
        return get_composed_model(model_name, model_clf_name)
    return get_model(model_name, in_channelsX=3, in_channelsY=3, n_class=1)

def get_multi_input_model(modelX_name, modelX_clf_name=None, modelY_name=None, modelY_clf_name=None, in_channelsX=3, in_channelsY=3, input_fusion=False, decision_fusion=False, composedX=False, composedY=False, fusion_function='conv'):
    # input fusion
    if input_fusion:
        if composedX:
            return get_composed_model(modelX_name, modelX_clf_name, in_channelsX=6)
        return get_model(modelX_name, in_channelsX=6, in_channelsY=in_channelsY, n_class=1)
    # decision fusion
    if decision_fusion:
        if composedX:
            modelX = get_composed_model(modelX_name, modelX_clf_name, in_channelsX=3)
        else:
            modelX = get_model(modelX_name, in_channelsX=3, in_channelsY=in_channelsY, n_class=1)
        if composedY:
            modelY = get_composed_model(modelY_name, modelY_clf_name, in_channelsX=3)
        else:
            modelY = get_model(modelY_name, in_channelsX=3, in_channelsY=in_channelsY, n_class=1)
        return get_decison_fusion_model(modelX, modelY, fusion_function=fusion_function)

    # layer fusion
    if composedX:
        return get_composed_model(modelX_name, modelX_clf_name, in_channelsX=3, in_channelsY=3, n_class=1)
    return get_model(modelX_name, in_channelsX=3, in_channelsY=3, n_class=1)

