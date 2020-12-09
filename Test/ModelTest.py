#!/usr/bin/env python  
#-*- coding:utf-8 _*-  

import unittest

from Model.ModelList import get_composed_model, get_decison_fusion_model, get_model, get_2i_composed_model
from Model._utils import testModel, test2IModel

model_name_list = ['DUCFCN','SegNet', 'FCN_ResNet50', 'FCN_ResNet101', 'DLV3__ResNet50', 'DLV3__ResNet101', 'DeepMadSeg', 'DenseASPP', 'FCDenseNet103',
                   'PointRend', 'U2NET', 'U2NETP', 'BASNet', 'autoencoder', 'UNet', 'UNet_2Plus', 'UNet_3Plus', 'ResNetUNet',
                   'DilatedUNet', 'NestedUNet', 'R2AttU_Net', 'AttU_Net', 'kiunet']
twoI_model_name_list = ['HyperDenseNet', 'IVDNet']
test_backbone_name_list = ['VGG', 'ResNet', 'DenseNet']
test_classifier_name_list = ['FCN', 'DeepLabV3']
test_UNet_backbone_list = ['DenseForUNet']
test_UNet_classifier_list = ['UNet_Classifier']
test_2I_backcone_name_list = ['TwoInput_NDDRLSC_BB']
test_2IForUNet_backcone_name_list = ['NDDRLSC_BB_ForUNet']

class TestModels(unittest.TestCase):

    def test_get_model(self):
        for model_name in model_name_list:
            print(model_name + ':')
            model = get_model(model_name)
            testModel(model)


    def test_2I_get_model(self):
        for model_name in twoI_model_name_list:
            print(model_name + ':')
            model = get_model(model_name)
            test2IModel(model)
    def test_get_composed_model(self):
        for backbone_name in test_backbone_name_list:
            for classifier_name in test_classifier_name_list:
                print(backbone_name + '+' + classifier_name + ":")
                model = get_composed_model(backbone_name, classifier_name)
                testModel(model)
        for backbone_name in test_UNet_backbone_list:
            for classifier_name in test_UNet_classifier_list:
                print(backbone_name + '+' + classifier_name + ":")
                model = get_composed_model(backbone_name, classifier_name)
                testModel(model)

    def test_get_2I_composed_model(self):
        for backbone_name in test_2I_backcone_name_list:
            for classifier_name in test_classifier_name_list:
                print(backbone_name + '+' + classifier_name + ":")
                model = get_2i_composed_model(backbone_name, classifier_name)
                test2IModel(model)
        for backbone_name in test_2IForUNet_backcone_name_list:
            for classifier_name in test_UNet_classifier_list:
                print(backbone_name + '+' + classifier_name + ":")
                model = get_2i_composed_model(backbone_name, classifier_name)
                test2IModel(model)


    def test_get_decision_fusion_model(self):
        model0 = get_composed_model('ResNet', 'FCN')
        model1 = get_composed_model('VGG', 'FCN')
        model = get_decison_fusion_model(model0, model1)
        test2IModel(model)

if __name__ == '__main__':
    TestModels.main(verbosity=2)
    # TestModels.test_get_2I_composed_model()