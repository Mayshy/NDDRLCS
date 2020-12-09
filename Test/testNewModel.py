#!/usr/bin/env python  
#-*- coding:utf-8 _*-
from Model.Backbone import BestPractice2I, BestPractice1I
from Model.Classifier import DeepLabHeadV3Plus
from Model.SegTron import TwoInputSegmentationModel, SimpleSegmentationModel
from Model._utils import test2IModel, testModel, testBackward
import warnings

warnings.filterwarnings('ignore')

backbone = BestPractice2I(3, 3, )
classifier = DeepLabHeadV3Plus(backbone.out_channels, backbone.fine_grained_channels, 2)
modelA = TwoInputSegmentationModel(backbone, classifier, if_point_rend_upsample=True)
test2IModel(modelA, eval=True)

backbone = BestPractice1I(3)
classifier = DeepLabHeadV3Plus(backbone.out_channels, backbone.fine_grained_channels, 2)
modelB = SimpleSegmentationModel(backbone, classifier, if_point_rend_upsample=True)
# testModel(modelB, eval=True)
testBackward(modelB)