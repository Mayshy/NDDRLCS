
import math
import torchvision.models
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import torch
from torch.hub import load_state_dict_from_url
import re

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

class MISOModel():
    def __init__(self, use_model, NUM_CLASSES = 4, mode=None, length_aux = 32, length_y = 10):
        if_pretrained = False
        if (use_model == 'alexnet'):
            if not mode:
                model = torchvision.models.alexnet(
                    pretrained=if_pretrained)
                model.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 1000),
                    nn.Linear(1000, NUM_CLASSES),
                )
            else:
                raise('Undone')

        elif (use_model == 'resnet152'):
            if not mode:
                model = torchvision.models.resnet152(
                    pretrained=if_pretrained)
                fc_features = model.fc.in_features
                # my_model.fc = nn.Linear(fc_features, NUM_CLASSES)
                model.fc = nn.Sequential(nn.Linear(fc_features, 1000), nn.ReLU(
                    inplace=True), nn.Linear(1000, NUM_CLASSES))
            else:
                raise('Undone')
                model = resnet152(pretrained=if_pretrained, num_classes=NUM_CLASSES, length_aux = length_aux , length_y = length_y, mode = mode)


        elif (use_model == 'inception_v3'):
            # Output has aux, bug
            if not mode:
                model = torchvision.models.inception_v3(
                    pretrained=if_pretrained)
                fc_features = model.fc.in_features
                # my_model.fc = nn.Linear(fc_features, NUM_CLASSES)
                model.fc = nn.Sequential(nn.Linear(fc_features, 1000), nn.ReLU(
                    inplace=True), nn.Linear(1000, NUM_CLASSES))
            else:
                raise Exception('Undone')
                # model = inception_v3(pretrained=if_pretrained, num_classes=NUM_CLASSES, length_aux = length_aux , length_y = length_y, mode = mode)
                
        elif (use_model == 'vgg19_bn'):
            if not mode:
                model = torchvision.models.vgg19_bn(
                    pretrained=if_pretrained, num_classes=NUM_CLASSES)
                
            else:
                raise Exception('Undone')

        elif (use_model == 'resnext101_32x8d'):
            if not mode:
                model = torchvision.models.resnext101_32x8d(
                    pretrained=if_pretrained)
                fc_features = model.fc.in_features
                # my_model.fc = nn.Linear(fc_features, NUM_CLASSES)
                model.fc = nn.Sequential(nn.Linear(fc_features, 1000), nn.ReLU(
                    inplace=True), nn.Linear(1000, NUM_CLASSES))
            else:
                raise('Undone')
                model = resnext101_32x8d(pretrained=if_pretrained, num_classes=NUM_CLASSES, length_aux = length_aux , length_y = length_y, mode = mode)
        elif (use_model == 'densenet121'):
            if not mode:
                model = torchvision.models.densenet121(
                    pretrained=if_pretrained)
                fc_features = model.classifier.in_features
                model.classifier = nn.Sequential(nn.Linear(fc_features, 1000), nn.ReLU(
                    inplace=True), nn.Linear(1000, NUM_CLASSES))
            else:
                 model = densenet121(pretrained=if_pretrained, num_classes=NUM_CLASSES, length_aux = length_aux , length_y = length_y, mode = mode)
        elif (use_model == 'densenet169'):
            if not mode:
                model = torchvision.models.densenet169(
                    pretrained=if_pretrained)
                fc_features = model.classifier.in_features
                model.classifier = nn.Sequential(nn.Linear(fc_features, 1000), nn.ReLU(
                    inplace=True), nn.Linear(1000, NUM_CLASSES))
            else:
                 model = densenet169(pretrained=if_pretrained, num_classes=NUM_CLASSES, length_aux=length_aux, length_y=length_y, mode=mode)
        elif (use_model == 'densenet201'):
            if not mode:
                model = torchvision.models.densenet201(
                    pretrained=if_pretrained)
                fc_features = model.classifier.in_features
                model.classifier = nn.Sequential(nn.Linear(fc_features, 1000), nn.ReLU(
                    inplace=True), nn.Linear(1000, NUM_CLASSES))
            else:
                 model = densenet201(pretrained=if_pretrained, num_classes=NUM_CLASSES, length_aux=length_aux, length_y=length_y, mode=mode)
        elif (use_model == 'densenet161'):
            if not mode:
                model = torchvision.models.densenet161(
                    pretrained=if_pretrained)
                fc_features = model.classifier.in_features
                model.classifier = nn.Sequential(nn.Linear(fc_features, 1000), nn.ReLU(
                    inplace=True), nn.Linear(1000, NUM_CLASSES))
            else:
                 model = densenet161(pretrained=if_pretrained, num_classes=NUM_CLASSES, length_aux=length_aux, length_y=length_y, mode=mode)
           
        elif (use_model == 'wide_resnet101_2'):
            if not mode:
                model = torchvision.models.wide_resnet101_2(
                    pretrained=if_pretrained)
                fc_features = model.fc.in_features
                model.fc = nn.Sequential(nn.Linear(fc_features, 1000), nn.ReLU(
                    inplace=True), nn.Linear(1000, NUM_CLASSES))
            else:
                raise('Undone')
        elif (use_model == 'magic_densenet161'):
            if not mode:
                model = magic_densenet161(pretrained=if_pretrained, num_classes=4, length_aux = length_aux , length_y = length_y, mode = mode)
                # raise Exception('Please check the model and mode parameters.')
            else:
                 model = magic_densenet161(pretrained=if_pretrained, num_classes=4, length_aux = length_aux , length_y = length_y, mode = mode)
        else:
            raise Exception("No error!")
        self.model = model

# _________________________________________________________________________________________
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

        
        


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

# 
class _DenseBlock(nn.Module):
    _version = 2
    __constants__ = ['layers']

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleDict()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.layers['denselayer%d' % (i + 1)] = layer

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.layers.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

    @torch.jit.ignore
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if (version is None or version < 2):
            # now we have a new nesting level for torchscript support
            for new_key in self.state_dict().keys():
                # remove prefix "layers."
                old_key = new_key[len("layers."):]
                old_key = prefix + old_key
                new_key = prefix + new_key
                if old_key in state_dict:
                    value = state_dict[old_key]
                    del state_dict[old_key]
                    state_dict[new_key] = value
        super(_DenseBlock, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

# BN，RELU，改变channel，全局平均池化
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    __constants__ = ['features']
    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=4, drop_rate=0, num_classes=1000, length_aux = 32 , length_y = 10, mode = None, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y ):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _load_state_dict(model, model_url, progress):
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained=False, progress=True, **kwargs):
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


def densenet161(pretrained=False, progress=True, **kwargs):
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)


def densenet169(pretrained=False, progress=True, **kwargs):
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)


def densenet201(pretrained=False, progress=True, **kwargs):
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)





# SIDC
class MagicDenseNet(nn.Module):
    __constants__ = ['features']
    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=4, drop_rate=0, num_classes=1000, length_aux = 32 , length_y = 10, mode = None, memory_efficient=False):

        super(MagicDenseNet, self).__init__()

        # First convolution
        # 这里将feature_map的channel从3个转为96个
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
  

  

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            # block_aux = _DenseBlock(
            #     num_layers=num_layers,
            #     num_input_features=num_features,
            #     bn_size=bn_size,
            #     growth_rate=growth_rate,
            #     drop_rate=drop_rate,
            #     memory_efficient=memory_efficient
            # )
            if (i == 0):
                self.block0 = block
                # self.block0_aux = block_aux
            elif (i == 1):
                self.block1 = block
                # self.block1_aux = block_aux
            elif (i == 2):
                self.block2 = block
                # self.block2_aux = block_aux
            elif (i == 3):
                self.block3 = block
                # self.block3_aux = block_aux
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                # trans_aux = _Transition(num_input_features=num_features,   
                #                     num_output_features=num_features // 2) 
                if (i == 0):
                    self.transition0 = trans
                    # self.transition0_aux = trans_aux
                    self.ls0_reshape = nn.Linear(num_features // 2, 2208)
                elif (i == 1):
                    self.transition1 = trans
                    # self.transition1_aux = trans_aux
                    self.ls1_reshape = nn.Linear(num_features // 2, 2208)
                elif (i == 2):
                    self.transition2 = trans
                    # self.transition2_aux = trans_aux
                    self.ls2_reshape = nn.Linear(num_features // 2, 2208)
                num_features = num_features // 2

        # Final batch norm
        self.final_bn = nn.BatchNorm2d(num_features)
        # self.final_bn_aux = nn.BatchNorm2d(num_features)

        # Linear layer
        self.fc = nn.Linear(num_features,64)
        # self.fc_aux =  nn.Linear(num_features,64)
        self.classifier = nn.Linear(64, 4) #TODO:Finetune
        # self.classifier_aux = nn.Linear(64, 2)
        self.f_classifier = nn.Linear(num_features, 4)
        # self.f_classifier_aux = nn.Linear(num_features, 2)
        # Official init from torch repo.
        


        self.cross3 = nn.Linear(num_features*2,num_features*2,bias=False)
        self.cross4 = nn.Linear(128, 128, bias=False)
        self.cross5 = nn.Linear(6, 6, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.betas_4layer = nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                stdv = 1./ math.sqrt(m.weight.size(1))
                nn.init.uniform_(m.weight,-stdv,stdv)
                # nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        
        self.mode = mode 
    def forward(self, x):
        if not self.mode:
#             0 torch.Size([16, 192, 28, 28])
# 1 torch.Size([16, 384, 14, 14]) 
# 2 torch.Size([16, 1056, 7, 7])  10G
# 3 torch.Size([16, 2208, 7, 7])  40G
# out0 torch.Size([16, 2208, 1, 1]) 18M
# out1 torch.Size([16, 2208])
            features = self.features(x)
            block0 = self.block0(features)
            transition0 = self.transition0(block0)
            print('0',transition0.shape)

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            print('1',transition1.shape)

            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)
            print('2',transition2.shape)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            print('3', transition3.shape)
            
            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            print('out0', out.shape)
            out = torch.flatten(out, 1)
            print('out1', out.shape)
            out = self.fc(out)
            out = F.relu(out, inplace=True)
            out = self.classifier(out)
            return out
        elif (self.mode == 'layer_stitch'):
            features = self.features(x)
            block0 = self.block0(features)
            transition0 = self.transition0(block0)

            ls0 = F.relu(transition0, inplace=True)
            ls0 = F.adaptive_avg_pool2d(ls0, (1, 1))
            ls0 = torch.flatten(ls0, 1)
            ls0 = self.ls0_reshape(ls0)


            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)

            ls1 = F.relu(transition1, inplace=True)
            ls1 = F.adaptive_avg_pool2d(ls1, (1, 1))
            ls1 = torch.flatten(ls1, 1)
            ls1 = self.ls1_reshape(ls1)


            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)

            ls2 = F.relu(transition2, inplace=True)
            ls2 = F.adaptive_avg_pool2d(ls2, (1, 1))
            ls2 = torch.flatten(ls2, 1)
            ls2 = self.ls2_reshape(ls2)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            
            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)

            out = torch.mul(ls0,self.betas_4layer[0]) + torch.mul(ls1,self.betas_4layer[1]) + torch.mul(ls2,self.betas_4layer[2]) +  torch.mul(out,self.betas_4layer[3])

            # out = self.fc(out)
            # out = F.relu(out, inplace=True)
            # out = self.classifier(out)
            out = self.f_classifier(out)
            return out
        elif (self.mode == 'SIDCCross34'):
            features = self.features(x)
            block0 = self.block0(features)
            transition0 = self.transition0(block0)

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block1_aux = self.block1_aux(transition0)
            transition1_aux = self.transition1_aux(block1_aux)

            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)
            block2_aux = self.block2_aux(transition1_aux)
            transition2_aux = self.transition2_aux(block2_aux)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            block3_aux = self.block3_aux(transition2_aux)
            transition3_aux = self.final_bn_aux(block3_aux)
            # print('$transition3:', transition3.shape)
            # print('$transition3_aux:',transition3_aux.shape) #TODO:这里用于测试
            
            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)

            out, out_aux = apply_cross(self.cross3, out, out_aux)
            out,out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)

            out = self.fc(out)
            out_aux = self.fc_aux(out_aux)
            out,out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)


            out,out_aux = apply_cross(self.cross4,out,out_aux)
            out,out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)

            out = self.classifier(out)
            out_aux = self.classifier_aux(out_aux)

            # out, out_aux = apply_cross(self.cross5,out,out_aux)

            return out, out_aux
        elif (self.mode == 'SIDCCross345'):
            features = self.features(x)
            block0 = self.block0(features)
            transition0 = self.transition0(block0)

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block1_aux = self.block1_aux(transition0)
            transition1_aux = self.transition1_aux(block1_aux)

            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)
            block2_aux = self.block2_aux(transition1_aux)
            transition2_aux = self.transition2_aux(block2_aux)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            block3_aux = self.block3_aux(transition2_aux)
            transition3_aux = self.final_bn_aux(block3_aux)
           
            
            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)

            out, out_aux = apply_cross(self.cross3, out, out_aux)
            out,out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)
            

            out = self.fc(out)
            out_aux = self.fc_aux(out_aux)
            out,out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)


            out,out_aux = apply_cross(self.cross4,out,out_aux)
            out,out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)


            out = self.classifier(out)
            out_aux = self.classifier_aux(out_aux)
            out, out_aux = self.sigmoid(out),self.sigmoid(out_aux)


            out,out_aux = apply_cross(self.cross5,out,out_aux)

            # out, out_aux = apply_cross(self.cross5,out,out_aux)

            return out, out_aux
        elif (self.mode == 'SIDCCross35'):
            features = self.features(x)
            block0 = self.block0(features)
            transition0 = self.transition0(block0)

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block1_aux = self.block1_aux(transition0)
            transition1_aux = self.transition1_aux(block1_aux)

            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)
            block2_aux = self.block2_aux(transition1_aux)
            transition2_aux = self.transition2_aux(block2_aux)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            block3_aux = self.block3_aux(transition2_aux)
            transition3_aux = self.final_bn_aux(block3_aux)
            # print('$transition3:', transition3.shape)
            # print('$transition3_aux:',transition3_aux.shape) #TODO:这里用于测试
            
            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)

            out, out_aux = apply_cross(self.cross3, out, out_aux)
            out,out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)

            out = self.f_classifier(out)
            out_aux = self.f_classifier_aux(out_aux)
            out, out_aux = self.sigmoid(out),self.sigmoid(out_aux)
            
            out,out_aux = apply_cross(self.cross5,out,out_aux)

            # out, out_aux = apply_cross(self.cross5,out,out_aux)

            return out, out_aux
        elif (self.mode == 'SIDCPure'):
            features = self.features(x)
            block0 = self.block0(features)
            transition0 = self.transition0(block0)

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block1_aux = self.block1_aux(transition0)
            transition1_aux = self.transition1_aux(block1_aux)

            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)
            block2_aux = self.block2_aux(transition1_aux)
            transition2_aux = self.transition2_aux(block2_aux)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            block3_aux = self.block3_aux(transition2_aux)
            transition3_aux = self.final_bn_aux(block3_aux)
            
            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = self.f_classifier(out)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)
            out_aux = self.f_classifier_aux(out_aux)                  

            return out, out_aux

        else:
            raise Exception("No mode matched! Please check your cmd parameters.")



def magic_densenet161(pretrained=False, progress=True, **kwargs):
    return _magic_densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)

def _magic_densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = MagicDenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model



# SIDC
class shyDenseNet(nn.Module):
    __constants__ = ['features']
    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=4, drop_rate=0, num_classes=1000, length_aux = 32 , length_y = 10, mode = None, memory_efficient=False):

        super(shyDenseNet, self).__init__()

        # First convolution
        # 这里将feature_map的channel从3个转为96个
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
  

  

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))


        # Linear layer
        self.fc = nn.Linear(num_features,64)
        self.fc_aux =  nn.Linear(num_features,64)
        self.classifier = nn.Linear(64, 4) #TODO:Finetune
        self.classifier_aux = nn.Linear(64, 2)
        self.f_classifier = nn.Linear(num_features, 4)
        self.f_classifier_aux = nn.Linear(num_features, 2)
        # Official init from torch repo.
        
        self.cross3 = nn.Linear(num_features*2,num_features*2,bias=False)
        self.cross4 = nn.Linear(128, 128, bias=False)
        self.cross5 = nn.Linear(6, 6, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # stdv = 1./ math.sqrt(m.weight.size(1))
                # nn.init.uniform_(m.weight,-stdv,stdv)
                # nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        
        self.mode = mode 
    def forward(self, x):
        if not self.mode:

            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = self.fc(out)
            out = F.relu(out, inplace=True)
            out = self.classifier(out)
            return out
        elif (self.mode == 'SIDCCross4'):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = out

            # out,out_aux = apply_cross(self.cross3,out,out_aux)
            # out,out_aux = F.relu(out, inplace=True),F.relu(out_aux, inplace=True)

            out = self.fc(out)
            out_aux = self.fc_aux(out_aux)
            out, out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)
            
            out,out_aux = apply_cross(self.cross4,out,out_aux)
            out, out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)
            
            out = self.classifier(out)
            out_aux = self.classifier_aux(out_aux)

            # out, out_aux = apply_cross(self.cross5,out,out_aux)

            return out, out_aux
        elif (self.mode == 'SIDCCross45'):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = out

            # out,out_aux = apply_cross(self.cross3,out,out_aux)
            # out,out_aux = F.relu(out, inplace=True),F.relu(out_aux, inplace=True)


            out = self.fc(out)
            out_aux = self.fc_aux(out_aux)
            out,out_aux = F.relu(out, inplace=True),F.relu(out_aux, inplace=True)


            out,out_aux = apply_cross(self.cross4,out,out_aux)
            out,out_aux = F.relu(out, inplace=True),F.relu(out_aux, inplace=True)


            out = self.classifier(out)
            out_aux = self.classifier_aux(out_aux)
            out, out_aux = self.sigmoid(out),self.sigmoid(out_aux)


            out,out_aux = apply_cross(self.cross5,out,out_aux)

            # out, out_aux = apply_cross(self.cross5,out,out_aux)

            return out, out_aux
       
        elif (self.mode == 'SIDCPure'):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = out

            out = self.fc(out)
            out = F.relu(out, inplace=True)
            out = self.classifier(out)
            out_aux = self.fc(out_aux)
            out_aux = F.relu(out_aux, inplace=True)
            out_aux = self.classifier(out_aux)
                          

            return out, out_aux

        else:
            raise Exception("No mode matched! Please check your cmd parameters.")



def shy_densenet161(pretrained=False, progress=True, **kwargs):
    return _shy_densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)

def _shy_densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = shyDenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model





class cleanDenseNet(nn.Module):
    __constants__ = ['features']
    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=4, drop_rate=0, num_classes=1000, length_aux = 32 , length_y = 10, mode = None, memory_efficient=False):

        super(cleanDenseNet, self).__init__()

        # First convolution
        # 这里将feature_map的channel从3个转为96个
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
  

  

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            block_aux = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            if (i == 0):
                self.block0 = block
            elif (i == 1):
                self.block1 = block
            elif (i == 2):
                self.block2 = block
            elif (i == 3):
                self.block3 = block
                self.block3_aux = block_aux
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                if (i == 0):
                    self.transition0 = trans
                elif (i == 1):
                    self.transition1 = trans
                elif (i == 2):
                    self.transition2 = trans
                    # self.transition2_aux = trans_aux
                num_features = num_features // 2

        # Final batch norm
        self.final_bn = nn.BatchNorm2d(num_features)
        self.final_bn_aux = nn.BatchNorm2d(num_features)

        # Linear layer
        self.fc = nn.Linear(num_features,64)
        self.fc_aux =  nn.Linear(num_features,64)
        self.classifier = nn.Linear(64, 4) #TODO:Finetune
        self.classifier_aux = nn.Linear(64, 2)
        self.f_classifier = nn.Linear(num_features, 4)
        self.f_classifier_aux = nn.Linear(num_features, 2)
        # Official init from torch repo.
        
        self.cross3 = nn.Linear(num_features*2,num_features*2,bias=False)
        self.cross4 = nn.Linear(128, 128, bias=False)
        self.cross5 = nn.Linear(6, 6, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # stdv = 1./ math.sqrt(m.weight.size(1))
                # nn.init.uniform_(m.weight,-stdv,stdv)
                # nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        
        self.mode = mode 
    def forward(self, x):
        if not self.mode:

            features = self.features(x)
            block0 = self.block0(features)
            transition0 = self.transition0(block0)
            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            
            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = self.fc(out)
            out = F.relu(out,replace=True)
            out = self.classifier(out)
            return out
        elif (self.mode == 'SIDCCross34'):
            features = self.features(x)
            block0 = self.block0(features)
            transition0 = self.transition0(block0)
            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)

            block3 = self.block3(transition2)
            block3_aux = self.block3_aux(transition2)
            transition3 = self.final_bn(block3)
            transition3_aux = self.final_bn_aux(block3_aux)
           
            
            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)

            out, out_aux = apply_cross(self.cross3, out, out_aux)
            out,out_aux = F.relu(out, inplace=True),F.relu(out_aux, inplace=True)

            out = self.fc(out)
            out_aux = self.fc_aux(out_aux)
            out,out_aux = F.relu(out, inplace=True),F.relu(out_aux, inplace=True)

            out, out_aux = apply_cross(self.cross4, out, out_aux)
            out,out_aux = F.relu(out, inplace=True),F.relu(out_aux, inplace=True)

            out = self.classifier(out)
            out_aux = self.classifier_aux(out_aux)

            # out, out_aux = apply_cross(self.cross5,out,out_aux)

            return out, out_aux
        elif (self.mode == 'SIDCCross345'):
            features = self.features(x)
            block0 = self.block0(features)
            transition0 = self.transition0(block0)
            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)

            block3 = self.block3(transition2)
            block3_aux = self.block3_aux(transition2)
            transition3 = self.final_bn(block3)
            transition3_aux = self.final_bn_aux(block3_aux)
           
            
            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)
            
            out, out_aux = apply_cross(self.cross3, out, out_aux)
            out,out_aux = F.relu(out, inplace=True),F.relu(out_aux, inplace=True)

            out = self.fc(out)
            out_aux = self.fc_aux(out_aux)
            out,out_aux = F.relu(out, inplace=True),F.relu(out_aux, inplace=True)

            out, out_aux = apply_cross(self.cross4, out, out_aux)
            out,out_aux = F.relu(out, inplace=True),F.relu(out_aux, inplace=True)

            out = self.classifier(out)
            out_aux = self.classifier_aux(out_aux)
            out, out_aux = self.sigmoid(out),self.sigmoid(out_aux)
            
            out,out_aux = apply_cross(self.cross5,out,out_aux)


            return out, out_aux
        elif (self.mode == 'SIDCCross35'):
            features = self.features(x)
            block0 = self.block0(features)
            transition0 = self.transition0(block0)
            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)

            block3 = self.block3(transition2)
            block3_aux = self.block3_aux(transition2)
            transition3 = self.final_bn(block3)
            transition3_aux = self.final_bn_aux(block3_aux)
           
            
            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)


            out, out_aux = apply_cross(self.cross3, out, out_aux)
            out,out_aux = F.relu(out, inplace=True),F.relu(out_aux, inplace=True)

            out = self.f_classifier(out)
            out_aux = self.f_classifier_aux(out_aux)
            out, out_aux = self.sigmoid(out),self.sigmoid(out_aux)
            
            out,out_aux = apply_cross(self.cross5,out,out_aux)


            return out, out_aux
        elif (self.mode == 'SIDCPure'):
            features = self.features(x)
            block0 = self.block0(features)
            transition0 = self.transition0(block0)
            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)

            block3 = self.block3(transition2)
            block3_aux = self.block3_aux(transition2)
            transition3 = self.final_bn(block3)
            transition3_aux = self.final_bn_aux(block3_aux)
           
            
            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)


            out = self.f_classifier(out)
            out_aux = self.f_classifier_aux(out_aux)

            return out, out_aux

        else:
            raise Exception("No mode matched! Please check your cmd parameters.")



def clean_densenet161(pretrained=False, progress=True, **kwargs):
    return _clean_densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)

def _clean_densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = cleanDenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model




def apply_cross(linear, input1, input2):
    input1_reshaped, input2_reshaped = torch.flatten(input1, 1), torch.flatten(input2, 1)

    input_reshaped = torch.cat((input1_reshaped, input2_reshaped), dim = 1)
    output = linear(input_reshaped)
    
    
    output1 = output[:,:input1_reshaped.shape[1]].view(input1.shape)
    output2 = output[:,input1_reshaped.shape[1]:].view(input2.shape)
    return output1, output2