

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


class NddrDenseNet(nn.Module):
    __constants__ = ['features']
    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=4, drop_rate=0, num_classes=2, length_aux = 10 , mode = None, nddr_drop_rate = 0, memory_efficient=False):

        super(NddrDenseNet, self).__init__()

        # First convolution
        # 这里将feature_map的channel从3个转为96个
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        self.features_aux = nn.Sequential(OrderedDict([
            ('conv0_aux', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0_aux', nn.BatchNorm2d(num_init_features)),
            ('relu0_aux', nn.ReLU(inplace=True)),
            ('pool0_aux', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.nddrf = NddrLayer(net0_channels=num_init_features, net1_channels= num_init_features, drop_rate= nddr_drop_rate)

  

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
                self.block0_aux = block_aux
            elif (i == 1):
                self.block1 = block
                self.block1_aux = block_aux
            elif (i == 2):
                self.block2 = block
                self.block2_aux = block_aux
            elif (i == 3):
                self.block3 = block
                self.block3_aux = block_aux
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                trans_aux = _Transition(num_input_features=num_features,   
                                    num_output_features=num_features // 2) 
                if (i == 0):
                    self.transition0 = trans
                    self.transition0_aux = trans_aux
                    self.nddr0 = NddrLayer(net0_channels=num_features // 2, net1_channels=num_features // 2, drop_rate=nddr_drop_rate)
                    self.sluice0_reshape = nn.Linear(num_features // 2, 2208)
                    self.sluice0_aux_reshape = nn.Linear(num_features // 2, 2208)
                    # self.nddr0_conv1d = nn.Conv1d(num_features // 2 *2, num_features // 2)
                    # self.nddr0_conv1d_aux =  
                    # self.nddr0_bn = nn.BatchNorm2d(num_features // 2)
                    # self.nddr0_bn_aux = nn.BatchNorm2d(num_features // 2)
                elif (i == 1):
                    self.transition1 = trans
                    self.transition1_aux = trans_aux
                    self.nddr1 = NddrLayer(net0_channels=num_features // 2, net1_channels= num_features // 2, drop_rate= nddr_drop_rate)
                    self.sluice1_reshape = nn.Linear(num_features // 2, 2208)
                    self.sluice1_aux_reshape = nn.Linear(num_features // 2, 2208)
                elif (i == 2):
                    self.transition2 = trans
                    self.transition2_aux = trans_aux
                    self.nddr2 = NddrLayer(net0_channels=num_features // 2, net1_channels=num_features // 2, drop_rate=nddr_drop_rate)
                    self.sluice2_reshape = nn.Linear(num_features // 2, 2208)
                    self.sluice2_aux_reshape = nn.Linear(num_features // 2, 2208)
                num_features = num_features // 2

        # Final batch norm
        
        self.final_bn = nn.BatchNorm2d(num_features)
        self.final_bn_aux = nn.BatchNorm2d(num_features)
        self.nddr3 = NddrLayer(net0_channels=num_features, net1_channels= num_features, drop_rate= nddr_drop_rate)
        self.sluice3_reshape = nn.Linear(num_features, 2208)
        self.sluice3_aux_reshape = nn.Linear(num_features, 2208)

        # Linear layer
        self.vggfc = nn.Linear(num_features,num_features)
        self.vggfc_aux = nn.Linear(num_features,num_features)
        self.fc = nn.Linear(num_features,1000)
        self.fc_aux =  nn.Linear(num_features,1000)
        self.classifier = nn.Linear(1000, num_classes) #TODO:Finetune
        self.classifier_aux = nn.Linear(1000, length_aux)
        self.f_classifier = nn.Linear(num_features, num_classes)
        self.f_classifier_aux = nn.Linear(num_features, length_aux)
        # Official init from torch repo.
        
        self.cross3 = nn.Linear(num_features*2,num_features*2,bias=False)
        self.cross4 = nn.Linear(2000, 2000, bias=False)
        self.cross5 = nn.Linear(6, 6, bias=False)
        self.betas_5layer = nn.Parameter(torch.tensor([0.05, 0.1, 0.1, 0.25, 0.5]))
        self.betas_5layer_aux = nn.Parameter(torch.tensor([0.05, 0.1, 0.1, 0.25, 0.5]))
        self.betas_6layer = nn.Parameter(torch.tensor([0.05, 0.1, 0.1, 0.25, 0.5,0.3]))
        self.betas_6layer_aux = nn.Parameter(torch.tensor([0.05, 0.1, 0.1, 0.25, 0.5,0.3]))
        # crossP3_stdv = 1. / math.sqrt(num_features*2)
        # self.crossP3 = nn.init.uniform_(nn.Parameter(torch.empty(num_features*2,num_features*2)),-crossP3_stdv,crossP3_stdv)
        # crossP4_stdv = 1. / math.sqrt(128)
        # self.crossP4 = nn.init.uniform_(nn.Parameter(torch.empty(128, 128)),-crossP4_stdv,crossP4_stdv)
        # crossP5_stdv = 1. / math.sqrt(6)
        # self.crossP5 = nn.init.uniform_(nn.Parameter(torch.empty(6, 6)),-crossP5_stdv,crossP5_stdv)
        self.sigmoid = nn.Sigmoid()
        

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
            # print('0',transition0.shape)

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            # print('1',transition1.shape)

            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)
            # print('2',transition2.shape)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            # print('3', transition3.shape)
            
            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            # print('out0', out.shape)
            out = torch.flatten(out, 1)
            # print('out1', out.shape)
            out = self.f_classifier(out)
            return out
            # done
        elif (self.mode == 'NddrPure'):
            features = self.features(x)
            features_aux = self.features_aux(x)
            
            features, features_aux = self.nddrf(features, features_aux)

            block0 = self.block0(features)
            transition0 = self.transition0(block0)
            block0_aux = self.block0_aux(features_aux)
            transition0_aux = self.transition0_aux(block0_aux)

            transition0, transition0_aux = self.nddr0(transition0, transition0_aux)

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block1_aux = self.block1_aux(transition0_aux)
            transition1_aux = self.transition1_aux(block1_aux)

            transition1, transition1_aux = self.nddr1(transition1, transition1_aux)

            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)         
            block2_aux = self.block2_aux(transition1_aux)
            transition2_aux = self.transition2_aux(block2_aux)

            transition2, transition2_aux = self.nddr2(transition2, transition2_aux)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            block3_aux = self.block3_aux(transition2_aux)
            transition3_aux = self.final_bn_aux(block3_aux)
            
            transition3, transition3_aux = self.nddr3(transition3, transition3_aux)

            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)

            out = self.f_classifier(out)
            out_aux = self.f_classifier_aux(out_aux)

            return out, out_aux
        
        elif (self.mode == 'NddrLSC'):
            # beta6
            features = self.features(x)
            features_aux = self.features_aux(x)
            
            features, features_aux = self.nddrf(features, features_aux)

            block0 = self.block0(features)
            transition0 = self.transition0(block0)
            block0_aux = self.block0_aux(features_aux)
            transition0_aux = self.transition0_aux(block0_aux)

            sluice0 = F.relu(transition0, inplace=True)
            sluice0 = F.adaptive_avg_pool2d(sluice0, (1, 1))
            sluice0 = torch.flatten(sluice0, 1)
            sluice0 = self.sluice0_reshape(sluice0)
            sluice0_aux = F.relu(transition0_aux, inplace=True)
            sluice0_aux = F.adaptive_avg_pool2d(sluice0_aux, (1, 1))
            sluice0_aux = torch.flatten(sluice0_aux, 1)
            sluice0_aux = self.sluice0_aux_reshape(sluice0_aux)
            #TODO
            # while β parameters control which layer outputs are used for prediction.


            transition0, transition0_aux = self.nddr0(transition0, transition0_aux)

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block1_aux = self.block1_aux(transition0_aux)
            transition1_aux = self.transition1_aux(block1_aux)


            sluice1 = F.relu(transition1, inplace=True)
            sluice1 = F.adaptive_avg_pool2d(sluice1, (1, 1))
            sluice1 = torch.flatten(sluice1, 1)
            sluice1 = self.sluice1_reshape(sluice1)
            sluice1_aux = F.relu(transition1_aux, inplace=True)
            sluice1_aux = F.adaptive_avg_pool2d(sluice1_aux, (1, 1))
            sluice1_aux = torch.flatten(sluice1_aux, 1)
            sluice1_aux = self.sluice1_aux_reshape(sluice1_aux)

            transition1, transition1_aux = self.nddr1(transition1, transition1_aux)

            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)         
            block2_aux = self.block2_aux(transition1_aux)
            transition2_aux = self.transition2_aux(block2_aux)

            sluice2 = F.relu(transition2, inplace=True)
            sluice2 = F.adaptive_avg_pool2d(sluice2, (1, 1))
            sluice2 = torch.flatten(sluice2, 1)
            sluice2 = self.sluice2_reshape(sluice2)
            sluice2_aux = F.relu(transition2_aux, inplace=True)
            sluice2_aux = F.adaptive_avg_pool2d(sluice2_aux, (1, 1))
            sluice2_aux = torch.flatten(sluice2_aux, 1)
            sluice2_aux = self.sluice2_aux_reshape(sluice2_aux)


            transition2, transition2_aux = self.nddr2(transition2, transition2_aux)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            block3_aux = self.block3_aux(transition2_aux)
            transition3_aux = self.final_bn_aux(block3_aux)
            
            sluice3 = F.relu(transition3, inplace=True)
            sluice3 = F.adaptive_avg_pool2d(sluice3, (1, 1))
            sluice3 = torch.flatten(sluice3, 1)
            sluice3 = self.sluice3_reshape(sluice3)
            sluice3_aux = F.relu(transition3_aux, inplace=True)
            sluice3_aux = F.adaptive_avg_pool2d(sluice3_aux, (1, 1))
            sluice3_aux = torch.flatten(sluice3_aux, 1)
            sluice3_aux = self.sluice3_aux_reshape(sluice3_aux)


            transition3, transition3_aux = self.nddr3(transition3, transition3_aux)

            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)

            outc, outc_aux = apply_cross(self.cross3, out, out_aux)

            out = torch.mul(sluice0,self.betas_6layer[0]) + torch.mul(sluice1,self.betas_6layer[1]) + torch.mul(sluice2,self.betas_6layer[2]) + torch.mul(sluice3,self.betas_6layer[3]) + torch.mul(out,self.betas_6layer[4]) + torch.mul(outc,self.betas_6layer[5])
            
            out_aux = torch.mul(sluice0_aux,self.betas_6layer_aux[0]) + torch.mul(sluice1_aux,self.betas_6layer_aux[1]) + torch.mul(sluice2_aux,self.betas_6layer_aux[2]) + torch.mul(sluice3_aux,self.betas_6layer_aux[3]) + torch.mul(out_aux,self.betas_6layer_aux[4]) + torch.mul(outc_aux,self.betas_6layer[5])

            out = self.f_classifier(out)
            out_aux = self.f_classifier_aux(out_aux)

            return out, out_aux
        
        
        elif (self.mode == 'NddrLS'):
            # beta5
            features = self.features(x)
            features_aux = self.features_aux(x)
            
            features, features_aux = self.nddrf(features, features_aux)

            block0 = self.block0(features)
            transition0 = self.transition0(block0)
            block0_aux = self.block0_aux(features_aux)
            transition0_aux = self.transition0_aux(block0_aux)

            sluice0 = F.relu(transition0, inplace=True)
            sluice0 = F.adaptive_avg_pool2d(sluice0, (1, 1))
            sluice0 = torch.flatten(sluice0, 1)
            sluice0 = self.sluice0_reshape(sluice0)
            sluice0_aux = F.relu(transition0_aux, inplace=True)
            sluice0_aux = F.adaptive_avg_pool2d(sluice0_aux, (1, 1))
            sluice0_aux = torch.flatten(sluice0_aux, 1)
            sluice0_aux = self.sluice0_aux_reshape(sluice0_aux)
            #TODO
            # while β parameters control which layer outputs are used for prediction.


            transition0, transition0_aux = self.nddr0(transition0, transition0_aux)

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block1_aux = self.block1_aux(transition0_aux)
            transition1_aux = self.transition1_aux(block1_aux)


            sluice1 = F.relu(transition1, inplace=True)
            sluice1 = F.adaptive_avg_pool2d(sluice1, (1, 1))
            sluice1 = torch.flatten(sluice1, 1)
            sluice1 = self.sluice1_reshape(sluice1)
            sluice1_aux = F.relu(transition1_aux, inplace=True)
            sluice1_aux = F.adaptive_avg_pool2d(sluice1_aux, (1, 1))
            sluice1_aux = torch.flatten(sluice1_aux, 1)
            sluice1_aux = self.sluice1_aux_reshape(sluice1_aux)

            transition1, transition1_aux = self.nddr1(transition1, transition1_aux)

            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)         
            block2_aux = self.block2_aux(transition1_aux)
            transition2_aux = self.transition2_aux(block2_aux)

            sluice2 = F.relu(transition2, inplace=True)
            sluice2 = F.adaptive_avg_pool2d(sluice2, (1, 1))
            sluice2 = torch.flatten(sluice2, 1)
            sluice2 = self.sluice2_reshape(sluice2)
            sluice2_aux = F.relu(transition2_aux, inplace=True)
            sluice2_aux = F.adaptive_avg_pool2d(sluice2_aux, (1, 1))
            sluice2_aux = torch.flatten(sluice2_aux, 1)
            sluice2_aux = self.sluice2_aux_reshape(sluice2_aux)


            transition2, transition2_aux = self.nddr2(transition2, transition2_aux)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            block3_aux = self.block3_aux(transition2_aux)
            transition3_aux = self.final_bn_aux(block3_aux)
            
            sluice3 = F.relu(transition3, inplace=True)
            sluice3 = F.adaptive_avg_pool2d(sluice3, (1, 1))
            sluice3 = torch.flatten(sluice3, 1)
            sluice3 = self.sluice3_reshape(sluice3)
            sluice3_aux = F.relu(transition3_aux, inplace=True)
            sluice3_aux = F.adaptive_avg_pool2d(sluice3_aux, (1, 1))
            sluice3_aux = torch.flatten(sluice3_aux, 1)
            sluice3_aux = self.sluice3_aux_reshape(sluice3_aux)


            transition3, transition3_aux = self.nddr3(transition3, transition3_aux)

            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)

            out = torch.mul(sluice0,self.betas_5layer[0]) + torch.mul(sluice1,self.betas_5layer[1]) + torch.mul(sluice2,self.betas_5layer[2]) + torch.mul(sluice3,self.betas_5layer[3]) + torch.mul(out,self.betas_5layer[4])
            out_aux = torch.mul(sluice0_aux,self.betas_5layer_aux[0]) + torch.mul(sluice1_aux,self.betas_5layer_aux[1]) + torch.mul(sluice2_aux,self.betas_5layer_aux[2]) + torch.mul(sluice3_aux,self.betas_5layer_aux[3]) + torch.mul(out_aux,self.betas_5layer_aux[4])

            out = self.f_classifier(out)
            out_aux = self.f_classifier_aux(out_aux)

            return out, out_aux
        
        
        
            # done
        elif (self.mode == 'NddrCross3'):
            features = self.features(x)
            features_aux = self.features_aux(x)

            features, features_aux = self.nddrf(features, features_aux)

            block0 = self.block0(features)
            transition0 = self.transition0(block0)
            block0_aux = self.block0_aux(features_aux)
            transition0_aux = self.transition0_aux(block0_aux)

            transition0, transition0_aux = self.nddr0(transition0, transition0_aux)

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block1_aux = self.block1_aux(transition0_aux)
            transition1_aux = self.transition1_aux(block1_aux)

            transition1, transition1_aux = self.nddr1(transition1, transition1_aux)

            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)         
            block2_aux = self.block2_aux(transition1_aux)
            transition2_aux = self.transition2_aux(block2_aux)

            transition2, transition2_aux = self.nddr2(transition2, transition2_aux)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            block3_aux = self.block3_aux(transition2_aux)
            transition3_aux = self.final_bn_aux(block3_aux)
            
            transition3, transition3_aux = self.nddr3(transition3, transition3_aux)

            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)

            out, out_aux = apply_cross(self.cross3, out, out_aux)

            out = self.f_classifier(out)
            out_aux = self.f_classifier_aux(out_aux)

            return out, out_aux
        elif (self.mode == 'NddrCross5'):
            features = self.features(x)
            features_aux = self.features_aux(x)

            features, features_aux = self.nddrf(features, features_aux)

            block0 = self.block0(features)
            transition0 = self.transition0(block0)
            block0_aux = self.block0_aux(features_aux)
            transition0_aux = self.transition0_aux(block0_aux)

            transition0, transition0_aux = self.nddr0(transition0, transition0_aux)

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block1_aux = self.block1_aux(transition0_aux)
            transition1_aux = self.transition1_aux(block1_aux)

            transition1, transition1_aux = self.nddr1(transition1, transition1_aux)

            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)         
            block2_aux = self.block2_aux(transition1_aux)
            transition2_aux = self.transition2_aux(block2_aux)

            transition2, transition2_aux = self.nddr2(transition2, transition2_aux)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            block3_aux = self.block3_aux(transition2_aux)
            transition3_aux = self.final_bn_aux(block3_aux)
            
            transition3, transition3_aux = self.nddr3(transition3, transition3_aux)

            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)

            
            out = self.f_classifier(out)
            out_aux = self.f_classifier_aux(out_aux)
            out, out_aux = apply_cross(self.cross5, out, out_aux)
            return out, out_aux
            # done
        elif (self.mode == 'NddrCross35'):
            features = self.features(x)
            features_aux = self.features_aux(x)

            features, features_aux = self.nddrf(features, features_aux)

            block0 = self.block0(features)
            transition0 = self.transition0(block0)
            block0_aux = self.block0_aux(features_aux)
            transition0_aux = self.transition0_aux(block0_aux)

            transition0, transition0_aux = self.nddr0(transition0, transition0_aux)

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block1_aux = self.block1_aux(transition0_aux)
            transition1_aux = self.transition1_aux(block1_aux)

            transition1, transition1_aux = self.nddr1(transition1, transition1_aux)

            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)         
            block2_aux = self.block2_aux(transition1_aux)
            transition2_aux = self.transition2_aux(block2_aux)

            transition2, transition2_aux = self.nddr2(transition2, transition2_aux)

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            block3_aux = self.block3_aux(transition2_aux)
            transition3_aux = self.final_bn_aux(block3_aux)
            
            transition3, transition3_aux = self.nddr3(transition3, transition3_aux)

            out = F.relu(transition3, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_aux = F.relu(transition3_aux, inplace=True)
            out_aux = F.adaptive_avg_pool2d(out_aux, (1, 1))
            out_aux = torch.flatten(out_aux, 1)

            out, out_aux = apply_cross(self.cross3, out, out_aux)

            out = self.f_classifier(out)
            out_aux = self.f_classifier_aux(out_aux)

            out, out_aux = apply_cross(self.cross5, out, out_aux)
            return out, out_aux
            # done
        elif (self.mode == 'SingleTasks'):
            features = self.features(x)
            features_aux = self.features_aux(x)

            block0 = self.block0(features)
            transition0 = self.transition0(block0)
            block0_aux = self.block0_aux(features_aux)
            transition0_aux = self.transition0_aux(block0_aux)

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            block1_aux = self.block1_aux(transition0_aux)
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
                      
            out = self.f_classifier(out)
            out_aux = self.f_classifier_aux(out_aux)
            # done
            # out = self.vggfc(out)
            # out = F.relu(out, inplace=True)
            # out = F.dropout(new_features, p=self.drop_rate,
            #   training=self.training)
            # out = self.fc(out)
            # out = F.relu(out, inplace=True)
            # out = self.dropout(out)
            # out = self.classifier(out)

            # out_aux = self.vggfc_aux(out_aux)
            # out_aux = F.relu(out_aux, inplace=True)
            # out_aux = self.dropout(out_aux)
            # out_aux = self.fc_aux(out_aux)
            # out_aux = F.relu(out_aux, inplace=True)
            # out_aux = self.dropout(out_aux)
            # out_aux = self.classifier_aux(out_aux)

            return out, out_aux

        elif (self.mode == 'MultiTasks'):
            features = self.features(x)

            block0 = self.block0(features)
            transition0 = self.transition0(block0)
           

            block1 = self.block1(transition0)
            transition1 = self.transition1(block1)
            

            block2 = self.block2(transition1)
            transition2 = self.transition2(block2)
           

            block3 = self.block3(transition2)
            transition3 = self.final_bn(block3)
            
            
            out_share = F.relu(transition3, inplace=True)
            out_share = F.adaptive_avg_pool2d(out_share, (1, 1))
            out_share = torch.flatten(out_share, 1)

        
                      
            out = self.f_classifier( out_share)
            out_aux = self.f_classifier_aux( out_share)
            # out = self.vggfc(out_share)
            # out = F.relu(out, inplace=True)
            # out = self.dropout(out)
            # out = self.fc(out)
            # out = F.relu(out, inplace=True)
            # out = self.dropout(out)
            # out = self.classifier(out)

            # out_aux = self.vggfc_aux(out_share)
            # out_aux = F.relu(out_aux, inplace=True)
            # out_aux = self.dropout(out_aux)
            # out_aux = self.fc_aux(out_aux)
            # out_aux = F.relu(out_aux, inplace=True)
            # out_aux = self.dropout(out_aux)
            # out_aux = self.classifier_aux(out_aux)

            return out, out_aux
            # done
        
        elif (self.mode == 'SIDCCross3'):
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
            out,out_aux = F.relu(out, inplace=False), F.relu(out_aux, inplace=False)

            # out = self.fc(out)
            # out_aux = self.fc_aux(out_aux)
            # out,out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)


            # out,out_aux = apply_cross(self.cross4,out,out_aux)
            # out,out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)

            out = self.f_classifier(out)
            out_aux = self.f_classifier_aux(out_aux)

            # out, out_aux = apply_cross(self.cross5,out,out_aux)

            return out, out_aux

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
            out,out_aux = F.relu(out, inplace=False), F.relu(out_aux, inplace=False)

            out = self.fc(out)
            out_aux = self.fc_aux(out_aux)
            out,out_aux = F.relu(out, inplace=False), F.relu(out_aux, inplace=False)


            out,out_aux = apply_cross(self.cross4,out,out_aux)
            out,out_aux = F.relu(out, inplace=False), F.relu(out_aux, inplace=False)

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
            # out,out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)
            

            out = self.fc(out)
            out_aux = self.fc_aux(out_aux)
            out,out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)


            out,out_aux = apply_cross(self.cross4,out,out_aux)
            # out,out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)


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
            # out,out_aux = F.relu(out, inplace=True), F.relu(out_aux, inplace=True)

            out = self.f_classifier(out)
            out_aux = self.f_classifier_aux(out_aux)
            # out, out_aux = self.sigmoid(out),self.sigmoid(out_aux)
            
            # out,out_aux = apply_cross(self.cross5,out,out_aux)

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



def nddr_densenet161(pretrained=False, progress=True, **kwargs):
    return _nddr_densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)

def _nddr_densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = NddrDenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        # 这里可以改成自己保存的state_dict
        _load_state_dict(model, model_urls[arch], progress)
    return model



def apply_cross(linear, input1, input2):
    input1_reshaped, input2_reshaped = torch.flatten(input1, 1), torch.flatten(input2, 1)

    input_reshaped = torch.cat((input1_reshaped, input2_reshaped), dim = 1)
    output = linear(input_reshaped)
    
    
    output1 = output[:,:input1_reshaped.shape[1]].view(input1.shape)
    output2 = output[:,input1_reshaped.shape[1]:].view(input2.shape)
    return output1, output2

def apply_cross_P(alpha, input1, input2):
    input1_reshaped, input2_reshaped = torch.flatten(input1, 1), torch.flatten(input2, 1)

    input_reshaped = torch.cat((input1_reshaped, input2_reshaped), dim=1)
    output = torch.matmul(input_reshaped, alpha)
    # output = linear(input_reshaped)
    
    
    output1 = output[:,:input1_reshaped.shape[1]].view(input1.shape)
    output2 = output[:,input1_reshaped.shape[1]:].view(input2.shape)
    return output1, output2


class NddrLayer(nn.Module):
    def __init__(self,net0_channels, net1_channels, drop_rate=0):
        super(NddrLayer, self).__init__()
        self.drop_rate = drop_rate
        self.nddr_conv1d_task0 = nn.Conv2d(net0_channels + net1_channels, net0_channels, 1)
        self.nddr_conv1d_task1 = nn.Conv2d(net0_channels + net1_channels, net1_channels, 1)
        # 可在此处加dropout      
        self.nddr_bn_task0 = nn.BatchNorm2d(net0_channels)
        self.nddr_bn_task1 = nn.BatchNorm2d(net1_channels)
        
    def forward(self, net0, net1):
        nddr = torch.cat((net0, net1), dim=1)
        net0 = self.nddr_conv1d_task0(nddr)
        net1 = self.nddr_conv1d_task1(nddr)
        if (self.drop_rate > 0):
            net0 = F.dropout(net0, p=self.drop_rate,
                                     training=self.training)
            net0 = F.dropout(net0, p=self.drop_rate,
                                     training=self.training)
        net0 = self.nddr_bn_task0(net0)
        net1 = self.nddr_bn_task1(net1)
        return net0, net1

        