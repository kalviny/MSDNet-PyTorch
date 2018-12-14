import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models.densenet import _Transition


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        if bn_size > 0:
            self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False)),
            self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu.2', nn.ReLU(inplace=True)),
            self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False)),
        else:
            self.add_module('conv.1', nn.Conv2d(num_input_features, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, chozen_num=-1):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.chosen = chozen_num - 1
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.layers.append(layer)

    def forward(self, x):
        temp_x = None # to store the result of intermediate classifier
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            # print(i, new_x.size())
            if i == self.chosen:
                temp_x = x
        # print('temp x', temp_x.size())
        return x, temp_x


def build_classifers(image_size, nIn):
    layers = []

    def scaling_down():
        layers.append(nn.Conv2d(nIn, nIn, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(nIn))
        layers.append(nn.ReLU(inplace=True))

    while image_size > 1:
        scaling_down()
        image_size //= 2

    layers.append(nn.AvgPool2d(2))
    return nn.Sequential(*layers)


class _ClassifierWL(nn.Module):
    def __init__(self, image_size, nIn):
        super(_ClassifierWL, self).__init__()
        self.conv_layer = build_classifers(image_size, nIn)
        self.linear = nn.Linear(nIn, 1000)

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class DenseNet_MC(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0, avgpool_size=7,
                 num_classes=1000):

        super(DenseNet_MC, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between '
        self.avgpool_size = avgpool_size
        self.steps = [6, 12, 12, 2]
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(3, 2, 1))
        ]))

        self.layers = nn.ModuleList()
        self.inter_classifers = nn.ModuleList()
        image_size = [56, 28, 14, 7]
        print('block config', block_config)

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate, chozen_num=self.steps[i])
            self.layers.append(block)
            self.inter_classifers[i] = _ClassifierWL(image_size[i], num_features + self.steps[i] * growth_rate)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features
                                                            * compression))
                self.layers.append(trans)
                num_features = int(num_features * compression)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        res = []
        for i in range(len(self.layers)):
            if i % 2 == 1:
                features = self.layers[i](features)
            else:
                features, inter_res = self.layers[i](features)
                res.append(self.inter_classifers[i // 2](inter_res))
                # print(self.inter_classifers[i // 2])
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(
            features.size(0), -1)
        out = self.classifier(out)
        res.append(out) # return the result for all classifiers
        return res


def createModel(data, depth=100, growth_rate=12, num_classes=1000, drop_rate=0,
                num_init_features=24, compression=0.5, bn_size=4, **kwargs):
    avgpool_size = 7 if data == 'imagenet' else 8
    N = (depth - 4) // 3
    suffix = '-'
    if bn_size > 0:
        N //= 2
        suffix += 'B'
    block_config = (6, 12, 24, 16)
    if compression < 1.:
        suffix += 'C'

    if suffix == '-':
        suffix = ''
    print('Create DenseNet MC 121 for {}'.format(data))
    network = DenseNet_MC(growth_rate=growth_rate, num_classes=num_classes,
                    compression=compression, drop_rate=drop_rate, bn_size=bn_size,
                    block_config=block_config, avgpool_size=avgpool_size)
    print('network created')
    return network
