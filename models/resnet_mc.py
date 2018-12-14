# This implementation is based on the DenseNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import math
import torch
from torch import nn
from torchvision.models.resnet import conv3x3


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



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
        # print('input', x.size())
        x = self.conv_layer(x)
        # print('conv', x.size())
        x = x.view(x.size(0), -1)
        # print('linear', x.size())
        # print(self.linear)
        return self.linear(x)


class _resBlock(nn.Module):
    def __init__(self, inplanes, block, planes, blocks, stride=1, chosen=-1):
        super(_resBlock, self).__init__()
        self.layers = nn.ModuleList()
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.chosen = chosen - 1
        self.layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            self.layers.append(block(self.inplanes, planes))

    def forward(self, x):
        temp_x = None
        for i in range(len(self.layers)):
            # print(i, self.layers[i], x.size())
            x = self.layers[i](x)
            # print(x.size())
            if i == self.chosen:
                temp_x = x
        return x, temp_x


class ResNetImageNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.steps = [3, 3, 3, 1]
        self.layer1 = _resBlock(64, block, 64, layers[0], stride=1, chosen=self.steps[0])
        # print(self.layer1)
        self.layer2 = _resBlock(64, block, 128, layers[1], stride=2, chosen=self.steps[1])
        self.layer3 = _resBlock(128, block, 256, layers[2], stride=2, chosen=self.steps[2])
        # print(self.layer3)
        self.layer4 = _resBlock(256, block, 512, layers[3], stride=2, chosen=self.steps[3])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.inter_classifier = nn.ModuleList()
        image_size = (56, 28, 14, 7)
        nIn = (64, 128, 256, 512)
        for i in range(4):
            self.inter_classifier.append(_ClassifierWL(image_size[i], nIn[i]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        res = []
        x, temp_x = self.layer1(x)
        temp_x = self.inter_classifier[0](temp_x)
        res.append(temp_x)
        # print(x.size())
        x, temp_x = self.layer2(x)
        # print(1, temp_x.size())
        temp_x = self.inter_classifier[1](temp_x)
        res.append(temp_x)

        x, temp_x = self.layer3(x)
        # print(2, temp_x.size())
        temp_x = self.inter_classifier[2](temp_x)
        res.append(temp_x)

        x, temp_x = self.layer4(x)
        # print(3, temp_x.size())
        temp_x = self.inter_classifier[3](temp_x)
        res.append(temp_x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        res.append(x)
        return res


def createModel(depth, data, num_classes, death_mode='none', death_rate=0.5,
                **kwargs):
    print('Create ResNet-50 for {}'.format(data))
    return ResNetImageNet(BasicBlock, (3, 4, 6, 3), num_classes)