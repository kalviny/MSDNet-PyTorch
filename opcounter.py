"""
    operation counter, modified from:
    https://github.com/ShichenLiu/CondenseNet/blob/master/utils.py and
    https://github.com/apaszke/torch-opCounter/blob/master/src/profiler.lua
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Variable
from functools import reduce
import operator

count_ops = 0
count_params = 0
count_tillnow = []
count_per_layer = []
classifiers = []

def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def convert_model(model, args):
    for m in model._modules:
        child = model._modules[m]
        convert_model(child, args)


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_ops, count_params, count_tillnow, count_per_layer
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)
    # print(type_name)
    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout',
                       'MSDNFirstLayer', 'ConvBasic', 'ConvBN',
                       'ParallelModule', 'MSDNet', 'Sequential']:
        delta_params = get_layer_param(layer)

    ### flags, use to divede scales/layers of MSDNet
    elif type_name in ['MSDNLayer', 'ConvDownNormal', 'ConvNormal',
                       'ClassifierModule']:

        if type_name == 'MSDNLayer':
            if len(count_per_layer) > 0:
                count_tillnow.append(count_per_layer)
                count_per_layer = [] # a new column of layer
                classifiers.append(False) # not a layer with classifier
        elif type_name == 'ClassifierModule':
            count_per_layer[-1] = count_ops
            classifiers[-1] = True
        else:
            count_per_layer.append(count_ops)

        delta_params = get_layer_param(layer)

    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def measure_model(model, H, W):
    global count_ops, count_params, count_per_layer, count_tillnow, classifiers
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, 3, H, W))

    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        # print(get_layer_info(m), count_ops)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                # print(get_layer_info(child))
                modify_forward(child)
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        # print(get_layer_info(m), count_ops)
                        return m.old_forward(x)
                    return lambda_forward

                child.old_forward = child.forward
                child.forward = new_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    # postprosses, to get number for each classifier
    count_tillnow.append([count_ops])
    count_tillnow[0][0] = 0
    conut_each_component = []

    for i in range(len(count_tillnow) - 1):
        tmp = []
        for j in range(len(count_tillnow[i]) - 1):
            tmp.append(count_tillnow[i][j + 1] - count_tillnow[i][j])
        tmp.append(count_tillnow[i + 1][0] - count_tillnow[i][-1])
        conut_each_component.append(tmp)

    res = []
    for i, is_classifer in enumerate(classifiers):
        if is_classifer:
            flop = count_tillnow[i + 1][0]
            for j in range(len(conut_each_component[0]))[::-1]:
                flop -= sum(conut_each_component[i - j][: -j])
            res.append(flop)

    return res, count_ops, count_params

