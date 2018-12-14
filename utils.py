import sys
import time
import os
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
import math
import random


class MyRandomSizedCrop(object):
    def __init__(self, size, augmentation=0.08, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.augmentation = augmentation

    def __call__(self, img):
        for _ in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.augmentation, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = transforms.Scale(self.size, interpolation=self.interpolation)
        crop = transforms.CenterCrop(self.size)
        return crop(scale(img))


def create_save_folder(save_path, force=False, ignore_patterns=[]):
    if os.path.exists(save_path):
        #print(save_path + ' already exists!', file=sys.stderr)
        if not force:
            ans = input('Do you want to overwrite it? [y/N]:')
            if ans not in ('y', 'Y', 'yes', 'Yes'):
                os.exit(1)
        # from getpass import getuser
        # tmp_path = '/tmp/{}-experiments/{}_{}'.format(getuser(),
        #                                               os.path.basename(save_path),
        #                                               time.time())
        # print('move existing {} to {}'.format(save_path, tmp_path))
        # shutil.copytree(save_path, tmp_path)
        # shutil.rmtree(save_path)
    else:
        os.makedirs(save_path)
        print('create folder: ' + save_path)

    # copy code to save folder
    # if save_path.find('debug') < 0:
    #     shutil.copytree('.', os.path.join(save_path, 'src'), symlinks=True,
    #                     ignore=shutil.ignore_patterns('*.pyc', '__pycache__',
    #                                                   '*.path.tar', '*.pth',
    #                                                   '*.ipynb', '.*', 'data',
    #                                                   'save', 'save_backup',
    #                                                   save_path,
    #                                                   *ignore_patterns))


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs, args):
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    lr = lr_init
    if args.data == 'imagenet':
        if epoch >= 30:
            lr *= decay_rate
        elif epoch >= 60:
            lr *= decay_rate ** 2
    else:
        if epoch >= num_epochs * 0.75:
            lr *= decay_rate**2
        elif epoch >= num_epochs * 0.5:
            lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))


def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum, nesterov=args.nesterov,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), args.lr,
                                beta=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res



import torch
from torch.autograd import Variable
from functools import reduce
import operator


count_ops = 0
count_params = 0


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


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


""" The input batch size should be 1 to call this function """
def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)
    print(str(layer), count_ops)
    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    ### ops_learned_conv
    elif type_name in ['LearnedGroupConv']:
        measure_layer(layer.relu, x)
        measure_layer(layer.norm, x)
        conv = layer.conv
        out_h = int((x.size()[2] + 2 * conv.padding[0] - conv.kernel_size[0]) /
                    conv.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * conv.padding[1] - conv.kernel_size[1]) /
                    conv.stride[1] + 1)
        delta_ops = conv.in_channels * conv.out_channels * conv.kernel_size[0] * \
                conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add
        delta_params = get_layer_param(conv) / layer.condense_factor

    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d', 'MaxPool2d']:
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
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def measure_model(model, H, W):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, 3, H, W)).cuda()

    def modify_forward(model):
        for child in model.children():
            if is_leaf(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

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

    return count_ops, count_params