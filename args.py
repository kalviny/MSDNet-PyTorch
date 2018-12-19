import os
import glob
import time
import argparse

import config

# model_names = list(map(lambda n: os.path.basename(n)[:-3],
#                        glob.glob('models/[A-Za-z]*.py')))
model_names = ['densenet_mc', 'resnet_mc', 'msdnet', 'msdnet_lazy']

arg_parser = argparse.ArgumentParser(
                description='Image classification PK main script')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save', default='save/default-{}'.format(time.time()),
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/debug)')
exp_group.add_argument('--resume', action='store_true',
                    help='whether use the latest checkpoint (default: none)')
exp_group.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
exp_group.add_argument('--evaluate-from', default=None, type=str, metavar='PATH',
                    help='path to evaluate model (default: none)')
exp_group.add_argument('--print-freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--seed', default=0, type=int,
                       help='random seed')
exp_group.add_argument('--gpu',
                    help='GPU available.')

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='cifar10',
                        choices=config.datasets.keys(),
                        help='datasets: ' +
                        ' | '.join(config.datasets.keys()) +
                        ' (default: cifar10)')
data_group.add_argument('--no_valid', action='store_false', dest='use_validset',
                        help='not hold out 10 percent of training data as validation')
data_group.add_argument('--data_root', metavar='DIR', default='data',
                        help='path to dataset (default: data)')
data_group.add_argument('-j', '--workers', dest='num_workers', default=4,
                        type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
data_group.add_argument('--normalized', action='store_true',
                        help='normalize the data into zero mean and unit std')
data_group.add_argument('--augmentation', default=0.08, type=float, metavar='M',
                        help='')

# model arch related
arch_group = arg_parser.add_argument_group('arch',
                                           'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet)')
arch_group.add_argument('-d', '--depth', default=56, type=int, metavar='D',
                        help='depth (default=56)')
arch_group.add_argument('--drop-rate', default=0.0, type=float,
                        metavar='DROPRATE', help='dropout rate (default: 0.2)')
arch_group.add_argument('--death-mode', default='none',
                        choices=['none', 'linear', 'uniform'],
                        help='death mode (default: none)')
arch_group.add_argument('--death-rate', default=0.5, type=float,
                        help='death rate rate (default: 0.5)')
arch_group.add_argument('--growth-rate', default=12, type=int,
                        metavar='GR', help='Growth rate of DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')
arch_group.add_argument('--bn-size', default=4, type=int,
                        metavar='B', help='bottle neck ratio of DenseNet'
                        ' (0 means dot\'t use bottle necks) (default: 4)')
arch_group.add_argument('--reduction', default=0.5, type=float,
                        metavar='C', help='compression ratio of DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')
# used to set the argument when to resume automatically
arch_resume_names = ['arch', 'depth', 'death_mode', 'death_rate', 'death_rate',
                     'growth_rate', 'bn_size', 'compression']

# msdnet config
arch_group.add_argument('--nBlocks', type=int, default=6)
arch_group.add_argument('--nChannels', type=int, default=32)
arch_group.add_argument('--base', type=int,default=4)
arch_group.add_argument('--stepmode', type=str, choices=['even', 'lin_grow'])
arch_group.add_argument('--step', type=int, default=1)
arch_group.add_argument('--growthRate', type=int, default=6)
arch_group.add_argument('--grFactor', default='1-2-4-4', type=str)
arch_group.add_argument('--prune', default='max', choices=['min', 'max'])
arch_group.add_argument('--bnFactor', default='1-2-4-4')
arch_group.add_argument('--bottleneck', default=True, type=bool)


# training related
optim_group = arg_parser.add_argument_group('optimization',
                                            'optimization setting')
optim_group.add_argument('--trainer', default='train', type=str,
                         help='trainer file name without ".py"'
                         ' (default: train)')
optim_group.add_argument('--epochs', default=164, type=int, metavar='N',
                         help='number of total epochs to run (default: 164)')
optim_group.add_argument('--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
optim_group.add_argument('--patience', default=0, type=int, metavar='N',
                         help='patience for early stopping'
                         '(0 means no early stopping)')
optim_group.add_argument('-b', '--batch-size', default=64, type=int,
                         metavar='N', help='mini-batch size (default: 64)')
optim_group.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                         help='optimizer (default=sgd)')
optim_group.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                         metavar='LR',
                         help='initial learning rate (default: 0.1)')
optim_group.add_argument('--lr-type', default='multistep', type=str, metavar='T',
                        help='learning rate strategy (default: multistep)',
                        choices=['cosine', 'multistep'])
optim_group.add_argument('--decay_rate', default=0.1, type=float, metavar='N',
                         help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum (default=0.9)')
optim_group.add_argument('--no_nesterov', dest='nesterov',
                         action='store_false',
                         help='do not use Nesterov momentum')
optim_group.add_argument('--alpha', default=0.99, type=float, metavar='M',
                         help='alpha for ')
optim_group.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
optim_group.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
optim_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                         metavar='W', help='weight decay (default: 1e-4)')
