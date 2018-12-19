#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import config
from dataloader import getDataloaders
from utils import save_checkpoint, load_checkpoint, create_save_folder
from args import arg_parser, arch_resume_names
from opcounter import measure_model
import models
from adaptive_inference import DynamicEvaluate

args = arg_parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.config_of_data = config.datasets[args.data]
args.num_classes = config.datasets[args.data]['num_classes']

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
# args.grFactor = [int(ele) for ele in args.grFactor.split('-')]
# args.bnFactor = [int(ele) for ele in args.bnFactor.split('-')]
args.nScales = len(args.grFactor)
torch.manual_seed(args.seed)

def main():
    # parse arg and start experiment
    global args
    best_err1, best_epoch = 100., 0

    if args.data in ['cifar10', 'cifar100']:
        IMAGE_SIZE = 32
    else:
        IMAGE_SIZE = 224

    create_save_folder(args.save)
    model = getattr(models, args.arch)(args)
    n_flops, _, _ = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
    torch.save(n_flops, os.path.join(args.save, 'flop.pth'))
    del(model)

    model = getattr(models, args.arch)(args)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_err1 = checkpoint['best_err1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    if args.evaluate_from is not None:
        args.evaluate = True
        m = torch.load(args.evaluate_from)
        model.load_state_dict(m)

    cudnn.benchmark = True

    train_loader, val_loader, test_loader = getDataloaders(
        splits=('train', 'val'), **vars(args))


    if args.evaluate:
        DynamicEvaluate(model, test_loader, val_loader, criterion, args)

    # create dataloader
    # if args.evaluate == 'train':
    #     train_loader, _, _ = getDataloaders(
    #         splits=('train'), **vars(args))
    #     trainer.test(train_loader, best_epoch)
    #     return
    # elif args.evaluate == 'val':
    #     _, val_loader, _ = getDataloaders(
    #         splits=('val'), **vars(args))
    #     trainer.test(val_loader, best_epoch)
    #     return
    # elif args.evaluate == 'test':
    #     _, _, test_loader = getDataloaders(
    #         splits=('test'), **vars(args))
    #     trainer.test(test_loader, best_epoch)
    #     return
    # elif args.evaluate == 'flop':
    #     flops, _, _ = measure_model(model, 32, 32)
    #     torch.save(flops, os.path.join(args.save, 'flop.pth'))
    #     return
    # elif args.evaluate == 'evaluate':
    #     args.save = os.path.dirname(args.resume)
    #     _, val_loader, test_loader = getDataloaders(
    #         splits=('val', 'test'), **vars(args))
    #     evaluate(model, test_loader, val_loader, args)
    #     return
    # else:
    #     # check if the folder exists
    #     create_save_folder(args.save, args.force)
    #     train_loader, val_loader, test_loader = getDataloaders(
    #         splits=('train', 'val'), **vars(args))

    # set up logging
    global log_print, f_log
    f_log = open(os.path.join(args.save, 'log.txt'), 'w')

    def log_print(*args):
        print(*args)
        print(*args, file=f_log)
    log_print('args:')
    log_print(args)
    print('model:', file=f_log)
    print(model, file=f_log)
    log_print('# of params:',
              str(sum([p.numel() for p in model.parameters()])))

    f_log.flush()
    torch.save(args, os.path.join(args.save, 'args.pth'))
    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_err1'
              '\tval_err1\ttrain_err5\tval_err']

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_loss, train_err1, train_err5, lr = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, val_err1, val_err5 = validate(val_loader, model, criterion)

        # save scores to a tsv file, rewrite the whole file to prevent
        # accidental deletion
        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_err1, val_err1, train_err5, val_err5))

        is_best = val_err1 < best_err1
        if is_best:
            best_err1 = val_err1
            best_epoch = epoch
            print('Best var_err1 {}'.format(best_err1))

        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, scores)

    print('Best val_err1: {:.4f} at epoch {}'.format(best_err1, best_epoch))

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()

    running_lr = None
    for i, (input, target) in enumerate(train_loader):

        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)
        # measure data loading time
        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        if not isinstance(output, list):
            output = [output]

        loss = 0.0
        for j in range(len(output)):
            loss += criterion(output[j], target_var)
        # measure error and record loss
        losses.update(loss.item(), input.size(0))

        for j in range(len(output)):
            err1, err5 = error(output[j].data, target, topk=(1, 5))
            top1[j].update(err1.item(), input.size(0))
            top5[j].update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.4f}\t'
                  'Err@1 {top1.val:.4f}\t'
                  'Err@5 {top5.val:.4f}'.format(
                      epoch, i + 1, len(train_loader),
                      batch_time=batch_time, data_time=data_time,
                      loss=losses, top1=top1[-1], top5=top5[-1]))

            # print('Epoch: {:3d} Train loss {loss.avg:.4f} '
            #       'Err@1 {top1.avg:.4f}'
            #       ' Err@5 {top5.avg:.4f}'
            #       .format(epoch, loss=losses, top1=top1[-1], top5=top5[-1]))
            # print('Epoch: {:3d}, Train Loss: {loss.avg:.4f}'.format(epoch, loss=losses))
            # for j in range(self.args.nBlocks):
            #     print('Exit {} Err@1 {top1.avg:.4f}\t'
            #           'Err@5 {top5.avg:.4f}'.format(j,
            #         top1=top1[j], top5=top5[j]))
        # break
    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        if not isinstance(output, list):
            output = [output]

        loss = 0.0
        for j in range(len(output)):
            loss += criterion(output[j], target_var)

        # measure error and record loss
        losses.update(loss.item(), input.size(0))

        for j in range(len(output)):
            err1, err5 = error(output[j].data, target, topk=(1, 5))
            top1[j].update(err1.item(), input.size(0))
            top5[j].update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # break

        if i % args.print_freq == 0:
            for j in range(args.nBlocks):
                print('Err@1 {top1.avg:.4f}\t'
                      'Err@5 {top5.avg:.4f}'.format(
                    top1=top1[j], top5=top5[j]))

    # print(' * Err@1 {top1.avg:.3f} Err@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return losses.avg, top1[-1].avg, top5[-1].avg

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

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='multistep'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data in ['cifar10', 'cifar10+', 'cifar100', 'cifar100+']: # add augmentation 
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()
