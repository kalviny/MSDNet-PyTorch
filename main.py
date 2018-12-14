#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import shutil
import warnings
import models
import config
from dataloader import get_dataloader
from args import arg_parser, arch_resume_names
from opcounter import measure_model

def evaluate(model, test_loader, val_loader, args):
    from train import Tester
    tester = Tester(model, args)
    if os.path.exists(os.path.join(args.savedir, 'logits_greedy.pth')):
        val_pred_greedy, val_target, test_pred_greedy, test_target = \
            torch.load(os.path.join(args.savedir, 'logits_greedy.pth'))
    else:
        if os.path.exists(os.path.join(args.savedir, 'logits_single.pth')):
            val_pred_single, val_target, test_pred_single, test_target = \
                torch.load(os.path.join(args.savedir, 'logits_single.pth'))
        else:
            val_pred_single, val_target = tester.calc_logit(val_loader)
            test_pred_single, test_target = tester.calc_logit(test_loader)
            torch.save((val_pred_single, val_target, test_pred_single, test_target),
                       os.path.join(args.savedir, 'logits_single.pth'))

        # early_exit
        best_ensemble_scheme = tester.find_greedy_ensemble(val_pred_single, val_target)
        val_pred_greedy, val_res = \
            tester.early_exit(val_pred_single, val_target, best_ensemble_scheme)
        test_pred_greedy, test_res = \
            tester.early_exit(test_pred_single, test_target, best_ensemble_scheme)

        torch.save((val_pred_greedy, val_target,
                    test_pred_greedy, test_target),
                   os.path.join(args.savedir, 'logits_greedy.pth'))
        print(test_res)
        torch.save((best_ensemble_scheme, test_res), 'early_exit_res.pth')

    # load flops
    flops = torch.load(os.path.join(args.savedir, 'flops.pth'))

    # dynamic evaluation
    for p in range(1, 41):
        _p = torch.FloatTensor(1).fill_(p / 20)
        # probs = [math.exp(math.log(p) * i) for i in in range(1, tester.args.nBlocks + 1)] # geometric distribution
        probs = torch.exp(torch.log(_p) * torch.range(1, tester.args.nBlocks))
        probs /= probs.sum()
        print(p, probs)
        acc_val, _, T = tester.dynamic_eval_find_threshold(
            val_pred_greedy, val_target, probs, flops)
        acc_test, exp_flops = tester.dynamic_eval_with_threshold(
            test_pred_greedy, test_target, flops, T)


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

args = arg_parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

args.config_of_data = config.datasets[args.data]
args.num_classes = config.datasets[args.data]['num_classes']

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

if args.no_valid:
    args.splits = ['train', 'val']
else:
    args.splits = ['train', 'val', 'test']

def main():
    # parse arg and start experiment
    global args, best_prec1
    best_err1, best_epoch = 100., 0

    model = getattr(models, args.model)(args)

    if args.model.startswith('alexnet') or args.model.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    if args.evaluate_from is not None:
        args.evaluate = True
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

    train_loader, val_loader, test_loader = get_dataloader(splits=args.splits, **vars(args))

    # create dataloader
    if args.evaluate == 'train':
        trainer.test(train_loader, best_epoch)
        return
    elif args.evaluate == 'val':
        trainer.test(val_loader, best_epoch)
        return
    elif args.evaluate == 'test':
        trainer.test(test_loader, best_epoch)
        return
    elif args.evaluate == 'flop':
        flops, _, _ = measure_model(model, 32, 32)
        torch.save(flops, os.path.join(args.savedir, 'flop.pth'))
        return
    elif args.evaluate == 'evaluate':
        args.savedir = os.path.dirname(args.resume)
        _, val_loader, test_loader = getDataloaders(
            splits=('val', 'test'), **vars(args))
        evaluate(model, test_loader, val_loader, args)
        return
    else:
        # check if the folder exists
        create_save_folder(args.savedir, args.force)
        print('=> getting dataloaders')
        train_loader, val_loader, test_loader = getDataloaders(
            splits=('train', 'val'), **vars(args))
        print('=> got dataloaders')

    Trainer = import_module(args.trainer).Trainer
    trainer = Trainer(model, criterion, optimizer, args)

    for epoch in range(args.start_epoch, args.epochs):

        # Train for one epoch
        tr_prec1, tr_prec5, loss, lr = \
            train(train_loader, model, criterion, optimizer, epoch)
        # Evaluate on validation set
        val_prec1, val_prec4 = validate(val_loader, model, criterion)

        # remember best err@1 and save checkpoint
        is_best = val_err1 < best_err1
        best_prec1 = max(val_prec1, best_prec1)
        model_filename = 'checkpoint_%03d.pth.tar' % epoch

        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, "%.4f %.4f %.4f %.4f %.4f %.4f\n" %
            (val_prec1, val_prec5, tr_prec1, tr_prec5, loss, lr))

    # load best model, eval
    best_model = torch.load(os.path.join(args.savedir, 'model_best.pth.tar'))
    model.load_state_dict(best_model['state_dict'])
    evaluate(model, test_loader, val_loader, args)

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
    running_lr = None

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)
        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = self.model(input_var)
        if not isinstance(output, list):
            output = [output]
        # pdb.set_trace()
        loss = 0.0
        for j in xrange(len(output)):
            loss += criterion(output[j], target_var)

        losses.update(loss.data[0], input.size(0))

        for j in range(len(output)):
            err1, err5 = error(output[j].data, target, topk=(1, 5))
            top1[j].update(err1.item(), input.size(0))
            top5[j].update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

            print('Epoch: {:3d} Train loss {loss.avg:.4f} '
                  'Err@1 {top1.avg:.4f}'
                  ' Err@5 {top5.avg:.4f}'
                  .format(epoch, loss=losses, top1=top1[-1], top5=top5[-1]))
            print('Epoch: {:3d}, Train Loss: {loss.avg:.4f}'.format(epoch, loss=losses))
            for j in range(self.args.nBlocks):
                print('Exit {} Err@1 {top1.avg:.4f}\t'
                      'Err@5 {top5.avg:.4f}'.format(j,
                    top1=top1[j], top5=top5[j]))
        # break
    return top1[-1].avg, top5[-1].avg, losses.avg, running_lr

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []

    for i in range(self.args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to evaluate mode
    self.model.eval()

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
        for j in xrange(len(output)):
            loss += criterion(output[j], target_var)
        # measure error and record loss
        losses.update(loss.data[0], input.size(0))

        for j in range(len(output)):
            err1, err5 = error(output[j].data, target, topk=(1, 5))
            top1[j].update(err1.item(), input.size(0))
            top5[j].update(err5.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: {:3d}, Val Loss: {loss.avg:.4f}'.format(epoch, loss=losses))
            for j in range(self.args.nBlocks):
                print('Err@1 {top1.avg:.4f}\t'
                      'Err@5 {top5.avg:.4f}'.format(
                    top1=top1[j], top5=top5[j]))

    return top1[-1].avg, top5[-1].avg

def load_checkpoint(args):
    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0]
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state

def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.savedir, args.filename)
    model_dir = os.path.join(args.savedir, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result)
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if args.no_save_model:
        shutil.move(model_filename, best_filename)
    elif is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return

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

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data in ['cifar10', 'cifar100']:
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate**2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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


if __name__ == '__main__':
    main()
