from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
from utils import AverageMeter, adjust_learning_rate, error
import time
import math

import pdb

class Trainer(object):
    def __init__(self, model, criterion=None, optimizer=None, args=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args

    def train(self, train_loader, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1, top5 = [], []
        for i in range(self.args.nBlocks):
            top1.append(AverageMeter())
            top5.append(AverageMeter())

        # switch to train mode
        self.model.train()

        lr = adjust_learning_rate(self.optimizer, self.args.lr,
                                  self.args.decay_rate, epoch,
                                  self.args.epochs,
                                  self.args)  # TODO: add custom
        print('Epoch {:3d} lr = {:.6e}'.format(epoch, lr))

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = self.model(input_var)
            if not isinstance(output, list):
                output = [output]
            # pdb.set_trace()
            loss = self.criterion(output[0], target_var)

            for j in range(1, len(output)):
                loss += self.criterion(output[j], target_var)
            # measure error and record loss
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

            if self.args.print_freq > 0 and \
                    (i + 1) % self.args.print_freq == 0:
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
        return losses.avg, top1[-1].avg, top5[-1].avg, lr

    def test(self, val_loader, epoch, silence=False):
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
            output = self.model(input_var)
            if not isinstance(output, list):
                output = [output]
            loss = self.criterion(output[0], target_var)
            for j in range(1, len(output)):
                loss += self.criterion(output[j], target_var)

            # measure error and record loss
            losses.update(loss.data[0], input.size(0))

            for j in range(len(output)):
                err1, err5 = error(output[j].data, target, topk=(1, 5))
                top1[j].update(err1.item(), input.size(0))
                top5[j].update(err5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # break

        if not silence:
            print('Epoch: {:3d}, Val Loss: {loss.avg:.4f}'.format(epoch, loss=losses))
            for j in range(self.args.nBlocks):
                print('Err@1 {top1.avg:.4f}\t'
                      'Err@5 {top5.avg:.4f}'.format(
                    top1=top1[j], top5=top5[j]))

        return losses.avg, top1[-1].avg, top5[-1].avg


class Tester(object):
    def __init__(self, model, args=None):
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1).cuda()
        self.args = args

    def calc_logit(self, val_loader, silence=False):
        self.model.eval()
        print("calculating logit")
        logits = []
        targets = []
        for i in range(self.args.nBlocks):
            logits.append([])

        for i, (input, target) in enumerate(val_loader):
            targets.append(target)

            input = input.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)

            # compute output
            output = self.model(input_var)
            if not isinstance(output, list):
                output = [output]

            for j in range(self.args.nBlocks):
                tmp = self.softmax(output[j])
                logits[j].append(tmp.cpu())

        for i in range(self.args.nBlocks):
            logits[i] = torch.cat(logits[i], dim=0)

        targets = torch.cat(targets, dim=0)
        print("finished!")

        return logits, targets

    def find_greedy_ensemble(self, logits, targets):
        greedy_scheme = [0]

        for num_block in range(1, len(logits)):
            accum_pred = torch.FloatTensor().resize_(logits[0].size()).zero_()
            for start_block_num in range(num_block):
                accum_pred += logits[start_block_num].data

            best_err, _ = error(accum_pred / (num_block + 1), targets, topk=(1, 5))
            # print(1, num_block, best_err[0])
            greedy_scheme.append(0)

            for start_block_num in range(num_block - 1):
                accum_pred -= logits[start_block_num].data
                tmp_err, _ = error(accum_pred / (num_block + 1), targets, topk=(1, 5))
                # print(start_block_num + 1, num_block, tmp_err[0])
                if tmp_err[0] < best_err[0]:
                    best_err[0] = tmp_err[0]
                    greedy_scheme[-1] = start_block_num + 1

        print("greedy scheme", greedy_scheme)
        return greedy_scheme

    def early_exit(self, logits, targets, greedy_scheme: list):
        print("early exit")
        N = self.args.nBlocks
        logits_greedy = []
        err_greedy = []
        for num_block in range(N):
            if num_block == 0:
                accum_pred = logits[0].data
            else:
                accum_pred = torch.FloatTensor().resize_(logits[0].size()).zero_()
                for j in range(greedy_scheme[num_block], num_block + 1):
                    accum_pred += logits[j].data
                accum_pred /= (num_block - greedy_scheme[num_block] + 1) # ensemble of logits
            best_err, _ = error(accum_pred, targets, topk=(1, 5))
            # print(num_block, best_err[0])

            err_greedy.append(best_err[0])
            logits_greedy.append(accum_pred.view(1, accum_pred.size(0), accum_pred.size(1)))

        logits_greedy = torch.cat(logits_greedy, dim=0)
        return logits_greedy, err_greedy

    def dynamic_eval_find_threshold(self, logits, targets, p, flops, silence=True):
        # m: num of model (exits)
        # n: num of samples
        m, n, _ = logits.shape
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False) # take max logits as confidence
        _, sorted_idx = max_preds.sort(dim=1, descending=True)

        filtered = torch.zeros(n) # samples that has been filtered out
        T = torch.Tensor(m).fill_(1e8) # threshold for each models (exits)

        for k in range(m - 1):
            acc, count = 0, 0
            out_n = math.floor(n * p[k])
            for i in range(n):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx]
                        break

            print("classifier #",k + 1, "output # samples", out_n, "thresholds: ", T[k])
            filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

        T[m - 1] = -1e8  # accept all the samples at last exit

        acc_rec, exp = torch.zeros(m), torch.zeros(m)
        acc, expected_flops = 0, 0
        for i in range(n):
            gold_label = targets[i]
            for k in range(m):
                if max_preds[k][i] >= T[k]: # this sample should exit at k
                    if gold_label == argmax_preds[k][i]:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break

        acc_all = 0
        for k in range(m):
            tmp = exp[k] / n
            expected_flops += tmp * flops[k]
            acc_all += acc_rec[k]
            if not silence:
                try:
                    print("acc of each:", acc_rec[k] / exp[k], 'accummulate:', acc_rec[k],
                          'T: ', T[k])
                except ZeroDivisionError:
                    pass

        print("acc_all", acc_all / n)
        return acc / n, expected_flops, T

    def dynamic_eval_with_threshold(self, logits, targets, flops, T, silence=True):
        m, n, _ = logits.shape
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False) # take max logits as confidence

        acc_rec, exp = torch.zeros(m), torch.zeros(m)
        acc, expected_flops = 0, 0
        for i in range(n):
            gold_label = targets[i]
            for k in range(m):
                if max_preds[k][i] >= T[k]: # this sample should exit at k
                    if gold_label == argmax_preds[k][i]:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break

        acc_all = 0
        for k in range(m):
            tmp = exp[k] / n
            expected_flops += tmp * flops[k]
            acc_all += acc_rec[k]
            if not silence:
                try:
                    print("acc of each:", acc_rec[k] / exp[k], 'accummulate:', acc_rec[k],
                          'T: ', T[k])
                except ZeroDivisionError:
                    pass

        print("acc_all", acc_all / n)
        return acc / n, expected_flops
