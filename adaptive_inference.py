from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import time
import math
import warnings

import torch


def dynamic_evaluate(model, test_loader, val_loader, args):
    print('--------------------------------')
    print(len(test_loader), len(val_loader))
    print('--------------------------------')

    tester = Tester(model, args)
    if os.path.exists(os.path.join(args.save, 'logits_greedy.pth')):
        val_pred_greedy, val_target, test_pred_greedy, test_target = \
            torch.load(os.path.join(args.save, 'logits_greedy.pth'))
    else:
        if os.path.exists(os.path.join(args.save, 'logits_single.pth')):
            val_pred_single, val_target, test_pred_single, test_target = \
                torch.load(os.path.join(args.save, 'logits_single.pth'))
        else:
            val_pred_single, val_target = tester.calc_logit(val_loader)
            test_pred_single, test_target = tester.calc_logit(test_loader)
            torch.save((val_pred_single, val_target, test_pred_single, test_target),
                       os.path.join(args.save, 'logits_single.pth'))

        # early_exit
        best_ensemble_scheme = tester.find_greedy_ensemble(val_pred_single, val_target)
        val_pred_greedy, val_res = \
            tester.early_exit(val_pred_single, val_target, best_ensemble_scheme)
        test_pred_greedy, test_res = \
            tester.early_exit(test_pred_single, test_target, best_ensemble_scheme)

        torch.save((val_pred_greedy, val_target,
                    test_pred_greedy, test_target),
                   os.path.join(args.save, 'logits_greedy.pth'))
        torch.save((best_ensemble_scheme, test_res), 'early_exit_res.pth')

    # load flops
    flops = torch.load(os.path.join(args.save, 'flop.pth'))
    print(flops)
    
    # dynamic evaluation
    with open(os.path.join(args.save, 'dynamic.txt'), 'w') as fout:
        for p in range(1, 41):
            print("*********************")
            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
            # probs = [math.exp(math.log(p) * i) for i in in range(1, tester.args.nBlocks + 1)] # geometric distribution
            probs = torch.exp(torch.log(_p) * torch.range(1, args.nBlocks))
            probs /= probs.sum()
            acc_val, _, T = tester.dynamic_eval_find_threshold(
                val_pred_greedy, val_target, probs, flops)
            acc_test, exp_flops = tester.dynamic_eval_with_threshold(
                test_pred_greedy, test_target, flops, T)
            print('valid acc: {:.3f}, test acc: {:.3f}, test flops: {:.2f}M'.format(acc_val, acc_test, exp_flops / 1e6))
            fout.write('{}\t{}\n'.format(acc_test, exp_flops.item()))

class Tester(object):
    def __init__(self, model, args=None):
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1).cuda()
        self.args = args

    def calc_logit(self, val_loader, silence=False):
        self.model.eval()
        print("calculating logit")
        # logits = [[] for _ in range(self.args.nBlocks)]
        logits = []
        for _ in range(self.args.nBlocks): logits.append([])
        targets = []

        for i, (input, target) in enumerate(val_loader):

            targets.append(target)

            input = input.cuda()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
            # compute output
            output = self.model(input_var)
            if not isinstance(output, list):
                output = [output]


            for j in range(self.args.nBlocks):
                _t = self.softmax(output[j])
                # print(_t.cpu().data)
                logits[j].append(self.softmax(output[j]).cpu().data)

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

            best_err = best_err.item()

            for start_block_num in range(num_block - 1):
                accum_pred -= logits[start_block_num].data
                tmp_err, _ = error(accum_pred / (num_block + 1), targets, topk=(1, 5))
                tmp_err = tmp_err.item()
                # print(start_block_num + 1, num_block, tmp_err[0])
                if tmp_err < best_err:
                    best_err = tmp_err
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

            err_greedy.append(best_err.item())
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

            # print("classifier #",k + 1, "output # samples", out_n, "thresholds: ", T[k])
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

        return acc * 1.0 / n, expected_flops, T

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

        return acc * 1.0 / n, expected_flops

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
