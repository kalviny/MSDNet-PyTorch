import torch.nn as nn
import torch
import math
import pdb

class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1,
                 padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck,
                 bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=1, stride=1, padding=0))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))

        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=1, padding=1))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=2, padding=1))
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut))
        layer.append(nn.ReLU(True))

        self.net = nn.Sequential(*layer)

    def forward(self, x):
        # print(self.net, x.size())
        # pdb.set_trace()
        return self.net(x)


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, nOut // 2, 'down',
                                bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, 'normal',
                                   bottleneck, bnWidth2)

    def forward(self, x):
        # print(self.conv_down, self.conv_normal, '\n========')
        # pdb.set_trace()
        res = [x[1],
               self.conv_down(x[0]),
               self.conv_normal(x[1])]
        # print(res[0].size(), res[1].size(), res[2].size())
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal',
                                   bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0],
               self.conv_normal(x[0])]

        return torch.cat(res, dim=1)


class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, growthRate, args):
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        layers = []
        if args.data.startswith('cifar'):
            layers.append(ConvBasic(nIn, nOut * args.grFactor[0],
                                         kernel=3, stride=1, padding=1))
        elif args.data == 'imagenet':
            raise NotImplementedError

        nIn = nOut * args.grFactor[0]
        nIns = [nIn]
        self.nScales = args.nScales

        # deal with the fisrt three "layer", which is a triangle
        for level in range(self.nScales):
            for i in range(level, self.nScales):
                if level == 0:
                    if i == 0:
                        continue
                    layers.append(ConvBasic(nIn, nOut * args.grFactor[i],
                                                 kernel=3, stride=2, padding=1))
                    nIn = nOut * args.grFactor[i] # + growthRate * args.grFactor[i]
                    nIns.append(nIn)
                else:
                    layers.append(
                            ConvDownNormal(old_nIns[i - level], old_nIns[i - level + 1],
                                           growthRate * args.grFactor[i],
                                           args.bottleneck, args.bnFactor[i - 1],
                                           args.bnFactor[i]),
                    )
                    nIns.append(old_nIns[i - level + 1] + growthRate * args.grFactor[i])
                    # print(level, i, nIns[-1])

            self.layers.append(nn.ModuleList(layers))
            layers = []
            print(level, nIns)
            old_nIns = nIns
            nIns = []

    def forward(self, x):
        res = []
        out = []
        for level in range(self.nScales):
            if level < 1:
                for i in range(len(self.layers[level])):
                    print(self.layers[level][i], x.size())
                    x = self.layers[level][i](x)
                    res.append(x)
                    if i == 0:
                        out.append(x)
                        print(level, x.size())

            else:
                # pdb.set_trace()
                for i in range(len(self.layers[level])):
                    inp = [pre_res[i], pre_res[i + 1]]
                    x = self.layers[level][i](inp)
                    res.append(x)
                    if i == 0:
                        out.append(x)
                        print(level, x.size())

            pre_res = res
            res = []

        return out


class MSDNLayer(nn.Module):
    def __init__(self, nIn, growthRate, args, inScales=None, outScales=None):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.growthRate = growthRate
        self.args = args
        self.inScales = inScales if inScales is not None else args.nScales
        self.outScales = outScales if outScales is not None else args.nScales
        self.nScales = args.nScales
        self.discard = self.inScales - self.outScales

        self.offset = self.nScales - self.outScales
        self.layers = nn.ModuleList()
        if self.discard > 0:
            # print("discard", nIn)
            nIn1 = (nIn + (self.offset - 1) * growthRate) * args.grFactor[self.offset - 1]
            nIn2 = (nIn + self.offset * growthRate) * args.grFactor[self.offset]
            _nOut = growthRate * args.grFactor[self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[self.offset - 1],
                                              args.bnFactor[self.offset]))
        else:
            self.layers.append(ConvNormal((nIn + self.offset * growthRate) * args.grFactor[self.offset],
                                          growthRate * args.grFactor[self.offset],
                                          args.bottleneck,
                                          args.bnFactor[self.offset]))

        for i in range(self.offset + 1, self.nScales):
            nIn1 = (nIn + (self.offset + i - 1) * growthRate) * args.grFactor[i - 1]
            nIn2 = (nIn + (self.offset + i) * growthRate) * args.grFactor[i]
            _nOut = growthRate * args.grFactor[i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[i - 1],
                                              args.bnFactor[i]))

    def forward(self, x):
        print(len(x), self.offset, self.discard, self.inScales, self.outScales)
        # construct input
        if self.discard > 0:
            inp = []
            # for i in range(self.offset, self.outScales + self.offset):
            for i in range(1, self.outScales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            # pdb.set_trace()
            for i in range(1, self.outScales):
                inp.append([x[i - 1], x[i]])

        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))
            # print(res[-1].size())
        pdb.set_trace()

        return res


class ParallelModule(nn.Module):
    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return res


class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_class):
        super(ClassifierModule, self).__init__()
        self.m = m
        self.linear = nn.Linear(channel, num_class)

    def forward(self, x):
        # pdb.set_trace()
        res = self.m(x[-1])
        res = res.view(res.size(0), -1)
        return self.linear(res)


class MSDNet(nn.Module):
    def __init__(self, nBlocks, nChannels, args):
        super(MSDNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = nBlocks
        self.steps = [args.base]
        # todo: how many block?
        n_layers_all, n_layer_curr = args.base, 0
        for i in range(1, nBlocks):
            self.steps.append(args.step if args.stepmode == 'even'
                             else args.step * i + 1)
            n_layers_all += self.steps[-1]

        print("building network of steps: ")
        print(self.steps)

        nIn = nChannels
        for i in range(nBlocks):
            print(' ----------------------- Block {} '
                  '-----------------------'.format(i + 1))
            m, nIn = \
                self._build_block(nIn, args, self.steps[i],
                                  n_layers_all, n_layer_curr)
            self.blocks.append(m)
            n_layer_curr += self.steps[i]

            if args.data.startswith('cifar10'):
                self.classifier.append(
                    self._build_classifier_cifar((nIn + (args.nScales - 1) * args.growthRate) * args.grFactor[-1], 10))
            else:
                raise NotImplementedError

    def _build_block(self, nIn, args, step, n_layer_all, n_layer_curr):
        if n_layer_curr == 0:
            layers = [MSDNFirstLayer(3, nIn, args.growthRate, args)]
            n_layer_curr = args.nScales - 1
        else:
            layers = []
        for i in range(step):
            n_layer_curr += 1
            if args.prune == 'min':
                inScales = min(args.nScales, n_layer_all - n_layer_curr + 2)
                outScales = min(args.nScales, n_layer_all - n_layer_curr + 1)
            elif args.prune == 'max':
                interval = math.ceil(n_layer_all / args.nScales)
                inScales = args.nScales - math.floor((max(0, n_layer_curr - 2)) / interval)
                outScales = args.nScales - math.floor((n_layer_curr - 1) / interval)
            else:
                raise ValueError
            print('|\t\tinScales {} outScales {}\t\t\t|'.format(inScales, outScales))

            layers.append(MSDNLayer(nIn, args.growthRate, args, inScales, outScales))

            nIn += args.growthRate

            if args.prune == 'max' and inScales > outScales and \
                    args.reduction > 0:
                offset = args.nScales - outScales
                layers.append(
                    self._build_transition(nIn, math.floor(args.reduction * nIn),
                                           outScales, offset, args))
                nIn = math.floor(args.reduction * nIn)
                print('|\t\tTransition layer inserted!\t\t|')
            elif args.prune == 'min' and args.reduction > 0 and \
                (n_layer_curr == math.floor(n_layer_all / 3)) or \
                n_layer_curr == math.floor(2 * n_layer_all / 3):
                offset = args.nScales - outScales
                layers.append(self._build_transition(nIn, math.floor(args.reduction * nIn),
                                                     outScales, offset, args))

                nIn = math.floor(args.reduction * nIn)
                print('|\t\tTransition layer inserted!\t\t|')

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, offset, args):
        net = []
        for i in range(outScales):
            net.append(ConvBasic(nIn * args.grFactor[offset + i],
                                 nOut * args.grFactor[offset + i],
                                 kernel=1, stride=1, padding=0))

        return ParallelModule(net)

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1),
            ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2),
        )
        return ClassifierModule(conv, interChannels2, num_classes)

    def forward(self, x):
        res = []
        for i in range(self.nBlocks):
            print("block", i)
            x = self.blocks[i](x)
            res.append(self.classifier[i](x))
        # return a tuple
        return res


def createModel(args):
    print('Create MSDNet{}-{:d} for {}'.format(args.nBlocks, args.step, args.data))
    return MSDNet(args.nBlocks, args.nChannels, args)
