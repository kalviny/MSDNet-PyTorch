# MSDNet-PyTorch

This repository contains the PyTorch implementation of the paper [Multi-Scale Dense Networks for Resource Efficient Image Classification](https://arxiv.org/pdf/1703.09844.pdf)

Citation:

    @inproceedings{huang2018multi,
        title={Multi-scale dense networks for resource efficient image classification},
        author={Huang, Gao and Chen, Danlu and Li, Tianhong and Wu, Felix and van der Maaten, Laurens and Weinberger, Kilian Q},
        journal={ICLR},
        year={2018}
    }

## Dependencies:

+ Python3
+ PyTorch >= 0.4.0

## Network Configurations

#### Train an MSDNet (block=7) on CIFAR-100 for *anytime prediction*: 

```
python3 main.py --data-root /PATH/TO/CIFAR100 --data cifar100 --save /PATH/TO/SAVE \
                --arch msdnet --batch-size 64 --epochs 300 --nBlocks 7 \
                --stepmode even --step 2 --base 4 --nChannels 16 \
                -j 16
```

#### Train an MSDNet (block=5) on CIFAR-100 for *efficient batch computation*:

```
python3 main.py --data-root /PATH/TO/CIFAR100 --data cifar100 --save /PATH/TO/SAVE \
                --arch msdnet --batch-size 64 --epochs 300 --nBlocks 5 \
                --stepmode lin_grow --step 1 --base 1 --nChannels 16 --use-valid \
                -j 16
```

#### Train an MSDNet (block=5, step=4) on ImageNet:

```

python3 main.py --data-root /PATH/TO/ImageNet --data ImageNet --save /PATH/TO/SAVE \
                --arch msdnet --batch-size 256 --epochs 90 --nBlocks 5 \
                --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 \
                --grFactor 1-2-4-4 --bnFactor 1-2-4-4 \
                --use-valid --gpu 0,1,2,3 -j 16 \
```

## Pre-trained MSDNet Models on ImageNet
1. [Download](https://www.dropbox.com/sh/7p758wfcq4wm6lf/AACU4hFtV1_4UQavexrsSs1Ba?dl=0) pretrained models and validation indeces on ImageNet.
2. Test script:
```
python3 main.py --data-root /PATH/TO/ImageNet --data ImageNet --save /PATH/TO/SAVE \
                --arch msdnet --batch-size 256 --epochs 90 --nBlocks 5 \
                --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 \
                --grFactor 1-2-4-4 --bnFactor 1-2-4-4 \
                --evalmode dynamic --evaluate-from /PATH/TO/CHECKPOINT/ \
                --use-valid --gpu 0,1,2,3 -j 16 \
```
   

## Acknowledgments

We would like to take immense thanks to [Danlu Chen](https://taineleau.me/), for providing us the prime version of codes.
