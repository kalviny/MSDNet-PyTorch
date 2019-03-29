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

#### Train an MSDNet (block=7) for *anytime prediction*: 

```
python3 main.py --data-root /PATH/TO/CIFAR100 --data cifar100 --save /PATH/TO/SAVE \
                --arch msdnet --batch-size 64 --epochs 300 --nBlocks 7 \
                --stepmode even --step 2 --base 4 --nChannels 16 \
                -j 16
```

#### Train an MSDNet (block=5) for *efficient batch computation*:

```
python3 main.py --data-root /PATH/TO/CIFAR100 --data cifar100 --save /PATH/TO/SAVE \
                --arch msdnet --batch-size 64 --epochs 300 --nBlocks 5 \
                --stepmode lin_grow --step 1 --base 1 --nChannels 16 --use-valid \
                -j 16
```

## Acknowledgments

We would like to take immense thanks to Danlu Chen, for providing us the prime version of codes and carrying us throughout this project.