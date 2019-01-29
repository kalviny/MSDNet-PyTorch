eval script:

```
python main.py --data cifar10+ --arch msdnet --epochs 300 --nBlocks 10 --stepmode even --step 2 --base 4 --print-freq 100 --evaluate evaluate --resume save/cifar10/model_best.pth.tar
```


training script:

```
CUDA_VISIBLE_DEVICES=3 python main.py --data cifar10+ --arch msdnet --save save/cifar10 --epochs 300 --nBlocks 10 --stepmode even --step 2 --base 4 --print-freq 100
```

------

DenseNet_MC for imagenet, please run:

```bash
python3 main.py --data imagenet --arch densenet_mc --save save/imagenet_densenet121_mc --epochs 90 --data_root /path/to/imagenet -b 256
```

ResNet_MC for imagenet, please run:

```bash
python3 main.py --data imagenet --arch resnet_mc --save save/imagenet_resnet50_mc --epochs 90 --data_root /path/to/imagenet -b 256
```

To-do list:

- [ ] check new dataloader

- [x] Dynamic Evaluation

- [ ] ImageNet training code

- [ ] args name adjustment
