eval script:

```
python main.py --data cifar10+ --arch msdnet --epochs 300 --nBlocks 10 --stepmode even --step 2 --base 4 --print-freq 100 --evaluate evaluate --resume save/cifar10/model_best.pth.tar
```


training script:

```
CUDA_VISIBLE_DEVICES=3 python main.py --data cifar10+ --arch msdnet --save save/cifar10 --epochs 300 --nBlocks 10 --stepmode even --step 2 --base 4 --print-freq 100
```

Current Implementation, reprodcued the result of Lua Torch

```
Epoch: 300, Val Loss: 2.9238
Err@1 9.1800    Err@5 0.3400
Err@1 7.6400    Err@5 0.3000
Err@1 6.8400    Err@5 0.2200
Err@1 6.1800    Err@5 0.2800
Err@1 6.0800    Err@5 0.1800
Err@1 5.9000    Err@5 0.2200
Err@1 5.8000    Err@5 0.2800
Err@1 5.7600    Err@5 0.2800
Err@1 5.5800    Err@5 0.2200
Err@1 5.6400    Err@5 0.2600
Best val_err1: 5.5800 at epoch 255
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

-[ ] Dynamic Evaluation

-[ ] ImageNet training code

-[ ] args name adjustment
