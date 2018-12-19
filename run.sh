export CUDA_VISIBLE_DEVICCES=4

python3 main.py --data cifar10 \
--save /ssd3/zhangh/results/check_msdnet_cifar10/ \
--data_root /ssd3/zhangh/dataset/cifar10/ \
--gpu 1 \
--arch msdnet \
--batch-size 64 \
--epochs 300 \
--nBlocks 10 \
--stepmode even \
--step 2 \
--base 4 \
-j 16
