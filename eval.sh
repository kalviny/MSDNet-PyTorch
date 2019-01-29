# method='check_cifar10+_full_version_init_once_loader_with_init'
# method='check_cifar10+_full_version_with_init'
method='cifar10'
### cifar10 checkpoint_289.pth.tar

python3 main.py --data-root /ssd3/zhangh/dataset/cifar10/ \
--save /ssd3/zhangh/results/$method/ \
--data cifar10 \
--gpu 7 \
--arch msdnet \
--batch-size 64 \
--epochs 300 \
--nBlocks 10 \
--stepmode even \
--step 2 \
--base 4 \
--nChannels 16 \
--use-valid \
--evaluate dynamic \
--evaluate-from /ssd3/zhangh/results/$method/save_models/model_best.pth.tar \
-j 16
