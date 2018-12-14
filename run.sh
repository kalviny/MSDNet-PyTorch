
/home/zhangfeihu/anaconda2/bin/python main.py --data cifar10 \
--save /ssd3/zhangh/results/msdnet_cifar10/ \
--arch msdnet \
-batchSize 64 \
-nEpochs 300 \
-nBlocks 10 \
-stepmode even \
-step 2 \
-base 4 \
-j 16 
