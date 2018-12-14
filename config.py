# this is used for storing configurations of datasets & models

datasets = {
    'cifar10': {
        'num_classes': 10,
        'augmentation': False,
    },
    'imagenet': {
        'num_classes': 1000,
        'augmentation': True, # by default, ImageNet use augmentation
    },
    'cifar10+': {
        'num_classes': 10,
        'augmentation': True,
    },
    'cifar100': {
        'num_classes': 100,
        'augmentation': False,
    },
    'cifar100+': {
        'num_classes': 100,
        'augmentation': True,
    },
}
