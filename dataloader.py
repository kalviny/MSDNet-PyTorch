import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
from utils import MyRandomSizedCrop
from imagnet_loader import ImageFolder


def get_dataloader(data, config_of_data, splits=['train', 'val', 'test'],
                   aug=True, use_validset=True, data_root='data', batch_size=64,
                   normalized=True, augmentation=0.08, resume=False,
                   num_workers=7, save=None, **kwargs):
    train_loader, val_loader, test_loader = None, None, None
    if data.find('cifar10') >= 0:
        print('loading ' + data)
        print(config_of_data)
        if data.find('cifar100') >= 0:
            d_func = dset.CIFAR100
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                             std=[0.2675, 0.2565, 0.2761])
        else:
            d_func = dset.CIFAR10
            normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                             std=[0.2471, 0.2435, 0.2616])
        if config_of_data['augmentation']:
            print('with data augmentation')
            aug_trans = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            aug_trans = []
        common_trans = [transforms.ToTensor()]
        if normalized:
            print('dataset is normalized')
            common_trans.append(normalize)
        train_compose = transforms.Compose(aug_trans + common_trans)
        test_compose = transforms.Compose(common_trans)

        if use_validset:
            # uses last 5000 images of the original training split as the
            # validation set
            train_set_index = None
            if 'train' in splits:

                train_set = d_func(data_root, train=True, transform=train_compose,
                                   download=True)

                train_set_index = torch.randperm(len(train_set))

                train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        train_set_index[0:45000]),
                    num_workers=num_workers, pin_memory=True)
            if 'val' in splits:
                val_set = d_func(data_root, train=True, transform=test_compose)
                val_loader = torch.utils.data.DataLoader(
                    val_set, batch_size=batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        train_set_index[-5000:]),
                    num_workers=num_workers, pin_memory=True)

            if 'test' in splits:
                test_set = d_func(data_root, train=False, transform=test_compose)
                test_loader = torch.utils.data.DataLoader(
                    test_set, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
        else:
            if 'train' in splits:
                train_set = d_func(data_root, train=True, transform=train_compose,
                                   download=True)
                train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
            if 'val' in splits or 'test' in splits:
                test_set = d_func(data_root, train=False, transform=test_compose)
                test_loader = torch.utils.data.DataLoader(
                    test_set, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
                val_loader = test_loader

    elif data == 'imagenet':
        print('loading ' + data)
        print(config_of_data)

        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_set = ImageFolder(traindir, transforms.Compose([
            MyRandomSizedCrop(224, augmentation=augmentation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

        test_set = ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

        if use_validset:
            train_set_index = torch.randperm(len(train_set))
            # used the last 50000 image as a validation set
            if 'train' in splits:
                train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        train_set_index[0:-50000]),
                    num_workers=num_workers, pin_memory=True)
            # torch.save(train_set.get_image_list(), os.path.join(save, 'trainset_indices.pth'))

            if 'val' in splits:
                val_set = ImageFolder(traindir, transforms.Compose([
                    transforms.Scale(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
                # if resume:
                #     val_set.set_image_list = train_set.get_image_list()

                val_loader = torch.utils.data.DataLoader(
                    val_set, batch_size=batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        train_set_index[-50000:]),
                    num_workers=num_workers, pin_memory=True)

            if 'test' in splits:
                test_loader = torch.utils.data.DataLoader(
                    test_set,
                    batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
        else:
            if 'train' in splits:
                train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
            if 'val' in splits or 'test' in splits:
                test_loader = torch.utils.data.DataLoader(
                    test_set,
                    batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
                val_loader = test_loader

    else:
        raise NotImplemented
    return train_loader, val_loader, test_loader
