from torchvision.models.resnet import resnet50


def createModel(depth, data, num_classes, death_mode='none', death_rate=0.5,
                **kwargs):
    print('Create ResNet-50 for {}'.format(data))
    return resnet50()
