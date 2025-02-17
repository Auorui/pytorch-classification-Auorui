from torchvision.models.regnet import RegNet, BlockParams
import torch.nn as nn
from functools import partial

def regnet_y_400mf(num_classes):
    params = BlockParams.from_init_params(
        depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))

def regnet_y_800mf(num_classes):
    params = BlockParams.from_init_params(
        depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))

def regnet_y_1_6gf(num_classes):
    params = BlockParams.from_init_params(
        depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))

def regnet_y_3_2gf(num_classes):
    params = BlockParams.from_init_params(
        depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))


def regnet_y_8gf(num_classes):
    params = BlockParams.from_init_params(
        depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))


def regnet_y_16gf(num_classes):
    params = BlockParams.from_init_params(
        depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, se_ratio=0.25
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))

def regnet_y_32gf(num_classes):
    params = BlockParams.from_init_params(
        depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, se_ratio=0.25
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))

def regnet_y_128gf(num_classes):
    params = BlockParams.from_init_params(
        depth=27, w_0=456, w_a=160.83, w_m=2.52, group_width=264, se_ratio=0.25
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))

def regnet_x_400mf(num_classes):
    params = BlockParams.from_init_params(
        depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))


def regnet_x_800mf(num_classes):
    params = BlockParams.from_init_params(
        depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))


def regnet_x_1_6gf(num_classes):
    params = BlockParams.from_init_params(
        depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))


def regnet_x_3_2gf(num_classes):
    params = BlockParams.from_init_params(
        depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))


def regnet_x_8gf(num_classes):
    params = BlockParams.from_init_params(
        depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))

def regnet_x_16gf(num_classes):
    params = BlockParams.from_init_params(
        depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))


def regnet_x_32gf(num_classes):
    params = BlockParams.from_init_params(
        depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168
    )
    return RegNet(params, num_classes=num_classes,
                  norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))

if __name__=='__main__':
    import torch
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = regnet_x_8gf(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
