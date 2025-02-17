from torchvision.models.efficientnet import EfficientNet, _efficientnet_conf
from functools import partial
import torch.nn as nn


def efficientnet_b0(num_classes):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    return EfficientNet(
        inverted_residual_setting, 0.2, last_channel=last_channel, num_classes=num_classes
    )


def efficientnet_b1(num_classes):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1)
    return EfficientNet(
        inverted_residual_setting, 0.2, last_channel=last_channel, num_classes=num_classes
    )


def efficientnet_b2(num_classes):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2)
    return EfficientNet(
        inverted_residual_setting, 0.3, last_channel=last_channel, num_classes=num_classes
    )


def efficientnet_b3(num_classes):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4)
    return EfficientNet(
        inverted_residual_setting, 0.3, last_channel=last_channel, num_classes=num_classes
    )


def efficientnet_b4(num_classes):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8)
    return EfficientNet(
        inverted_residual_setting, 0.4, last_channel=last_channel, num_classes=num_classes
    )


def efficientnet_b5(num_classes):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2)
    return EfficientNet(
        inverted_residual_setting, 0.4, last_channel=last_channel,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), num_classes=num_classes
    )


def efficientnet_b6(num_classes):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b6", width_mult=1.8, depth_mult=2.6)
    return EfficientNet(
        inverted_residual_setting, 0.5, last_channel=last_channel,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), num_classes=num_classes
    )


def efficientnet_b7(num_classes):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1)
    return EfficientNet(
        inverted_residual_setting, 0.5, last_channel=last_channel,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), num_classes=num_classes
    )


def efficientnet_v2_s(num_classes):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return EfficientNet(
        inverted_residual_setting, 0.2, last_channel=last_channel,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03), num_classes=num_classes
    )



def efficientnet_v2_m(num_classes):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
    return EfficientNet(
        inverted_residual_setting, 0.3, last_channel=last_channel,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03), num_classes=num_classes
    )


def efficientnet_v2_l(num_classes):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
    return EfficientNet(
        inverted_residual_setting, 0.4, last_channel=last_channel,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03), num_classes=num_classes
    )

if __name__=='__main__':
    import torch
    from torchsummary import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = efficientnet_v2_s(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    summary(net, (3, 224, 224))




