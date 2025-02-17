from torchvision.models.mnasnet import MNASNet



def mnasnet0_5(num_classes) -> MNASNet:
    r"""MNASNet with depth multiplier of 0.5 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    """
    model = MNASNet(0.5, num_classes=num_classes)
    return model


def mnasnet0_75(num_classes) -> MNASNet:
    r"""MNASNet with depth multiplier of 0.75 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.75, num_classes=num_classes)
    return model


def mnasnet1_0(num_classes) -> MNASNet:
    r"""MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    """
    model = MNASNet(1.0, num_classes=num_classes)
    return model


def mnasnet1_3(num_classes) -> MNASNet:
    r"""MNASNet with depth multiplier of 1.3 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    """
    model = MNASNet(1.3, num_classes=num_classes)
    return model


if __name__=='__main__':
    import torch
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = mnasnet0_75(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
