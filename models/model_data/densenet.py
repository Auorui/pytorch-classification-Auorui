from torchvision.models.densenet import DenseNet


def densenet121(num_classes) -> DenseNet:
    return DenseNet(32, (6, 12, 24, 16), 64, num_classes=num_classes)


def densenet161(num_classes) -> DenseNet:
    return DenseNet(48, (6, 12, 36, 24), 96, num_classes=num_classes)


def densenet169(num_classes) -> DenseNet:
    return DenseNet(32, (6, 12, 32, 32), 64, num_classes=num_classes)


def densenet201(num_classes) -> DenseNet:
    return DenseNet(32, (6, 12, 48, 32), 64, num_classes=num_classes)


if __name__=='__main__':
    import torch
    import torchsummaryX
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = densenet161(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummaryX.summary(net, x=input)