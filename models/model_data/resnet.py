from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck



def resnet18(num_classes) -> ResNet:
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def resnet34(num_classes) -> ResNet:
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model

def resnet50(num_classes) -> ResNet:
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model

def resnet101(num_classes) -> ResNet:
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return model

def resnet152(num_classes) -> ResNet:
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    return model

if __name__=='__main__':
    import torch
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = resnet50(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))