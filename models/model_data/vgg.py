from torchvision.models.vgg import VGG, make_layers, cfgs

def vgg11_bn(num_classes) -> VGG:
    model = VGG(make_layers(cfgs['A'], batch_norm=True), num_classes=num_classes)
    return model


def vgg13_bn(num_classes) -> VGG:
    model = VGG(make_layers(cfgs['B'], batch_norm=True), num_classes=num_classes)
    return model


def vgg16_bn(num_classes) -> VGG:
    model = VGG(make_layers(cfgs['D'], batch_norm=True), num_classes=num_classes)
    return model


def vgg19_bn(num_classes) -> VGG:
    model = VGG(make_layers(cfgs['E'], batch_norm=True), num_classes=num_classes)
    return model


if __name__=='__main__':
    import torch
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = vgg16_bn(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
