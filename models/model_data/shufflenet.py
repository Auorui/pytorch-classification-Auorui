from torchvision.models.shufflenetv2 import ShuffleNetV2



def shufflenet_v2_x0_5(num_classes):
    return ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], num_classes=num_classes)


def shufflenet_v2_x1_0(num_classes):
    return ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], num_classes=num_classes)


def shufflenet_v2_x1_5(num_classes):
    return ShuffleNetV2([4, 8, 4], [24, 176, 352, 704, 1024], num_classes=num_classes)


def shufflenet_v2_x2_0(num_classes):
    return ShuffleNetV2([4, 8, 4], [24, 244, 488, 976, 2048], num_classes=num_classes)


if __name__=='__main__':
    import torch
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = shufflenet_v2_x1_5(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))







































