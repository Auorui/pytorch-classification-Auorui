from torchvision.models.swin_transformer import SwinTransformer, SwinTransformerBlockV2, PatchMergingV2


def swin_t(num_classes):
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        num_classes=num_classes
    )


def swin_s(num_classes):
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.3,
        num_classes=num_classes
    )



def swin_b(num_classes):
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        num_classes=num_classes
    )



def swin_v2_t(num_classes):
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        num_classes=num_classes,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
    )

def swin_v2_s(num_classes):
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        num_classes=num_classes,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
    )

def swin_v2_b(num_classes):
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        num_classes=num_classes,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
    )


if __name__=='__main__':
    import torch
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = swin_v2_s(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))


