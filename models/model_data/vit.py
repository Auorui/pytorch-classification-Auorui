from torchvision.models.vision_transformer import VisionTransformer



def vit_b_16(num_classes) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=num_classes
    )



def vit_b_32(num_classes) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=num_classes
    )

def vit_l_16(num_classes) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        num_classes=num_classes
    )


def vit_l_32(num_classes) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        num_classes=num_classes
    )


def vit_h_14(num_classes) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        num_classes=num_classes
    )

if __name__=='__main__':
    import torch
    from torchinfo import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = vit_b_32(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    summary(net, (1, 3, 224, 224))