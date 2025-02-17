import os
from torch.hub import load_state_dict_from_url

model_urls = {
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    "mnasnet0_5": "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
    "mnasnet0_75": "https://download.pytorch.org/models/mnasnet0_75-7090bc5f.pth",
    "mnasnet1_0": "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
    "mnasnet1_3": "https://download.pytorch.org/models/mnasnet1_3-a4c69d6f.pth",
    "vit_b_16": "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
    "vit_b_32": "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
    "vit_l_16": "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
    "vit_l_32": "https://download.pytorch.org/models/vit_l_32-c7638314.pth",
    "vit_h_14": "https://download.pytorch.org/models/vit_h_14_swag-80465313.pth",
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-bac287d4.pth",
    "efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth",
    "efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-b3899882.pth",
    "efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth",
    "efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-1a07897c.pth",
    "efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-24a108a5.pth",
    "efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-c5b4e57e.pth",
    "efficientnet_v2_s": "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
    "efficientnet_v2_m": "https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth",
    "efficientnet_v2_l": "https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth",
    "swin_t": "https://download.pytorch.org/models/swin_t-704ceda3.pth",
    "swin_s": "https://download.pytorch.org/models/swin_s-5e29d889.pth",
    "swin_b": "https://download.pytorch.org/models/swin_b-68c6b09e.pth",
    "swin_v2_t": "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
    "swin_v2_s": "https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth",
    "swin_v2_b": "https://download.pytorch.org/models/swin_v2_b-781e5279.pth",
    "shufflenet_v2_x0_5": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
    "shufflenet_v2_x1_0": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
    "shufflenet_v2_x1_5": "https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth",
    "shufflenet_v2_x2_0": "https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth",
    "regnet_x_8gf": "https://download.pytorch.org/models/regnet_x_8gf-03ceed89.pth",
    "regnet_y_8gf": "https://download.pytorch.org/models/regnet_y_8gf-d0d0e4a8.pth",
    "regnet_x_16gf": "https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth",
    "regnet_y_16gf": "https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth",
    "regnet_x_32gf": "https://download.pytorch.org/models/regnet_x_32gf-9d47f8d0.pth",
    "regnet_y_32gf": "https://download.pytorch.org/models/regnet_y_32gf-4dee3f7a.pth",
    "regnet_y_128gf": "https://download.pytorch.org/models/regnet_y_128gf_swag-c8ce3e52.pth",
    "regnet_x_400mf": "https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth",
    "regnet_y_400mf": "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth",
    "regnet_x_800mf": "https://download.pytorch.org/models/regnet_x_800mf-ad17e45c.pth",
    "regnet_y_800mf": "https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth",
    "regnet_x_1_6gf": "https://download.pytorch.org/models/regnet_x_1_6gf-e3633e7f.pth",
    "regnet_y_1_6gf": "https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth",
    "regnet_x_3_2gf": "https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth",
    "regnet_y_3_2gf": "https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth",
}

def download_from_url(name=None, model_dir=None, progress=True):
    if model_dir is None:
        model_dir = os.getcwd()
    load_state_dict_from_url(model_urls[name], model_dir=model_dir, progress=progress)

if __name__=="__main__":
    name = "mobilenet_v2"
    
    model_dir = "./download_pth"
    download_from_url(name=name, model_dir=model_dir)