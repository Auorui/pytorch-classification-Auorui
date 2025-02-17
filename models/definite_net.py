import torch
import torch.nn as nn
from models.model_data import *
import numpy as np

MODEL_CLASSES = {
    "alexnet": alexnet,
    "vgg11_bn": vgg11_bn,
    "vgg13_bn": vgg13_bn,
    "vgg16_bn": vgg16_bn,
    "vgg19_bn": vgg19_bn,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "google_net": google_net,
    "mobilenet_v2": mobilenet_v2,
    "squeezenet1_0": squeezenet1_0,
    "squeezenet1_1": squeezenet1_1,
    "mnasnet0_5": mnasnet0_5,
    "mnasnet0_75": mnasnet0_75,
    "mnasnet1_0": mnasnet1_0,
    "mnasnet1_3": mnasnet1_3,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "densenet201": densenet201,
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b3": efficientnet_b3,
    "efficientnet_b4": efficientnet_b4,
    "efficientnet_b5": efficientnet_b5,
    "efficientnet_b6": efficientnet_b6,
    "efficientnet_b7": efficientnet_b7,
    "efficientnet_v2_s": efficientnet_v2_s,
    "efficientnet_v2_m": efficientnet_v2_m,
    "efficientnet_v2_l": efficientnet_v2_l,
    "shufflenet_v2_x0_5": shufflenet_v2_x0_5,
    "shufflenet_v2_x1_0": shufflenet_v2_x1_0,
    "shufflenet_v2_x1_5": shufflenet_v2_x1_5,
    "shufflenet_v2_x2_0": shufflenet_v2_x2_0,
    "regnet_x_8gf": regnet_x_8gf,
    "regnet_y_8gf": regnet_y_8gf,
    "regnet_x_16gf": regnet_x_16gf,
    "regnet_y_16gf": regnet_y_16gf,
    "regnet_x_32gf": regnet_x_32gf,
    "regnet_y_32gf": regnet_y_32gf,
    "regnet_y_128gf": regnet_y_128gf,
    "regnet_x_400mf": regnet_x_400mf,
    "regnet_y_400mf": regnet_y_400mf,
    "regnet_x_800mf": regnet_x_800mf,
    "regnet_y_800mf": regnet_y_800mf,
    "regnet_x_1_6gf": regnet_x_1_6gf,
    "regnet_y_1_6gf": regnet_y_1_6gf,
    "regnet_x_3_2gf": regnet_x_3_2gf,
    "regnet_y_3_2gf": regnet_y_3_2gf,
    "vit_b_16": vit_b_16,
    "vit_b_32": vit_b_32,
    "vit_l_16": vit_l_16,
    "vit_l_32": vit_l_32,
    "vit_h_14": vit_h_14,
    "swin_t": swin_t,
    "swin_s": swin_s,
    "swin_b": swin_b,
    "swin_v2_t": swin_v2_t,
    "swin_v2_s": swin_v2_s,
    "swin_v2_b": swin_v2_b
}


def get_networks(name,num_classes,weights=None):
    if name in MODEL_CLASSES:
        model=MODEL_CLASSES[name](num_classes=num_classes)
    else:
        raise ValueError(f"Model {name} not found in MODEL_CLASSES")
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model=nn.DataParallel(model)
    else:
        model=model.to(device)
    if weights != None:
        # 加载权重的部分不需要再检查设备，因为模型已经在正确的设备上
        print(f'\033[34mLoad weights {weights}. to {name}')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weights, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        
        print(f"\033[34mThere are a total of {len(pretrained_dict)} keys.\n"
              f"Successfully loaded {len(load_key)} keys.")
        print("\033[34mFirst 10 loaded keys:")
        for key in load_key[:10]:
            print(f"- {key}")
            
        if no_load_key:
            print("\033[34mThe following keys have mismatched shapes or are not present in the model:")
            for key in no_load_key:
                print(f"- {key}")
        else:
            print("\033[34mAll weights were successfully loaded.")
            model.load_state_dict(model_dict)
    else:
        print(f"\033[31mNo weights specified.Model {name} will start training from scratch")

    return model

def get_networks_for_ui(name, num_classes, weights_path):
    if name in MODEL_CLASSES:
        model=MODEL_CLASSES[name](num_classes=num_classes)
    else:
        raise ValueError(f"Model {name} not found in MODEL_CLASSES")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path))
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model=nn.DataParallel(model)
    else:
        model=model.to(device)
    return model

if __name__=="__main__":
    # 要加载模型，先在这里测试一下
    model_name = "mobilenet_v2"
    num_classes = 4
    weights = r"./download_pth\mobilenet_v2-b0353104.pth"
    networks = get_networks(model_name, num_classes, weights)
