"""
以下函数均来自于 pyzjr(
    https://github.com/Auorui/pyzjr
    https://pypi.org/project/pyzjr/
),为了防止大家不习惯用我自己自定义的包,所以这里给出了本项目所用的函数,若是还有以后的教程就不再这样做了
"""
import gc
import os
import cv2
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from shutil import rmtree

def load_owned_device():
    """
    Return appropriate device objects based on whether the system supports CUDA.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def release_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
def SeedEvery(seed=11, rank=0):
    """设置随机种子"""
    combined_seed = seed + rank
    random.seed(combined_seed)
    np.random.seed(combined_seed)
    torch.manual_seed(combined_seed)
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        import torch.backends.cuda as cuda
        cuda.matmul.allow_tf32 = True
        cudnn.benchmark = True
        
def show_config(head="Configurations", args=None, **kwargs):
    """显示配置信息"""
    print(f'{head}:')
    print('-' * 113)
    print('|%5s | %45s | %55s|' % ('order', 'keys', 'values'))
    print('-' * 113)
    counter = 0
    if args is not None:
        if isinstance(args, argparse.Namespace):
            config_dict = vars(args)
            for key, value in config_dict.items():
                counter += 1
                print(f'|%5d | %45s | %55s|' % (counter, key, value))
        elif isinstance(args, list):
            for arg in args:
                counter += 1
                print(f'|%5d | %45s | {"": >55}|' % (counter, arg))  # Assuming each element in the list is a key
        else:
            counter += 1
            print(f'|%5d | %45s | %55s|' % (counter, "arg", args))  # Assuming args is a single value

    for key, value in kwargs.items():
        counter += 1
        print(f'|%5d | %45s | %55s|' % (counter, key, value))

    print('-' * 113)
    
def loss_weights_dirs(root_path='./logs', format = "%Y_%m_%d_%H_%M_%S"):
    """训练代码常用路径, 保存loss信息, 以及保存最佳模型"""
    time_str = f"{datetime.now().strftime(format)}"
    time_dir = os.path.join(root_path, time_str)
    loss_log_dir = os.path.join(time_dir, 'loss')
    save_model_dir = os.path.join(time_dir, 'weights')
    multi_makedirs(loss_log_dir, save_model_dir)
    return loss_log_dir, save_model_dir, time_dir

def multi_makedirs(*args):
    """
    为给定的多个路径创建目录, 如果路径不存在, 则创建它
    """
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)

def rm_makedirs(file_path: str):
    # 如果文件夹存在，则先删除原文件夹在重新创建
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

def num_worker(batch_size):
    """
    Determine the number of parallel worker processes used for data loading
    """
    return min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])


def PutRectangleText(image, text, bgcolor=(255, 255, 255), fontcolor=(0, 0, 0), font_scale=1.0, thickness=2, padding=10):
    """
    在图像左上角显示白底黑字的文字。

    Args:
        image (numpy.ndarray): 输入图像。
        text (str): 要显示的文字。
        font_scale (float): 字体大小。
        thickness (int): 字体粗细。
        padding (int): 文字背景的边距。

    Return:
        numpy.ndarray: 添加文字后的图像。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x1 = padding
    y1 = padding
    x2 = x1 + text_width + 2 * padding
    y2 = y1 + text_height + 2 * padding
    cv2.rectangle(image, (x1, y1), (x2, y2), bgcolor, -1)
    cv2.putText(image, text, (x1 + padding, y1 + text_height + padding), font, font_scale, fontcolor, thickness)
    return image