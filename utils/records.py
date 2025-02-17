"""
这里主要用于记录训练过程中的一些信息
"""
import os
import torch
import re
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal as signal
from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        try:
            device = 'cpu'
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1]).to(device)
            self.writer.add_graph(model.to(device), dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(f"Epoch {epoch}: train: {loss} val: {val_loss}")
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        losses_cpu = torch.tensor(self.losses).cpu().numpy()
        val_loss_cpu = torch.tensor(self.val_loss).cpu().numpy()

        plt.figure()
        plt.plot(iters, losses_cpu, 'red', linewidth=2, label='train loss')
        plt.plot(iters, val_loss_cpu, 'coral', linewidth=2, label='val loss')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            smoothed_train_loss = signal.savgol_filter(losses_cpu, num, 3)
            smoothed_val_loss = signal.savgol_filter(val_loss_cpu, num, 3)

            plt.plot(iters, smoothed_train_loss, 'green', linestyle='--', linewidth=2, label='smooth train loss')
            plt.plot(iters, smoothed_val_loss, '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

    def close(self):
        self.writer.close()


class AverageMeter(object):
    """A simple class that maintains the running average of a quantity

    Example:
    ```
        loss_avg = AverageMeter()
        loss_avg.update(2)
        loss_avg.update(4)
        loss_avg() = 3
    ```
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class ConsoleLogger:
    def __init__(self, log_file, encoding='utf-8'):
        self.log_file = log_file
        self.clean(self.log_file)
        self.terminal = sys.stdout
        self.encoding = encoding

    def write(self, message):
        message_no_color = self.remove_ansi_colors(message)
        with open(self.log_file, 'a', encoding=self.encoding) as log:
            log.write(message_no_color)

        self.terminal.write(message)

    def clean(self,log_file):
        if os.path.exists(log_file):
            os.remove(log_file)

    def flush(self):
        # 为了兼容一些不支持flush的环境
        self.terminal.flush()

    @staticmethod
    def remove_ansi_colors(text):
        """
        去除 ANSI 颜色代码。
        """
        # 正则表达式匹配 ANSI 转义序列
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)


def redirect_console(log_path='./out.log'):
    """
    将控制台输出重定向到指定文件。

    Args:
        log_path (str): 日志文件的路径。
    """
    logger = ConsoleLogger(log_path)
    sys.stdout = logger  # 将标准输出重定向到 Logger