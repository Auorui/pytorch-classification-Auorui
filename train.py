import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import (load_owned_device, release_gpu_memory, AverageMeter, get_lr, SeedEvery,
                   show_config, loss_weights_dirs, get_optimizer, get_criterion, num_worker,
                   get_lr_scheduler, LossHistory, ClassificationDataset, redirect_console)
from models import get_networks

def parse_args(known=False):
    parser = argparse.ArgumentParser(description='Classification Train')
    # 选择哪种网络
    parser.add_argument('--model', type=str, default='mobilenet_v2', help='model name')
    # 加载预训练权重，默认None
    parser.add_argument('--resume_training', type=str,
                        default=r"D:\PythonProject\pytorch_classification_Auorui\models\download_pth\mobilenet_v2-b0353104.pth",
                        help="resume training from last checkpoint")
    # 日志文件存放路径
    parser.add_argument('--log_dir',  type=str, default=r'./logs', help='log file path')
    # 数据集路径
    parser.add_argument('--dataset_path', type=str, default=r'D:\PythonProject\pytorch_classification_Auorui\data\flower_data', help='dataset path')
    # 训练轮次epochs次数，默认为100轮
    parser.add_argument('--epochs', type=int, default=100, help='Training rounds')
    # 图片大小
    parser.add_argument('--input_shape', default=[224, 224], help='input image shape')
    # batch_size 批量大小 2 4 8,爆内存就改为1试试
    # 详细可看此篇 : https://blog.csdn.net/m0_62919535/article/details/132725967
    # 试过之后还是不行，那就给你的电脑放个假（关机休息）
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    # 初始学习率
    parser.add_argument('--lr', default=2e-4, help='Initial learning rate')
    # 用于优化器的动量参数，控制梯度更新的方向和速度。
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    # 用于优化器的权重衰减参数，用于抑制权重的过度增长，防止过拟合。
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    # 优化器选择，可选adam、adamw、sgd
    parser.add_argument('--optimizer_type', type=str, default="adam", help='Optimizer selection, optional adam、adamw and sgd')
    # 支持多步骤下降和余弦退火，cos 和 step 和 multistep 和 warmup, 推荐使用 cos
    parser.add_argument('--lr_schedule_type', type=str, default="cos", help='Learning rate descent algorithm')
    # 训练过程中的保存pth文件频率, 不宜太频繁
    parser.add_argument('--freq', type=int, default=40, help='Save PTH file frequency')
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


class ClassificationTrainEpoch():
    """
    用于训练和评估分类模型的工具类
    """
    def __init__(self,
                 model,
                 total_epoch,
                 loss_function,
                 optimizer,
                 lr_scheduler,
                 device=load_owned_device()
    ):
        self.device = device
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.total_epoch = total_epoch
        self.best_val_loss = float('inf')  # 用于跟踪最佳验证损失
        release_gpu_memory()
        
    def train_one_epoch(self, train_loader, epoch):
        # 初始化部分设备还是在cuda上面，但一到这里就变为了cpu，暂时没弄清楚原因。
        self.model.to(self.device) # 不加会有问题
        self.model.train()
        train_losses = AverageMeter()
        train_acc = AverageMeter()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{self.total_epoch}', postfix = dict, mininterval = 0.3) as pbar :
            for batch in train_loader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.loss_function(output, labels)
                acc = self.accuracy_all_classes(output, labels)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                train_losses.update(loss.item())
                train_acc.update(acc, labels.size(0))
                pbar.set_postfix(**{'train_loss': train_losses.avg,
                                    'accuracy': train_acc.avg,
                                    'lr': get_lr(self.optimizer)})
                pbar.update(1)
        return train_losses.avg
    
    def evaluate(self, val_loader, epoch):
        self.model.to(self.device) # 不加会有问题
        self.model.eval()
        val_losses = AverageMeter()
        val_acc = AverageMeter()
        with tqdm(total=len(val_loader), desc=f'Epoch {epoch}/{self.total_epoch}', postfix=dict, mininterval=0.3) as pbar:
            for batch in val_loader:
                images, labels = batch
                with torch.no_grad():
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    output = self.model(images)
                    loss = self.loss_function(output, labels)
                    acc = self.accuracy_all_classes(output, labels)
                    val_losses.update(loss.item())
                    val_acc.update(acc, labels.size(0))

                pbar.set_postfix(**{'val_loss': val_losses.avg,
                                    'accuracy': val_acc.avg})
                pbar.update(1)
                
        return val_losses.avg, val_acc.avg

    def accuracy_all_classes(self,output,label):
        _,pred = output.max(1)
        correct = pred.eq(label).float().sum().item()
        accuracy = correct/label.size(0)
        return accuracy

    def save_model(self, save_dir, epoch, train_loss, val_loss, metric, save_period=20):
        print('Epoch:' + str(epoch) + '/' + str(self.total_epoch))
        print('Total Loss: %.5f || Val Loss: %.5f || acc: %.5f' % (train_loss, val_loss, metric))
        
        # 如果你想要选择其他的指标作为保存的策略 (比如这里提供的acc)，可以直接在此处修改即可
        if epoch % save_period == 0:
            torch.save(self.model.state_dict(), os.path.join(save_dir,
            f"epoch{epoch}_train{train_loss:.2f}_val{val_loss:.2f}.pth"))
            print(f"The {epoch} round model has been saved")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, f"best_model.pth")
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch} with val_loss {val_loss:.2f}")

        if epoch == self.total_epoch:
            last_model_path = os.path.join(save_dir, "last_model.pth")
            torch.save(self.model.state_dict(), last_model_path)
            print(f"Last model saved at epoch {epoch}")

if __name__=="__main__":
    args = parse_args()
    SeedEvery(11)
    loss_log_dir, save_model_dir, timelog_dir=loss_weights_dirs(args.log_dir)
    redirect_console(os.path.join(timelog_dir, 'out.log'))
    show_config(head="Auorui's custom classification training template", args=args)
    train_dataset = ClassificationDataset(args.dataset_path, target_shape=args.input_shape,
                                            is_train=True)
    val_dataset = ClassificationDataset(args.dataset_path, target_shape=args.input_shape,
                                            is_train=False)
    num_classes = train_dataset.num_classes

    network = get_networks(args.model, num_classes, weights=args.resume_training)
    criterion = get_criterion(num_classes=num_classes)
    optimizer = get_optimizer(
        network,
        optimizer_type=args.optimizer_type,
        init_lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    lr_scheduler = get_lr_scheduler(optimizer, args.lr_schedule_type, args.epochs)
    classification = ClassificationTrainEpoch(
        network, total_epoch=args.epochs, loss_function=criterion,
        optimizer=optimizer, lr_scheduler=lr_scheduler
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=False,
                              num_workers=num_worker(args.batch_size),
                              shuffle=True,  drop_last=True)
    # 一般验证集不太多, 所以默认给1, 不计算梯度所以速度也很快
    val_loader = DataLoader(val_dataset, batch_size=1, pin_memory=False,
                            num_workers=num_worker(args.batch_size))
    
    loss_history = LossHistory(loss_log_dir, network, input_shape=args.input_shape)
    for epoch in range(args.epochs):
        epoch = epoch + 1
        train_loss = classification.train_one_epoch(train_loader, epoch)
        val_loss, total_acc = classification.evaluate(val_loader, epoch)

        loss_history.append_loss(epoch, train_loss, val_loss)
        classification.save_model(
            save_dir=save_model_dir,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            save_period=args.freq,
            metric=total_acc,
        )

    loss_history.close()