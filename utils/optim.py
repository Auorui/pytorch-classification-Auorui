"""
可供自己定义 优化器 和 学习率下降策略, 官方实现的有很多, 你也可以自己手写实现
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR, MultiStepLR
from torch.optim.lr_scheduler import _LRScheduler

def get_optimizer(
    network, optimizer_type = 'adam', init_lr = 0.001, momentum = 0.9, weight_decay = 1e-4
):
    """
    返回指定优化器及学习率调度器。

    Args:
        network (torch.nn.Module): 模型网络。
        optimizer_type (str): 优化器类型，支持 'sgd', 'adam', 'adamw'。
        init_lr (float): 初始学习率。
        momentum (float): 动量（SGD专用）。
        weight_decay (float): 权重衰减。

    Returns:
        optimizer, scheduler: 返回优化器和学习率调度器。
    """
    # 选择优化器
    if optimizer_type in ['sgd', 'SGD']:
        optimizer = optim.SGD(network.parameters(), lr = init_lr, momentum = momentum,
                              weight_decay = weight_decay)
    elif optimizer_type in ['adam', 'Adam']:
        optimizer = optim.Adam(network.parameters(), lr = init_lr,
                               weight_decay = weight_decay)
    elif optimizer_type in ['adamw', 'AdamW']:
        optimizer = optim.AdamW(network.parameters(), lr = init_lr,
                                weight_decay = weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_type} is not supported.")
    
    return optimizer


def get_lr(optimizer):
    """获取 lr 学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

class WarmUp(_LRScheduler):
    """
    warmup_training learning rate scheduler
    Args:
        optimizer: optimzier
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def get_lr_scheduler(
    optimizer, scheduler_type='step', total_epochs=100,
    warmup_steps=None, milestones=None, gamma=0.75
):
    """
    根据指定的调度策略返回学习率调度器。

    Args:
        optimizer: 优化器
        scheduler_type: 学习率调度器类型，支持 'step', 'multistep', 'cos', 'warmup'
        total_epochs: 训练总轮数，仅在 'cos' 调度器时使用
        warmup_steps: warmup步数，仅在 'warmup' 调度器时使用
        milestones: 多步下降的关键点列表，仅在 'multistep' 调度器时使用
        gamma: 学习率下降的倍率，默认是 0.1

    Returns:
        scheduler: 学习率调度器
    """
    if scheduler_type == 'step':
        # StepLR: 每隔 step_size epoch 学习率下降一次
        scheduler = StepLR(optimizer, step_size = 30, gamma = gamma)
    elif scheduler_type == 'multistep':
        # MultiStepLR: 在指定的 milestones(epoch 列表) 时学习率下降
        if milestones is None :
            # 如果没有提供 milestones，就平均分配
            milestones = [total_epochs // 5 * i for i in range(1, 5)]
            # print(milestones)
        scheduler = MultiStepLR(optimizer, milestones = milestones, gamma = gamma)
    elif scheduler_type == 'cos':
        # CosineAnnealingLR: 余弦退火调度
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
    elif scheduler_type == 'warmup':
        # Linear warmup: 线性增加学习率，直到达到 warmup_steps
        if warmup_steps is None:
            warmup_steps = int(total_epochs * 0.25)
        if warmup_steps <= 0:
            raise ValueError("For 'warmup' scheduler, 'warmup_steps' must be greater than 0.")
        scheduler = WarmUp(optimizer, warmup_steps)
    else:
        raise ValueError(f"Scheduler type '{scheduler_type}' is not supported.")
    
    return scheduler



if __name__=="__main__":
    class Simpletest(torch.nn.Module) :
        def __init__(self) :
            super(Simpletest,self).__init__()
            self.fc = torch.nn.Linear(10, 2)  # 假设输入是10维，输出是2维
        
        def forward(self,x) :
            return self.fc(x)
    
    model = Simpletest()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    scheduler_types = ['step', 'multistep', 'cos', 'warmup']
    total_epochs = 100
    
    for scheduler_type in scheduler_types:
        print(f"\nTesting {scheduler_type} scheduler:")
        
        scheduler = get_lr_scheduler(
            optimizer,
            scheduler_type = scheduler_type,
            total_epochs = total_epochs,
        )
        
        for epoch in range(total_epochs):
            optimizer.zero_grad()
            optimizer.step()
            scheduler.step()
            current_lr = get_lr(optimizer)
            print(f"Epoch {epoch + 1}/{total_epochs}, Learning Rate: {current_lr:.6f}")
