"""
你可以根据你自己的任务来，自己写一个损失函数，比如diceloss, focalloss等等，
但是对于分类任务来说，这些也就够了
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self,alpha=1, gamma=2, smooth=1e-6, class_weights=None):
        """Focal Loss for multi-class and binary classification tasks"""
        super(FocalLoss,self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.smooth=smooth
        self.class_weights=class_weights
    
    def forward(self,input,labels):
        probs = F.softmax(input,dim=1)  # (batch_size, num_classes)
        log_probs = F.log_softmax(input,dim=1)
        
        class_probs = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
        log_class_probs = log_probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
        focal_factor = (1 - class_probs) ** self.gamma
        focal_loss = -self.alpha*focal_factor*log_class_probs
        
        if self.class_weights is not None:
            focal_loss *= self.class_weights[labels]
        
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, class_weights=None):
        """DiceLoss for multi-class and binary classification tasks."""
        super(DiceLoss,self).__init__()
        self.smooth = smooth
        self.class_weights = class_weights
    
    def forward(self,input,labels):
        probs = F.softmax(input, dim=1)  # (batch_size, num_classes)
        num_classes = probs.shape[1]
        total_loss = 0.0
        
        for i in range(num_classes):
            # Select the probabilities for the current class
            class_probs = probs[:, i]
            
            class_labels = (labels == i).float()
            intersection = torch.sum(class_probs * class_labels)  # Intersection
            union = torch.sum(class_probs) + torch.sum(class_labels)  # Union
            
            dice = (2.*intersection + self.smooth)/(union + self.smooth)
            dice_loss = 1 - dice  # Dice Loss = 1 - Dice Coefficient
            if self.class_weights is not None:
                dice_loss *= self.class_weights[i]
            
            total_loss += dice_loss
        
        return total_loss / num_classes


class Joint2Loss(nn.Module):
    def __init__(self, criterion1, criterion2, alpha=.5, beta=None):
        super(Joint2Loss, self).__init__()
        self.loss_1 = criterion1()
        self.loss_2 = criterion2()
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        
    def forward(self, input, target):
        loss1 = self.loss_1(input, target)
        loss2 = self.loss_2(input, target)
        return loss1 * self.alpha + loss2 * self.beta
        
        
def get_criterion(num_classes=None, alpha=.5, beta=None):
    """
    根据任务类型（二分类或多分类）选择合适的损失函数。
    二分类常用的是BCEWithLogitsLoss，但这个函数的输入参数要求与我们的这个dataset载入的不太一样，所以这里放弃使用
    这里采用的是联合损失，你可以自己随机组合，也可以只使用 celoss
    """
    if num_classes is None:
        criterion = nn.CrossEntropyLoss()
    else:
        if num_classes == 2:
            # criterion = nn.BCEWithLogitsLoss()  # 二分类通常使用 BCEWithLogitsLoss
            criterion = Joint2Loss(nn.CrossEntropyLoss, FocalLoss, alpha=alpha, beta=beta)
        elif num_classes > 2:
            criterion = Joint2Loss(nn.CrossEntropyLoss, DiceLoss, alpha=alpha, beta=beta)  # 多分类使用 CrossEntropyLoss
        else:
            raise ValueError("num_classes must be greater than or equal to 2")
    return criterion

if __name__ == "__main__":
    # 多分类：输出形状 (batch_size, num_classes)
    output=torch.randn(4,4)  # 模拟模型输出 logits (4, 4)，4个样本，每个样本4个类别
    labels=torch.tensor([0,2,1,3])  # 每个样本的标签
    # 计算损失（多分类）
    criterion1=nn.CrossEntropyLoss()
    criterion2=DiceLoss()
    criterion3=FocalLoss()
    criterion4=get_criterion(num_classes=4)
    loss=criterion1(output,labels)  # 计算分类损失
    print("CrossEntropyLoss (多分类) Loss:",loss)
    loss=criterion2(output,labels)  # 计算分类损失
    print("DiceLoss (多分类) Loss:",loss)
    loss=criterion3(output,labels)
    print("FocalLoss (多分类) Loss:",loss)
    loss=criterion4(output,labels)  # 计算分类损失
    print("get_criterion (多分类) Loss:",loss)
    # 二分类：输出形状 (batch_size, 2)
    output=torch.randn(4, 2)
    labels=torch.tensor([0, 1, 1, 1])
    # 计算损失（二分类）
    criterion1=nn.CrossEntropyLoss()
    criterion2=DiceLoss()
    criterion3=FocalLoss()
    criterion4=get_criterion(num_classes=2)
    loss=criterion1(output,labels)
    print("CrossEntropyLoss (二分类) Loss:",loss)
    loss=criterion2(output,labels)  # 计算分类损失
    print("DiceLoss (二分类) Loss:",loss)
    loss=criterion3(output,labels)  # 计算分类损失
    print("FocalLoss (二分类) Loss:",loss)
    loss=criterion4(output,labels)  # 计算分类损失
    print("get_criterion (二分类) Loss:",loss)