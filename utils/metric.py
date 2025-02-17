"""
这里仅适用于二分类与多分类的指标。不支持多标签任务
                 | Predicted Positive | Predicted Negative |
Actual Positive  |        TP          |         FN         |
Actual Negative  |        FP          |         TN         |
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import numpy as np
from torchmetrics.classification import BinaryConfusionMatrix, MulticlassConfusionMatrix, \
    Accuracy, AUROC, Precision, Recall, JaccardIndex, Specificity, HammingDistance,\
    F1Score, FBetaScore, CohenKappa, MatthewsCorrCoef, CalibrationError

class ConfusionMatrix(nn.Module):
    def __init__(
        self,
        num_classes,
        threshold=.5,
        ignore_index=None,
    ):
        super(ConfusionMatrix, self).__init__()
        if num_classes == 2:
            self.matrix = BinaryConfusionMatrix(threshold=threshold, ignore_index=ignore_index)
        elif num_classes > 2:
            self.matrix = MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
        else:
            raise ValueError(
                "num_classes should be either 2 for binary classification or greater than 2 for multiclass classification.")
        self.matrix.reset()
        self.num_classes = num_classes

    def update(self, preds, target):
        self.matrix.update(preds, target)

    @property
    def get_matrix(self):
        return self.matrix.compute()

    def ravel(self):
        confusion_matrix = self.matrix.compute()
        metrics=np.zeros((4,))
        if self.num_classes > 2:
            for i in range(self.num_classes):
                TP = confusion_matrix[i,i]
                FN = torch.sum(confusion_matrix[i,:]).item() - TP
                FP = torch.sum(confusion_matrix[:,i]).item() - TP
                # 多分类任务中通常不计算TN,因为对于每个类别,其他所有类别都可以被视为“负类”,这使得TN的计算变得复杂且不直观
                TN = torch.sum(confusion_matrix).item() - (TP + FN + FP)
                metrics[0] += TP
                metrics[1] += FN
                metrics[2] += FP
                metrics[3] += TN
        else:
            metrics = confusion_matrix.flatten().numpy()

        return metrics

    def plot_confusion_matrix(self, save_path="./class_confusionmatrix.png"):
        cm = self.get_matrix
        if self.num_classes == 2:
            class_names=['Negative','Positive']
        else:
            class_names=[f'Class {i}' for i in range(self.num_classes)]
        matplotlib.use('TkAgg')
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d',xticklabels=class_names,yticklabels=class_names,cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()

class ClassificationMetricIndex(nn.Module):
    def __init__(self, num_classes, headline='ClassificationMetricIndex'):
        super(ClassificationMetricIndex, self).__init__()
        self.num_classes = num_classes
        self.headline = headline
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        task = "binary" if num_classes == 2 else "multiclass"

        self.accuracy = Accuracy(task=task, num_classes=num_classes)
        self.precision = Precision(task=task, num_classes=num_classes)
        self.recall = Recall(task=task, num_classes=num_classes)
        self.iou = JaccardIndex(task=task, num_classes=num_classes)
        self.mcc = MatthewsCorrCoef(task=task, num_classes=num_classes)
        self.kappa = CohenKappa(task=task, num_classes=num_classes)
        self.f1score = F1Score(task=task, num_classes=num_classes)
        self.fbetascore = FBetaScore(task=task, num_classes=num_classes)
        self.specificity = Specificity(task=task, num_classes=num_classes)
        self.hamming = HammingDistance(task=task, num_classes=num_classes)
        self.auroc = AUROC(task=task, num_classes=num_classes)
        self.calibrationerror = CalibrationError(task=task, num_classes=num_classes)

    def update(self, preds, target):
        if self.num_classes > 2:
            preds = F.one_hot(preds, num_classes=self.num_classes).float()
        self.accuracy.update(preds, target)
        self.precision.update(preds, target)
        self.recall.update(preds, target)
        self.iou.update(preds, target)
        self.mcc.update(preds, target)
        self.kappa.update(preds, target)
        self.f1score.update(preds, target)
        self.fbetascore.update(preds, target)
        self.specificity.update(preds, target)
        self.hamming.update(preds, target)
        self.auroc.update(preds, target)
        self.calibrationerror.update(preds, target)

    def get_index(self):
        metrics_data = {
            'Accuracy': self.accuracy.compute().item(),
            'Precision': self.precision.compute().item(),
            'Recall': self.recall.compute().item(),
            'IOU': self.iou.compute().item(),
            'MCC': self.mcc.compute().item(),
            'Kappa': self.kappa.compute().item(),
            'F1Score': self.f1score.compute().item(),
            'FBetaScore': self.fbetascore.compute().item(),
            'Specificity': self.specificity.compute().item(),
            'HammingDistance': self.hamming.compute().item(),
            'AUROC': self.auroc.compute().item(),
            'CalibrationError': self.calibrationerror.compute().item(),
        }
        return metrics_data

    def compute(self):
        metrics_data = self.get_index()
        format_string=f'{{:^{15}}}'
        num_metrics=len(metrics_data)
        print(f"{self.headline} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print('='*(18*num_metrics))

        for i, metric in enumerate(metrics_data.keys()):
            formatted_metric=format_string.format(metric)
            if i < num_metrics - 1:
                print(formatted_metric,end=' | ')
            else:
                print(formatted_metric)

        for i, value in enumerate(metrics_data.values()):
            formatted_value=format_string.format(f'{value:.5f}')
            if i < num_metrics - 1:
                print(formatted_value,end=' | ')
            else:
                print(formatted_value)
        print('='*(18*num_metrics))

if __name__ == "__main__":

    # 二分类测试
    num_classes_binary = 2
    binary_cm = ConfusionMatrix(num_classes=num_classes_binary)
    preds_binary = torch.rand(100, 1)
    labels_binary = (torch.rand(100) > 0.5).long()  # 随机生成0或1标签
    binary_cm.update(preds_binary.squeeze(), labels_binary)
    TP, FN, FP, TN = binary_cm.ravel()
    print("Binary Confusion Matrix:")
    print(binary_cm.get_matrix)  # 修正：调用 compute() 获取混淆矩阵
    print("TP FN FP TN")
    print(TP, FN, FP, TN)

    mindex = ClassificationMetricIndex(num_classes=2)
    mindex.update(preds_binary.squeeze(), labels_binary)
    print(mindex.get_index())
    mindex.compute()

    # 多分类测试
    num_classes_multiclass = 5
    multiclass_cm = ConfusionMatrix(num_classes=num_classes_multiclass)
    preds_multiclass = torch.randint(0, num_classes_multiclass, (100,))
    labels_multiclass = torch.randint(0, num_classes_multiclass, (100,))
    multiclass_cm.update(preds_multiclass, labels_multiclass)
    TP, FN, FP, TN = multiclass_cm.ravel()
    print("Multiclass Confusion Matrix:")
    print(multiclass_cm.get_matrix)
    print("TP FN FP TN")
    print(TP, FN, FP, TN)

    mindexs = ClassificationMetricIndex(num_classes=5)
    mindexs.update(preds_multiclass, labels_multiclass)
    print(mindexs.get_index())
    mindexs.compute()


