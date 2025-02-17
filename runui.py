from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

import cv2
import sys
import os
import random
import torch
import torch.nn.functional as F
import qimage2ndarray
from datetime import datetime
import numpy as np
from models import get_networks_for_ui
from natsort import natsorted

from ui_classification.load_image import Ui_ImageClassification
from utils import PutRectangleText, ClassificationMetricIndex, load_owned_device



flower_data = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
base = ["crack", "rust", "spalling", "stoma"]
cat_dog = ["cats", "dogs"]
target_project_path = r"E:/PythonProject/Pytorch_Classification_Auorui/"
log_path = os.path.join(target_project_path, "logs")
#############################################################    修改参数
input_shape = [224, 224]
categories = flower_data
num_classes = len(categories)
models = r'mobilenet_v2'
weights_path = os.path.join(log_path, "2025_02_11_17_24_54", "weights/best_model.pth")
mindex = ClassificationMetricIndex(num_classes).to(load_owned_device())
title_ = "基于Pytorch的花卉分类"
#############################################################

def detect_image_ui(image, img_path, mindex):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = get_networks_for_ui(models, num_classes, weights_path).to(device)
    category = natsorted(categories)
    nw, nh = input_shape
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = image[..., ::-1] / 255.0
    bchw_image = np.expand_dims(np.transpose(new_image, (2, 0, 1)), 0)
    label = -1
    cg_label = None
    for idx, cg in enumerate(category):
        if cg in img_path:
            label = idx
            cg_label = cg
            break
    if label == -1:
        return ValueError(f"Warning: No category found in the path.")
    label = torch.tensor([label]).long().to(device)  # 将 label 移动到与模型相同的设备
    with torch.no_grad():
        images = torch.from_numpy(bchw_image)
        images = images.to(device).float()
        output = network(images)
        _, predicted = output.max(1)
        # print(predicted, label)
        mindex.update(predicted, label)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities).item()
    # print("Output tensor:", predicted_class)
    predicted_label = category[predicted_class]
    predicted_prob = probabilities[0][predicted_class].item()
    img = image.copy()

    if predicted_class == label.item():
        correct_status = "Correct"
    else:
        correct_status = "Incorrect"

    text = f"Predicted: {predicted_label} ({predicted_prob:.2f}) | Actual: {cg_label} | {correct_status}"

    img = PutRectangleText(img, text, font_scale=.4, thickness=1)

    return img, text, mindex

class ImageClassificationWindow(QMainWindow, Ui_ImageClassification):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.PrepParameters()
        self.CallBackFunctions()
        self.mindex = mindex
        self.titlelabel.setText(title_)

        
    def PrepParameters(self):
        self.PFilePath = r" "  # 初始化路径为空
        self.SFilePath = r" "  # 初始化保存路径为空
        self.time_str = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        self.PFilePathLiEd.setText(self.PFilePath)
        self.SFilePathLiEd.setText(self.SFilePath)

        self.pimage_list = []  # 存储图像文件列表
        self.current_image_idx = -1  # 用于追踪当前显示的图像索引
        self.current_image = None  # 用于存储当前显示的图像数据
        self.is_autoplay = False  # 是否启用自动播放标志
        self.is_loop = False  # 是否循环播放标志
        self.is_paused = False  # 是否暂停自动播放标志

        self.plainTextEdit.clear()  # 清除文本框内容

    def CallBackFunctions(self):
        self.PFilePathBt.clicked.connect(self.SetPFilePath)
        self.SFilePathBt.clicked.connect(self.SetSFilePath)
        self.RunBt.clicked.connect(self.start_image_show)  # 运行按钮，开始显示图像
        self.LastBt.clicked.connect(self.show_last_image)  # 上一张按钮
        self.NextBt.clicked.connect(self.show_next_image)  # 下一张按钮
        self.ExitBt.clicked.connect(self.close_application)  # 退出按钮
        self.AutoplaycheckBox.stateChanged.connect(self.toggle_autoplay)  # 连接自动播放勾选框
        self.StopRecoverBt.clicked.connect(self.pause_or_resume_autoplay)  # 连接暂停/恢复按钮
        self.exitaction.triggered.connect(self.close_application)
        self.RecordBt.clicked.connect(self.save_to_txt)

    def SetPFilePath(self):
        dirname = QFileDialog.getExistingDirectory(self, "浏览", '.')
        if dirname:
            self.PFilePathLiEd.setText(dirname)
            self.PFilePath = dirname + '/'
            self.pimage_list=[]
            for root,dirs,files in os.walk(self.PFilePath):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg','.jpeg','.bmp','.gif')):
                        full_path = os.path.join(root, f)
                        self.pimage_list.append(full_path)
            self.current_image_idx = -1  # 重置图像索引
            random.shuffle(self.pimage_list)

    def SetSFilePath(self):
        dirname = QFileDialog.getExistingDirectory(self, "浏览", '.')
        if dirname:
            self.SFilePathLiEd.setText(dirname)
            self.SFilePath = dirname + '/'

    def toggle_autoplay(self, state):
        """根据勾选状态设置是否启用自动播放"""
        self.is_autoplay = state == 2  # 2表示勾选状态
        self.StopRecoverBt.setEnabled(self.is_autoplay)  # 只有启用自动播放时才启用暂停/恢复按钮

    def start_image_show(self):
        if not self.pimage_list:
            QMessageBox.warning(self, '警告', '没有找到图像文件！', QMessageBox.Ok)
            return

        # 如果保存路径为空，弹出警告窗口询问是否继续
        if self.SFilePath == r" ":
            reply = QMessageBox.question(
                self, '保存路径未选择', '保存路径未选择，是否继续显示图像？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return  # 如果用户选择 No，返回不继续

        self.current_image_idx = 0  # 从第一张图像开始显示
        self.show_image(self.current_image_idx)

        if self.is_autoplay:
            # 如果是自动播放模式，弹出对话框确认是否循环播放
            reply = QMessageBox.question(
                self, '循环播放', '是否循环播放图像？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.is_loop = True  # 启用循环播放
            else:
                self.is_loop = False  # 不循环播放

            # 启动自动播放
            self.start_autoplay()

        else:
            # 启用和禁用按钮
            self.update_navigation_buttons()

        # 如果是自动播放并且已经暂停了，点击运行时，确保按钮文本恢复为'暂停'
        if self.is_paused:
            self.StopRecoverBt.setText('暂停')  # 恢复为暂停按钮文本

    def start_autoplay(self):
        """启动自动播放模式"""
        if self.is_paused:
            # 如果当前是暂停状态，直接恢复定时器
            self.is_paused = False
            self.StopRecoverBt.setText('暂停')  # 修改按钮文本为 '暂停'

        self.LastBt.setEnabled(False)  # 禁用上一张按钮
        self.NextBt.setEnabled(False)  # 禁用下一张按钮

        # 使用QTimer定时器进行自动播放
        if not hasattr(self, 'autoplay_timer'):  # 如果定时器不存在，则创建
            self.autoplay_timer = QTimer(self)
            self.autoplay_timer.timeout.connect(self.next_image_in_autoplay)

        self.autoplay_timer.start(1000)  # 每1秒切换一张图像

    def next_image_in_autoplay(self):
        """自动播放下一张图像"""
        if self.is_paused:
            return  # 如果已暂停，不进行任何操作

        if self.current_image_idx < len(self.pimage_list) - 1:
            self.current_image_idx += 1
            self.show_image(self.current_image_idx)
        else:
            if self.is_loop:
                self.current_image_idx = 0  # 如果是循环播放，回到第一张
                self.show_image(self.current_image_idx)
            else:
                self.stop_autoplay()  # 自动播放完成后停止并恢复按钮

    def stop_autoplay(self):
        """停止自动播放"""
        if hasattr(self, 'autoplay_timer'):
            self.autoplay_timer.stop()
        self.update_navigation_buttons()  # 恢复按钮状态

    def pause_or_resume_autoplay(self):
        """暂停或恢复自动播放"""
        if self.is_paused:
            self.is_paused = False
            self.StopRecoverBt.setText('暂停')  # 修改按钮文本为 '暂停'
            self.start_autoplay()  # 恢复播放
        else:
            self.is_paused = True
            self.StopRecoverBt.setText('恢复')  # 修改按钮文本为 '恢复'
            self.autoplay_timer.stop()  # 暂停定时器
            
    def display_info(self, info:str):
        self.plainTextEdit.appendPlainText(info)

    def show_image(self, idx):
        if 0 <= idx < len(self.pimage_list):
            # 获取图像文件路径
            img_path = os.path.join(self.PFilePath, self.pimage_list[idx])
            self.display_info(f"{idx}-{img_path}")
            image = cv2.imread(img_path)
            ################################################################### 指标
            image, text, self.mindex = detect_image_ui(image, img_path, self.mindex)
            # self.mindex.compute()
            index_data = self.mindex.get_index()
            self.accuracyLE.setText(f"{index_data['Accuracy']:.5f}")
            self.precisionLE.setText(f"{index_data['Precision']:.5f}")
            self.recallLE.setText(f"{index_data['Recall']:.5f}")
            self.iouLE.setText(f"{index_data['IOU']:.5f}")
            self.f1LE.setText(f"{index_data['F1Score']:.5f}")
            self.specificityLE.setText(f"{index_data['Specificity']:.5f}")
            self.fbetaLE.setText(f"{index_data['FBetaScore']:.5f}")
            self.mccLE.setText(f"{index_data['MCC']:.5f}")
            self.hammingLE.setText(f"{index_data['HammingDistance']:.5f}")
            self.kappaLE.setText(f"{index_data['Kappa']:.5f}")
            self.calibrationerrorLE.setText(f"{index_data['CalibrationError']:.5f}")
            self.aurocLE.setText(f"{index_data['AUROC']:.5f}")
            ###################################################################
            self.display_info(text)
            label_size = self.OutputLab.size()
            label_width = label_size.width()
            label_height = label_size.height()
            image = cv2.resize(image, (label_width, label_height),interpolation=cv2.INTER_AREA)
            if image is not None:
                # 转换为RGB模式
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.current_image = image
                qimg = qimage2ndarray.array2qimage(rgb_image)
                pixmap = QPixmap.fromImage(qimg)
                self.OutputLab.setPixmap(pixmap)

                # 如果有保存路径，才进行自动保存
                if self.SFilePath != r' ':
                    self.auto_save_image(self.pimage_list[idx])

            else:
                print(f"无法读取图像: {img_path}")

    def auto_save_image(self, image_filename):
        if self.current_image is None:
            return
        save_path = os.path.join(self.SFilePath, self.time_str)
        os.makedirs(save_path, exist_ok=True)
        save_img_path = os.path.join(save_path, os.path.basename(image_filename))

        print(os.path.basename(image_filename), save_img_path)
        cv2.imwrite(save_img_path, self.current_image)
        self.display_info(f"图像已自动保存到: {save_img_path}")
        # print(f"图像已自动保存到: {save_path}")

    def show_last_image(self):
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.show_image(self.current_image_idx)
        self.update_navigation_buttons()

    def show_next_image(self):
        if self.current_image_idx < len(self.pimage_list) - 1:
            self.current_image_idx += 1
            self.show_image(self.current_image_idx)
        self.update_navigation_buttons()

    def save_to_txt(self):
        # 获取plainTextEdit中的内容
        content = self.plainTextEdit.toPlainText()

        if not content:
            QMessageBox.warning(self, "警告", "没有内容可保存！", QMessageBox.Ok)
            return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "保存文本文件", "", "Text Files (*.txt);;All Files (*)",
                                                   options=options)

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
                self.display_info(f"内容已保存到: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"保存失败: {str(e)}", QMessageBox.Ok)

    def update_navigation_buttons(self):
        """更新上一张和下一张按钮的状态"""
        if self.current_image_idx == 0:
            self.LastBt.setEnabled(False)  # 禁用上一张按钮
        else:
            self.LastBt.setEnabled(True)  # 启用上一张按钮

        if self.current_image_idx == len(self.pimage_list) - 1:
            self.NextBt.setEnabled(False)  # 禁用下一张按钮
        else:
            self.NextBt.setEnabled(True)  # 启用下一张按钮

    def close_application(self):
        """关闭应用程序"""
        reply = QMessageBox.question(self, '退出', '您确定要退出程序吗?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            QApplication.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassificationWindow()
    window.show()
    sys.exit(app.exec_())
