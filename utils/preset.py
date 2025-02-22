"""
用于测试在ui中的检测是否正常
"""
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from models import get_networks_for_ui
from natsort import natsorted

from utils import PutRectangleText, ClassificationMetricIndex

flower_data = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
base = ["crack", "rust", "spalling", "stoma"]
cat_dog = ["cats", "dogs"]

target_project_path = r"//"
log_path = os.path.join(target_project_path, "../logs")

input_shape = [224, 224]
categories = flower_data
num_classes = len(categories)
models = r'mobilenet_v2'
weights_path = os.path.join(log_path, "2025_02_11_17_24_54", "weights/best_model.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mindex = ClassificationMetricIndex(num_classes).to(device)


def detect_image_ui(image, img_path, mindex):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = get_networks_for_ui(models, num_classes, weights_path).to(device)
    category = natsorted(categories)
    nw, nh = input_shape
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = image[..., ::-1] / 255.0
    bchw_image = np.expand_dims(np.transpose(new_image, (2, 0, 1)), 0)
    label = -1  # 默认标签为 -1，表示未找到
    for idx, cg in enumerate(category):
        if cg in img_path:
            label = idx
            break
    # 如果 img_path 中没有找到任何类别，则使用模型预测的类别
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
    text = f"{predicted_label}: {predicted_prob:.2f}"
    img = PutRectangleText(img, text)

    return img, text, mindex


if __name__ == "__main__":
    image_path = r"/data/flower_data/test/daisy/169371301_d9b91a2a42.jpg"
    image = cv2.imread(image_path)
    img, text, conf = detect_image_ui(image, image_path, mindex)
    print(text)

    mindex.compute()

    cv2.imshow("test", img)
    cv2.waitKey(0)