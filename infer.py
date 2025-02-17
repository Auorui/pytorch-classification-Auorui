import os
import random
import torch
import argparse
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from natsort import natsorted
from datetime import datetime

from models.definite_net import get_networks
from utils.tools import load_owned_device, release_gpu_memory, multi_makedirs

flower_data = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
base = ["crack", "rust", "spalling", "stoma"]
cat_dog = ["cats", "dogs"]

def parse_args(known=False):
	parser=argparse.ArgumentParser(description='Classification Infer')
	# 网络模型
	parser.add_argument('--model', default='mobilenet_v2', type=str, help='model name')
	# 检测的图像类别
	parser.add_argument('--categories',
						default=flower_data,
						type=list, help='classification category')
	# 图像的文件夹和单张路径预测
	parser.add_argument('--image_path',default=r'E:\PythonProject\pytorch_classification_Auorui\data\flower_data\test',type=str,
						help='Image path or image folder path')
	# 加载训练好的模型权重
	parser.add_argument('--weights', type=str,
						default=r"E:\PythonProject\pytorch_classification_Auorui\logs\2025_02_11_17_24_54\weights\best_model.pth",
						help="Trained model weights")
	# 图片大小
	parser.add_argument('--input_shape', default=[224, 224],help='input image shape')
	# 结果存储路径
	parser.add_argument('--save_dir',default=r"./result",help='Result storage path')
	
	
	return parser.parse_known_args()[0] if known else parser.parse_args()


class class_infernet(object):
	def __init__(
		self,
		model: str,
		weights: str,
		categories,
		input_shape,
		device=load_owned_device()
	):
		super(class_infernet, self).__init__()
		self.device = device
		self.categories = natsorted(categories)
		self.num_classes = len(self.categories)
		network = get_networks(model, self.num_classes, weights=weights)
		self.network = network.to(device)
		self.network = self.network.eval()
		self.input_shape = input_shape
		release_gpu_memory()
		
	def image_to_bchw(self, image, target_shape):
		h,w=target_shape
		ih,iw=image.shape[:2]
		scale=min(w/iw,h/ih)
		nw=int(iw*scale)
		nh=int(ih*scale)
		resized_image=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_CUBIC)
		new_image=np.full((h,w,3),(128,128,128),dtype=np.uint8)
		top=(h - nh)//2
		left=(w - nw)//2
		new_image[top:top + nh, left:left + nw]=resized_image
		new_image = new_image[..., ::-1] / 255.0
		image_data = np.expand_dims(np.transpose(new_image, (2,0,1)), 0)
		return image_data, ih, iw
	
	def detect_image(self, image):
		bchw_image, origh, origw = self.image_to_bchw(image, self.input_shape)
		with torch.no_grad():
			images = torch.from_numpy(bchw_image)
			images = images.to(self.device).float()
			output = self.network(images)
			probabilities = F.softmax(output, dim=1)
			predicted_class = torch.argmax(probabilities).item()
		# print("Output tensor:", output)
		predicted_label = self.categories[predicted_class]
		predicted_prob = probabilities[0][predicted_class].item()
		return predicted_label, predicted_prob

if __name__=="__main__":
	args = parse_args()
	classnet = class_infernet(
		args.model, args.weights, categories=args.categories, input_shape=args.input_shape
	)
	multi_makedirs(args.save_dir)
	
	if os.path.isfile(args.image_path) and args.image_path.lower().endswith(('.png','.jpg','.jpeg')):
		image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
		predicted_label, predicted_prob = classnet.detect_image(image)
		
		print(f"Predicted class: {predicted_label}")
		print(f"Probabilities:{predicted_prob*100:.2f}%")
		
		matplotlib.use('TkAgg')
		plt.imshow(image)
		plt.title(f"Predicted class: {predicted_label}, Probability: {predicted_prob*100:.2f}%")
		plt.axis('off')
		plt.show()
		
	elif os.path.isdir(args.image_path):
		"""
		如果目录如下所示，传入的image_path应该为 : "./base/test"
		base
			- test
				- crack
				- rust
				- spalling
				- stoma
			- train
			- val
		"""
		all_images=[]
		all_labels=[]
		all_probs=[]
		
		# 遍历文件夹中每个类别的子文件夹
		for category in os.listdir(args.image_path):
			category_path=os.path.join(args.image_path, category)
			
			# 确保是文件夹
			if os.path.isdir(category_path):
				images=[os.path.join(category_path,img) for img in os.listdir(category_path)
						if img.lower().endswith(('.png','.jpg','.jpeg'))]
				all_images.extend(images)
			
		# 随机选择5张图片进行显示
		random.shuffle(all_images)
		images_to_show=all_images[:5]
		images_to_save=all_images[5:]
		
		# 显示随机选择的5张图片
		matplotlib.use('TkAgg')
		for img_path in images_to_show:
			image=cv2.imread(img_path,cv2.IMREAD_COLOR)
			predicted_label,predicted_prob=classnet.detect_image(image)
			
			print(f"Image: {img_path}")
			print(f"Predicted class: {predicted_label}")
			print(f"Probability: {predicted_prob*100:.2f}%")
			print("-"*50)
			
			plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
			plt.title(f"Predicted class: {predicted_label}, Probability: {predicted_prob*100:.2f}%")
			plt.axis('off')
			plt.show()
		
		# 将其余图片的推理结果保存到txt文件
		output_file=os.path.join(args.save_dir, f"inference_results{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.txt")
		with open(output_file,"w") as f:
			for img_path in images_to_save:
				image=cv2.imread(img_path,cv2.IMREAD_COLOR)
				predicted_label,predicted_prob=classnet.detect_image(image)
				
				f.write(f"Image: {img_path}\n")
				f.write(f"Predicted class: {predicted_label}\n")
				f.write(f"Probability: {predicted_prob*100:.2f}%\n")
				f.write("-"*50 + "\n")
		
		print(f"推理结果已保存到: {output_file}")