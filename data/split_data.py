"""
参考： https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/data_set
"""
import os
import random
import shutil
from utils.tools import rm_makedirs

# 验证集和测试集划分比例
val_split_rate = 0.1  # 验证集比例
test_split_rate = 0.1  # 测试集比例，若为 0 则不划分测试集

target_path = r'D:\PythonProject\pytorch_classification_Auorui\data\flower_photos'
save_path = r'D:\PythonProject\pytorch_classification_Auorui\data\flower_data'

if __name__ == "__main__":
    # 保证随机可复现
    random.seed(0)

    assert os.path.exists(target_path), "path '{}' does not exist.".format(target_path)

    flower_class = [cla for cla in os.listdir(target_path)
                    if os.path.isdir(os.path.join(target_path, cla))]
    print(flower_class)

    # 建立保存训练集的文件夹
    train_root = os.path.join(save_path, "train")
    rm_makedirs(train_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        rm_makedirs(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(save_path, "val")
    rm_makedirs(val_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        rm_makedirs(os.path.join(val_root, cla))

    if test_split_rate > 0:
        # 建立保存测试集的文件夹
        test_root = os.path.join(save_path, "test")
        rm_makedirs(test_root)
        for cla in flower_class:
            # 建立每个类别对应的文件夹
            rm_makedirs(os.path.join(test_root, cla))

    for cla in flower_class:
        cla_path = os.path.join(target_path, cla)
        images = os.listdir(cla_path)
        num = len(images)

        # 随机采样测试集的索引
        test_count = int(num * test_split_rate)
        val_count = int(num * val_split_rate)

        test_index = random.sample(images, k=test_count) if test_split_rate > 0 else []
        remaining_images = [img for img in images if img not in test_index]
        eval_index = random.sample(remaining_images, k=val_count)

        for index, image in enumerate(images):
            image_path = os.path.join(cla_path, image)
            if image in test_index:
                # 将分配至测试集中的文件复制到相应目录
                new_path = os.path.join(test_root, cla)
                shutil.copy(image_path, new_path)
            elif image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                new_path = os.path.join(val_root, cla)
                shutil.copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                new_path = os.path.join(train_root, cla)
                shutil.copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()

    print("processing done!")
