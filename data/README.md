### 标准数据集格式
    data
        base
            -train
                -crack
                -rust
                -spalling
                -stoma
            -val
            -test
标准训练集如上所示，训练集和验证集必须存在，测试集不一定要有。
### 示例数据集
二分类 ：https://www.kaggle.com/datasets/tongpython/cat-and-dog

多分类 ：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/data_set
### 划分脚本

**./data/split_data.py**

修改目标文件夹以及保存路径，运行即可, 测试集比例若为 0 则不划分测试集

    flower_photos                    
        - daisy                                      flower_data
        - dandelion                                      - val
        - roses            --------------------->        - test
        - sunflowers                                     - train
        - tulips                                         



