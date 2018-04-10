## 某场景下目标检测

### 开发背景
该实验是基于四百张红细胞显微图，使用Transfer Learning(迁移学习)和Faster RCNN，在Azure数据科学虚拟机里利用预装的Keras和Tensorflow框架完成红细胞的检测。

### 开发环境
- Windows
- Keras 2.0.9
- Tensorflow 1.4.0
- Anaconda Python 3.5.2 with CUDA 8.0


### 数据说明
训练数据未展示。根据labelImg打标签


### 代码说明

- train_frcnn.py: 模型训练。
   - 通过epoch_length和num_epochs修改训练次数和每一次的训练长度。 

- test_frcnn.py: 测试模型。
   - 将图片放到测试文件夹中。
   - 需要读取训练时保存的config.pickle文件和训练好的模型model.hdf5。

