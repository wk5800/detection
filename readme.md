## 某场景下目标检测

### 开发背景
某场景下的目标检测，希望通过此算法改变现有检测算法的不准确性

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

