# 手写数字识别
## 模块介绍
### core

  - \_\_init__.py
    - 导入同级目录中的函数便于其他模块调用
  - dataset.py
    - 封装了获取、处理数据集操作的API
  - models.py
    - 封装了创建模型用的API
  - photo_utils.py
    - 封装了用于处理用户自己拍摄的照片的API
  - predict.py
    - 封装了用于预测模型的API，提供了少许功能
    - 可以用model.predict代替
    
### data
  - mnist
    - 保存了mnist数据集中测试集的可视化图片
  - my_images
    - 用于(建议)存放用户自己拍摄的图片
  - gen_mnist.py
    - 用于生存mnist文件夹中的图片
  - mnist.npz
    - mnist数据集的原文件
  - weights.h5
    - 上一次训练好的模型权重，可直接导入
  
### mnist_train.py
    用于训练模型
### test.py
    用于测试用户拍摄的手写数字图片
    （见data/my_images文件夹中）
### view.py
    对测试集合中的图片进行可视化的预测
    按空格切换下一张
    按Esc关闭程序