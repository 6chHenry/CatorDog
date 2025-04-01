**我的训练方法在Master分支下。**

训练过程分为三步：
1. 基线模型:Base CNN(只有两个Conv-Layer)
2. 改进后的CNN:增加了Dropout层和Batch Normalization
3. ResNet:加上了残差连接，并且让模型更加复杂

最终，ResNet达到了87%的正确率(batch_size=32,epochs=20)，用于在线分类。

训练所用到的数据集为Kaggle提供，🔗[Kaggle: Cat VS Dog](https://www.kaggle.com/competitions/dogs-vs-cats/data)

* IMPROVEDCNN文件暂时无法上传
