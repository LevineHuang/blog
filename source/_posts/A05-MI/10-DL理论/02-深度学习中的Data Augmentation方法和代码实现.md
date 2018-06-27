---
title: DL之02-深度学习中的Data Augmentation方法
date: 2017-02-23 08:47:00
updated	: 2017-23 08:47:00
permalink: abc
tags:
- Deep Learning
- AI
- MI
categories:
- DeepLearning
---

## DL之02-深度学习中的Data Augmentation方法
----
在深度学习中，为了避免出现过拟合（Overfitting），通常我们需要输入充足的数据量。当数据量不够大时候，常常采用以下几种方法：


1. Data Augmentation：通过平移、 翻转、加噪声等方法从已有数据中创造出一批“新”的数据，人工增加训练集的大小。

2. Regularization：数据量比较小会导致模型过拟合, 使得训练误差很小而测试误差特别大. 通过在Loss Function 后面加上正则项可以抑制过拟合的产生。缺点是引入了一个需要手动调整的hyper-parameter。

3. Dropout：这也是一种正则化手段，不过跟以上不同的是它通过随机将部分神经元的输出置零来实现。详见 http://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

4. Unsupervised Pre-training：用Auto-Encoder或者RBM的卷积形式一层一层地做无监督预训练, 最后加上分类层做有监督的Fine-Tuning。参考 http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.207.1102&rep=rep1&type=pdf

5. Transfer Learning：在某些情况下，训练集的收集可能非常困难或代价高昂。因此，有必要创造出某种高性能学习机（learner），使得它们能够基于从其他领域易于获得的数据上进行训练，并能够在对另一领域的数据进行预测时表现优异。这种方法，就是所谓的迁移学习（transfer learning）。

### 数据增强变换（Data Augmentation Transformation）
#### 数字图像处理中图像几何变换方法：
+ 旋转 | 反射变换(Rotation/reflection): 随机旋转图像一定角度; 改变图像内容的朝向;
+ 翻转变换(flip): 沿着水平或者垂直方向翻转图像;
+ 缩放变换(zoom): 按照一定的比例放大或者缩小图像;
+ 平移变换(shift): 在图像平面上对图像以一定方式进行平移;可以采用随机或人为定义的方式指定平移范围和平移步长, 沿水平或竖直方向进行平移. 改变图像内容的位置;
+ 尺度变换(scale): 对图像按照指定的尺度因子, 进行放大或缩小; 或者参照SIFT特征提取思想, 利用指定的尺度因子对图像滤波构造尺度空间. 改变图像内容的大小或模糊程度;
+ 对比度变换(contrast): 在图像的HSV颜色空间，改变饱和度S和V亮度分量，保持色调H不变. 对每个像素的S和V分量进行指数运算(指数因子在0.25到4之间), 增加光照变化;
+ 噪声扰动(noise): 对图像的每个像素RGB进行随机扰动, 常用的噪声模式是椒盐噪声和高斯噪声;
+ 颜色变换(color): 在训练集像素值的RGB颜色空间进行PCA。

不同的任务背景下, 我们可以通过图像的几何变换, 使用以下一种或多种组合数据增强变换来增加输入数据的量。 几何变换不改变像素值, 而是改变像素所在的位置。 通过Data Augmentation方法扩张了数据集的范围, 作为输入时, 以期待网络学习到更多的图像不变性特征。

#### [Keras中的图像几何变换方法](https://keras.io/preprocessing/image/)
Keras中ImageDataGenerator　实现了大多数上文中提到的图像几何变换方法。如下：

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    dim_ordering=K.image_dim_ordering())

```

##### 参数说明：
+ featurewise_center: Boolean. Set input mean to 0 over the dataset, feature-wise.
+ featurewise_std_normalization: Boolean. Divide inputs by std of the dataset, feature-wise.
+ zca_whitening: Boolean. Apply ZCA whitening.
+ samplewise_std_normalization: Boolean. Divide each input by its std.
+ width_shift_range: Float (fraction of total width). Range for random horizontal shifts.
+ rotation_range: Int. Degree range for random rotations.
+ height_shift_range: Float (fraction of total height). Range for random vertical shifts.
+ shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
+ zoom_range: Float or [lower, upper]. Range for random zoom. If a float,  [lower, upper] = [1-zoom_range, 1+zoom_range].
+ channel_shift_range: Float. Range for random channel shifts.
+ fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}. Points outside the boundaries of the input are filled according to the given mode.
+ cval: Float or Int. Value used for points outside the boundaries when fill_mode = "constant".
+ horizontal_flip: Boolean. Randomly flip inputs horizontally.
+ vertical_flip: Boolean. Randomly flip inputs vertically.
+ rescale: rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation).
+ dim_ordering: One of {"th", "tf"}. "tf" mode means that the images should have shape  (samples, height, width, channels), "th" mode means that the images should have shape  (samples, channels, height, width). It defaults to the image_dim_ordering value found in your Keras config file at  ~/.keras/keras.json. If you never set it, then it will be "tf".

##### 其它方法
+ Label shuffle: 类别不平衡数据的增广，参见海康威视ILSVRC2016的report
