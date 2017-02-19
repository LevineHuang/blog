---
title: 01-Keras之用MNIST数据集训练一个CNN
date: 2017-01-19 18:23:00
updated	: 2017-01-19 18:23:00
permalink: abc
tags:
- Keras
- Deep Learning
- AI
- MI
categories:
- DeepLearning
- Keras
---

## 01-Keras之用MNIST数据集训练一个CNN
----

#### 模型code
```python
# -*- coding: utf-8 -*-

'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
# Keras的底层库使用Theano或TensorFlow
from keras import backend as K

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 在如何表示一组彩色图片的问题上，Theano和TensorFlow发生了分歧.
# ’th’模式，也即Theano模式会把100张RGB三通道的16×32（高为16宽为32）彩色图表示为下面这种形式（100,3,16,32），Caffe采取的也是这种方式。第0个维度是样本维，代表样本的数目，第1个维度是通道维，代表颜色通道数。后面两个就是高和宽了。
# 而TensorFlow，即’tf’模式的表达形式是（100,16,32,3），即把通道维放在了最后。

# 根据backend模式reshape输入数据
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

# 卷积层    
# 二维卷积层对二维输入进行滑动窗卷积
# keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='th', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)

# nb_filter：卷积核的数目,（即输出的维度）
# nb_row：卷积核的行数
# nb_col：卷积核的列数
# border_mode：边界模式，为“valid”，“same”或“full”，full需要以theano为后端

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

# keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th')
# 空域信号施加最大值池化
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

```
#### 模型运行结果
```sh
Using TensorFlow backend.
X_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Train on 60000 samples, validate on 10000 samples
Epoch 1/12
60000/60000 [==============================] - 46s - loss: 0.3732 - acc: 0.8859 - val_loss: 0.0886 - val_acc: 0.9719
Epoch 2/12
60000/60000 [==============================] - 45s - loss: 0.1350 - acc: 0.9597 - val_loss: 0.0627 - val_acc: 0.9796
Epoch 3/12
60000/60000 [==============================] - 45s - loss: 0.1027 - acc: 0.9697 - val_loss: 0.0562 - val_acc: 0.9822
Epoch 4/12
60000/60000 [==============================] - 45s - loss: 0.0884 - acc: 0.9741 - val_loss: 0.0438 - val_acc: 0.9858
Epoch 5/12
60000/60000 [==============================] - 45s - loss: 0.0779 - acc: 0.9772 - val_loss: 0.0415 - val_acc: 0.9867
Epoch 6/12
60000/60000 [==============================] - 46s - loss: 0.0709 - acc: 0.9786 - val_loss: 0.0379 - val_acc: 0.9869
Epoch 7/12
60000/60000 [==============================] - 45s - loss: 0.0650 - acc: 0.9811 - val_loss: 0.0360 - val_acc: 0.9889
Epoch 8/12
60000/60000 [==============================] - 45s - loss: 0.0609 - acc: 0.9813 - val_loss: 0.0354 - val_acc: 0.9883
Epoch 9/12
60000/60000 [==============================] - 45s - loss: 0.0557 - acc: 0.9838 - val_loss: 0.0330 - val_acc: 0.9885
Epoch 10/12
60000/60000 [==============================] - 45s - loss: 0.0541 - acc: 0.9836 - val_loss: 0.0318 - val_acc: 0.9897
Epoch 11/12
60000/60000 [==============================] - 45s - loss: 0.0497 - acc: 0.9857 - val_loss: 0.0322 - val_acc: 0.9897
Epoch 12/12
60000/60000 [==============================] - 45s - loss: 0.0476 - acc: 0.9856 - val_loss: 0.0327 - val_acc: 0.9893
Test score: 0.0326897691154
Test accuracy: 0.9893

```
